# paper_predictors.py
"""
The paper's three hallucination predictors (u_r, u_f, u_s, raw and
motion-normalized) on the frozen annotation sets, for comparison with the
manual labels in eval_annotations.py.

The predictors live in the model (latents, denoising runs), and only decoded
frames were frozen at export time, so each set's rollout is first REPRODUCED:
same seed, same window order and batching, same code path as
export_annotation_set.py (hallucination_labels.rollout_pass), with the
predicted latents captured. The decoded frames are compared with the frozen
ones (exact uint8 match reported per set); the predictors are only meaningful
for sequences that reproduce.

Then, along each reproduced trajectory, per imagined step t (context = GT
context latents + the rollout's own predictions up to t-1, the rollout's
actions, the paper's sampler: shortcut d=0.25, tau_ctx 0.1):
    u_r     RMS(z_t - enc(dec(z_t)))                      uncertainty.URNormScorer
    u_f     flow instability: mean RMS change of the denoiser's x1_hat over the
            second half of the Euler steps, mean over seeds
                                                          interactive.sample_one_timestep_packed
    u_s     variance over --n_seeds denoising seeds of the next latent from the
            same context and action, mean over elements    uncertainty.CrossSeedScorer
    motion  RMS(z_t - z_{t-1}) (predicted latent step)
    u_*_norm = u_* / max(motion, 1e-3)  (the paper's motion normalization)

Output: <root>/_hidden_metrics/pred_<set>/<id>_<task>.csv (one row per
imagined frame) and <root>/_hidden_metrics/pred_<set>/reproduction.csv.

Run from src/ (GPU):
    python paper_predictors.py --sets dmc_v1 dmc_v2 mw_v1 mw_v2
"""
import argparse
import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

import hallucination_labels as HL
from interactive import load_dynamics_from_ckpt, load_tokenizer_from_ckpt
from interactive import sample_one_timestep_packed as sample_with_instability
from metric_io import paper_dpsnr
from train_dynamics import make_tau_schedule
from uncertainty import URNormScorer
from wm_dataset import WMDataset


def reproduce(set_dir, device):
    """Re-run the export's rollout; returns (rows0, z_pred (N,T,S,D), actions, masks, lang, frames, key, meta)."""
    key = list(csv.DictReader(open(os.path.join(set_dir, "key.csv"))))
    meta = json.loads(str(np.load(os.path.join(set_dir, "data", key[0]["id"] + ".npz"))["meta"]))
    idx_rows = sorted(key, key=lambda r: int(r["idx"]))
    tasks = []
    for r in idx_rows:                                  # dataset task order = ascending idx blocks
        if r["task"] not in tasks:
            tasks.append(r["task"])
    data_dir = idx_rows[0].get("data_dir", "./data/test")
    a = SimpleNamespace(ctx=meta["ctx"], horizon=meta["horizon"], extra_horizon=0, max_ctx=24,
                        eval_d=meta["eval_d"], tau_ctx=meta["tau_ctx"], batch_size=16, num_workers=4,
                        packing_factor=2, viz_per_task=1)
    torch.manual_seed(int(meta["seed"]))                # same point as in export_annotation_set.main
    tok, tok_info = load_tokenizer_from_ckpt(meta["tokenizer_ckpt"], device)
    dyn, _r, _p, dyn_info = load_dynamics_from_ckpt(
        meta["dynamics_ckpt"], device=device, d_bottleneck=int(tok_info["d_bottleneck"]),
        n_latents=int(tok_info["n_latents"]), packing_factor=2)
    encoder, decoder = tok.encoder.eval(), tok.decoder.eval()
    dyn.eval()
    T = a.ctx + a.horizon
    ds = WMDataset(data_dir=data_dir, frames_dir=data_dir.rstrip("/") + "-shards", seq_len=T, img_size=224,
                   action_dim=16, lang_dim=512, tasks_json="../tasks.json", tasks=tasks, verbose=False)
    indices = [int(r["idx"]) for r in idx_rows]
    captured = []
    orig = HL.rollout_windowed

    def capture(*args, **kw):
        z = orig(*args, **kw)
        captured.append(z.detach().float().cpu())
        return z

    HL.rollout_windowed = capture
    try:
        rows0, _z_gt, a_all, m_all, l_all, viz = HL.rollout_pass(a, ds, indices, encoder, decoder, dyn,
                                                                 tok_info, dyn_info, device)
    finally:
        HL.rollout_windowed = orig
    return dict(rows0=rows0, z=torch.cat(captured), act=a_all, mask=m_all, lang=l_all, wm=viz["wm"],
                real=viz["real"], key={int(r["idx"]): r for r in key}, a=a, tok=tok, tok_info=tok_info,
                dyn=dyn, dyn_info=dyn_info)


@torch.no_grad()
def predictors(R, device, n_seeds, chunk):
    """Per sequence and imagined step: u_r, u_f, u_s, motion (+ normalized)."""
    a, dyn, tok_info = R["a"], R["dyn"], R["tok_info"]
    k_max = int(R["dyn_info"]["k_max"])
    sched = make_tau_schedule(k_max=k_max, schedule="shortcut", d=a.eval_d)
    S = R["z"].shape[2]
    scorer = URNormScorer(R["tok"].encoder, R["tok"].decoder, patch=int(tok_info["patch"]), packing_factor=2,
                          n_spatial=S, H=int(tok_info["H"]), W=int(tok_info["W"]), C=int(tok_info["C"]))
    N, T = R["z"].shape[:2]
    out = {k: torch.zeros(N, T - a.ctx) for k in ("u_r", "u_f", "u_s", "motion")}
    for c0 in range(0, N, chunk):
        sl = slice(c0, min(c0 + chunk, N))
        z = R["z"][sl].to(device)
        act, msk, lang = R["act"][sl].to(device), R["mask"][sl].to(device), R["lang"][sl].to(device)
        b = z.shape[0]
        for j, t in enumerate(range(a.ctx, T)):
            t0 = max(0, t - (a.max_ctx - 1))
            a_win, m_win = HL._window_actions(act, msk, t0, t + 1)
            rep = lambda x: x.repeat_interleave(n_seeds, dim=0)
            zn, inst = sample_with_instability(
                dyn, past_packed=rep(z[:, t0:t]), k_max=k_max, sched=sched, actions=rep(a_win),
                act_mask=rep(m_win), use_amp=True, tau_ctx=a.tau_ctx, lang_emb=rep(lang),
                per_sample_instability=True)
            zn = zn.float().reshape(b, n_seeds, *zn.shape[1:])
            out["u_f"][sl, j] = inst.float().reshape(b, n_seeds).mean(1).cpu()
            out["u_s"][sl, j] = zn.var(dim=1).mean(dim=(1, 2)).cpu()
            comp = scorer.score_components(z[:, t:t + 1], z[:, t - 1])
            out["u_r"][sl, j] = comp["u_r"].cpu()
            out["motion"][sl, j] = comp["motion"].cpu()
        print(f"  [predictors] {sl.stop}/{N}", flush=True)
    m = out["motion"].clamp(min=1e-3)
    for k in ("u_r", "u_f", "u_s"):
        out[k + "_norm"] = out[k] / m
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", default=["dmc_v1", "dmc_v2", "mw_v1", "mw_v2"])
    p.add_argument("--root", default="../../logs/annotation")
    p.add_argument("--n_seeds", type=int, default=4)
    p.add_argument("--chunk", type=int, default=32, help="sequences per batch in the predictor pass")
    args = p.parse_args()
    device = torch.device("cuda")

    for s in args.sets:
        set_dir = os.path.join(args.root, s)
        out_dir = os.path.join(args.root, "_hidden_metrics", f"pred_{s}")
        os.makedirs(out_dir, exist_ok=True)
        R = reproduce(set_dir, device)
        rep_rows, ok = [], []
        for i, r0 in enumerate(R["rows0"]):
            k = R["key"][int(r0["idx"])]
            frozen = np.load(os.path.join(set_dir, "data", k["id"] + ".npz"))
            wm_now = R["wm"][i].permute(0, 2, 3, 1).numpy()
            same = float((wm_now == frozen["wm"]).mean())
            mse = float(((wm_now.astype(np.float32) - frozen["wm"].astype(np.float32)) / 255.0).__pow__(2).mean())
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            rep_rows.append({"id": k["id"], "task": k["task"], "exact_pixel_fraction": f"{same:.6f}",
                             "psnr_vs_frozen_db": f"{psnr:.2f}"})
            ok.append(same)
        with open(os.path.join(out_dir, "reproduction.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
            w.writeheader()
            w.writerows(rep_rows)
        ok = np.array(ok)
        print(f"[{s}] reproduction: {int((ok == 1).sum())}/{len(ok)} sequences pixel-exact, "
              f"min exact fraction {ok.min():.4f}", flush=True)

        P = predictors(R, device, args.n_seeds, args.chunk)
        ctx = R["a"].ctx
        for i, r0 in enumerate(R["rows0"]):
            k = R["key"][int(r0["idx"])]
            frozen = np.load(os.path.join(set_dir, "data", k["id"] + ".npz"))
            pw, pc, _ = paper_dpsnr(frozen["real"], frozen["wm"], ctx)
            cols = sorted(P)
            with open(os.path.join(out_dir, f"{k['id']}_{k['task']}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "paper_psnr_wm", "paper_psnr_copy"] + cols)
                for j in range(P["u_r"].shape[1]):
                    w.writerow([ctx + j, f"{pw[j]:.6g}", f"{pc[j]:.6g}"] + [f"{float(P[c][i, j]):.6g}" for c in cols])
        print(f"[{s}] predictors -> {out_dir}", flush=True)
        del R
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
