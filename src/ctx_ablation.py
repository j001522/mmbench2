# ctx_ablation.py
"""
How much does the number of real context frames change hallucination? Offline
ablation on the annotated windows (logs/annotation/<set>), to explain why the
closed-loop planner (plan_mpc_env.py: 4 real context frames, 1 in the first
chunk) hallucinates more than the annotated rollouts (8 frames).

Every annotated window is 24 frames: context 0..7, imagined 8..23. Here the 16
imagined frames stay the same (same targets, same actions) and only the real
context before them changes: frames [8-c, 8) for c in --ctxs. The first kept
frame's action is cleared, as on every training window (parse_batch). Rollout
and one-pass decode as export_annotation_set.py (hallucination_labels.
rollout_windowed, shortcut d, tau_ctx); fresh samples, so c=8 is a new draw of
the annotated rollout, not the frozen one.

Variants (name: ctx, eval_d, tau_ctx):
    c{N}        N real frames, the annotation sampler (d=0.25, tau_ctx 0.1)
    c{N}_plan   N real frames, the planner sampler (d=0.125, tau_ctx 0)
    c{N}_d125   (--split_sampler) d=0.125, tau_ctx 0.1
    c{N}_tau0   (--split_sampler) d=0.25, tau_ctx 0

Per window and variant: mean over the 16 frames of the mask Hausdorff (px)
and the per-task-threshold label (plan_halluc.THRESH). The human labels belong
to the frozen c8 rollout; they are only used for a sanity AUROC of c8.

Run from src/ (GPU):
    python ctx_ablation.py --out ../../logs/ctx_ablation/x
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

import hallucination_labels as HL
from eval_annotations import load_set
from interactive import load_dynamics_from_ckpt, load_tokenizer_from_ckpt
from model import temporal_patchify
from plan_halluc import THRESH, auroc, hausdorff
from train_dynamics import decode_packed_to_frames, make_tau_schedule
from wm_dataset import WMDataset

ROOT = "../../logs/annotation"
CTX_FULL, HOR = 8, 16


def variants(ctxs, split_sampler=False):
    out = []
    for c in ctxs:
        out.append((f"c{c}", c, 0.25, 0.1))
    for c in ctxs:
        out.append((f"c{c}_plan", c, 0.125, 0.0))
    if split_sampler:
        # which half of the planner sampler matters: step size or context noise
        for c in ctxs:
            out.append((f"c{c}_d125", c, 0.125, 0.1))
            out.append((f"c{c}_tau0", c, 0.25, 0.0))
    return out


@torch.no_grad()
def run_set(name, args, models, device):
    tok, tok_info, dyn, dyn_info = models
    sdir = os.path.join(ROOT, name)
    key = list(csv.DictReader(open(os.path.join(sdir, "key.csv"))))
    labels = {s["id"]: s["y"] for s in load_set(name, ROOT)}
    idx_rows = sorted(key, key=lambda r: int(r["idx"]))
    tasks = []
    for r in idx_rows:                                  # dataset task order = ascending idx blocks
        if r["task"] not in tasks:
            tasks.append(r["task"])
    data_dir = idx_rows[0].get("data_dir", "./data/test")
    meta = json.loads(str(np.load(os.path.join(sdir, "data", key[0]["id"] + ".npz"))["meta"]))
    assert int(meta["ctx"]) == CTX_FULL and int(meta["horizon"]) == HOR, meta
    ds = WMDataset(data_dir=data_dir, frames_dir=data_dir.rstrip("/") + "-shards", seq_len=CTX_FULL + HOR,
                   img_size=224, action_dim=16, lang_dim=512, tasks_json="../tasks.json", tasks=tasks,
                   verbose=False)
    indices = [int(r["idx"]) for r in idx_rows]
    by_idx = {int(r["idx"]): r for r in key}
    loader = DataLoader(torch.utils.data.Subset(ds, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=HL.collate_batch)
    H, W, C, patch = (int(tok_info[k]) for k in ("H", "W", "C", "patch"))
    k_max = int(dyn_info["k_max"])
    enc, dec = tok.encoder.eval(), tok.decoder.eval()
    rows, k = [], 0
    for batch in loader:
        frames, actions, act_mask, _r, lang = HL.parse_batch(batch, use_rewards=False, device=device)
        B = frames.shape[0]
        z_btLd, _ = enc(temporal_patchify(frames, patch))
        z_gt = HL.pack_bottleneck_to_spatial(z_btLd, n_spatial=z_btLd.shape[2] // 2, k=2)
        real = (frames[:, CTX_FULL:].clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 1, 3, 4, 2).cpu().numpy()
        for vname, c, d, tau in variants(args.ctxs, args.split_sampler):
            s0 = CTX_FULL - c
            a, m = actions[:, s0:].clone(), act_mask[:, s0:].clone()
            a[:, 0] = 0
            m[:, 0] = 0
            sched = make_tau_schedule(k_max=k_max, schedule="shortcut", d=d)
            for rep in range(args.reps):
                z_pred = HL.rollout_windowed(dyn, z_gt[:, s0:], ctx=c, n_steps=HOR, max_ctx=24, k_max=k_max,
                                             sched=sched, actions=a, act_mask=m, tau_ctx=tau, lang_emb=lang)
                wm = decode_packed_to_frames(dec, z_packed=z_pred, H=H, W=W, C=C, patch=patch,
                                             packing_factor=2)[:, c:]
                wm = (wm.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 1, 3, 4, 2).cpu().numpy()
                for b in range(B):
                    r = by_idx[indices[k + b]]
                    hd = [hausdorff(real[b, t], wm[b, t], r["task"]) for t in range(HOR)]
                    hm = float(np.mean(hd))
                    rows.append({"set": name, "id": r["id"], "task": r["task"], "episode": r["episode"],
                                 "variant": vname, "ctx": c, "eval_d": d, "tau_ctx": tau, "rep": rep,
                                 "hausdorff_mean": hm, "halluc": int(hm >= THRESH[r["task"]]),
                                 "hd_first8": float(np.mean(hd[:8])), "hd_last8": float(np.mean(hd[8:])),
                                 "human": labels.get(r["id"], "")})
        k += B
        print(f"  [{name}] {k}/{len(indices)}", flush=True)
    return rows


def summarize(rows, ctxs, split_sampler=False):
    out = []
    P = out.append
    names = [v[0] for v in variants(ctxs, split_sampler)]
    for split in ("all", "val"):
        P(f"\n== {split} windows ({'dmc_v2 + mw_v1' if split == 'val' else 'all 4 sets'}) ==")
        P(f"{'task':13s} {'variant':9s} {'n':>4s} {'rate':>6s} {'HD med':>7s} {'HD mean':>8s} "
          f"{'first8':>7s} {'last8':>7s} {'AUROC(human)':>12s}")
        for task in THRESH:
            for v in names:
                g = [r for r in rows if r["task"] == task and r["variant"] == v
                     and (split == "all" or r["set"] in ("dmc_v2", "mw_v1"))]
                if not g:
                    continue
                hm = np.array([r["hausdorff_mean"] for r in g])
                hl = [(r["hausdorff_mean"], r["human"]) for r in g if r["human"] != ""]
                au = auroc([x for x, _ in hl], [y for _, y in hl]) if v == f"c{CTX_FULL}" else np.nan
                P(f"{task:13s} {v:9s} {len(g):4d} {np.mean([r['halluc'] for r in g]):6.3f} {np.median(hm):7.2f} "
                  f"{hm.mean():8.2f} {np.mean([r['hd_first8'] for r in g]):7.2f} "
                  f"{np.mean([r['hd_last8'] for r in g]):7.2f} {au:12.3f}")
    # paired: per window, rate difference vs c8 (same targets)
    P("\n== paired vs c8 (all windows): mean label difference, 95% CI bootstrap over episodes ==")
    base = {(r["set"], r["id"], r["rep"]): r for r in rows if r["variant"] == f"c{CTX_FULL}"}
    rng = np.random.default_rng(0)
    for task in THRESH:
        for v in names:
            if v == f"c{CTX_FULL}":
                continue
            diffs = defaultdict(list)
            for r in rows:
                if r["task"] == task and r["variant"] == v:
                    b = base[(r["set"], r["id"], r["rep"])]
                    diffs[(r["set"], r["episode"])].append(r["halluc"] - b["halluc"])
            eps = list(diffs.values())
            mean = np.mean([x for e in eps for x in e])
            bs = [np.mean([x for i in rng.integers(0, len(eps), len(eps)) for x in eps[i]]) for _ in range(1000)]
            P(f"  {task:13s} {v:9s} {mean:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=["dmc_v1", "dmc_v2", "mw_v1", "mw_v2"])
    p.add_argument("--ctxs", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--reps", type=int, default=2, help="rollout samples per window and variant")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_sampler", action="store_true",
                   help="also d=0.125 with tau_ctx 0.1, and d=0.25 with tau_ctx 0")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    meta = json.loads(str(np.load(os.path.join(ROOT, "dmc_v1", "data", "s001.npz"))["meta"])) \
        if os.path.exists(os.path.join(ROOT, "dmc_v1", "data", "s001.npz")) else None
    ck = meta or {"tokenizer_ckpt": "./checkpoints/combined/tokenizer.pt",
                  "dynamics_ckpt": "./checkpoints/combined/dynamics.pt"}
    tok, tok_info = load_tokenizer_from_ckpt(ck["tokenizer_ckpt"], device)
    dyn, _r, _p, dyn_info = load_dynamics_from_ckpt(
        ck["dynamics_ckpt"], device=device, d_bottleneck=int(tok_info["d_bottleneck"]),
        n_latents=int(tok_info["n_latents"]), packing_factor=2)
    dyn.eval()
    rows = []
    for s in args.sets:
        rows += run_set(s, args, (tok, tok_info, dyn, dyn_info), device)
        with open(os.path.join(args.out, "windows.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    s = summarize(rows, args.ctxs, args.split_sampler)
    print(s)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(s + "\n")


if __name__ == "__main__":
    main()
