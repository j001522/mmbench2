# export_annotation_set.py
"""
Export held-out test sequences for MANUAL hallucination annotation, with a
local HTML annotator (dreamer4/annotation/annotate.html).

Sequences: the paper's pool and length -- held-out test data (./data/test),
24-frame windows (ctx 8 + horizon 16), NON-OVERLAPPING: no two picked windows
of a task share a frame (|start_i - start_j| > seq_len on the task's frame
axis; windows never cross episodes), nor with the sets in --exclude_keys.
    --pick random   uniform per task, as hallucination_labels.select_indices
                    (the paper's sampling). Test episodes spend most of their
                    time at the goal, so these are mostly static windows with
                    almost no hallucination: use for prevalence only.
    --pick dynamic  uniform over the windows whose ground-truth motion in the
                    imagined part is in the task's top --dynamic_top fraction
                    (real frames only, no model or metric involved, so the
                    selection cannot favour any metric). Use for calibration.

Rollout: exactly the paper's, via hallucination_labels.rollout_pass (shortcut
d=0.25, tau_ctx 0.1, context + rollout decoded in one pass). The rollout is
stochastic, so it is done ONCE here and frozen: every metric that is later
compared with the annotations must be computed from the saved frames
(data/<id>.npz), not from a new rollout.

Output, in --out_dir:
    annotate.html            the annotator (open it in a browser, no server needed)
    manifest.js              sequence list for the annotator (no task, no metrics)
    frames/<id>/gt_NN.png    simulation frames (native 224 px, lossless)
    frames/<id>/wm_NN.png    world-model frames (reconstruction for t < ctx)
    data/<id>.npz            real, wm, recon (uint8 T,H,W,3), actions, act_mask, meta
    key.csv                  id -> task, dataset idx, start, episode (do not open
                             while annotating: ids are shuffled across tasks on purpose)

Run from src/ (GPU):
    python export_annotation_set.py --tasks reacher-easy cup-catch --n_per_task 50 \
        --out_dir ../../logs/annotation/dmc_v1
"""
import argparse
import csv
import json
import os
import shutil

import imageio.v2 as imageio
import numpy as np
import torch

from hallucination_labels import rollout_pass
from interactive import load_dynamics_from_ckpt, load_tokenizer_from_ckpt
from train_dynamics import decode_packed_to_frames
from wm_dataset import WMDataset

TEMPLATE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "annotation", "annotate.html"))


def horizon_motion(ds: WMDataset, ti: int, ctx: int):
    """Per valid window of task ti: mean ground-truth motion over the imagined
    part (MSE between consecutive real frames, 4x downsampled, frames
    start+ctx-1 -> start+T-1). Real frames only, no model."""
    n_frames = int(ds.seg_cum_frames[ti][-1])
    small = torch.cat([ds._get_frames(ti, s, min(2048, n_frames - s))[:, :, ::4, ::4]
                       for s in range(0, n_frames, 2048)]).float()
    m = (small[1:] - small[:-1]).pow(2).mean(dim=(1, 2, 3)).numpy() / 255.0 ** 2   # m[i]: i -> i+1
    cs = np.concatenate([[0.0], np.cumsum(m)])
    starts = ds.valid_starts[ti].numpy()
    a, b = starts + ctx - 1, starts + ds.T - 1                    # transitions a .. b-1
    return (cs[b] - cs[a]) / (b - a)


def nonoverlapping_indices(ds: WMDataset, n_per_task: int, seed: int, *, ctx: int,
                           pick: str = "random", dynamic_top: float = 0.2, exclude=None):
    """Up to n_per_task windows per task, no two sharing a frame, none sharing
    a frame with `exclude` ({task: [starts]}, e.g. an earlier export).
      random   uniform over the task's valid windows (the paper's sampling)
      dynamic  uniform over the windows whose ground-truth motion in the
               imagined part is in the task's top `dynamic_top` fraction
    Returns (global dataset indices, {global idx: motion score})."""
    rng = np.random.default_rng(seed)
    out, score_of, lo = [], {}, 0
    for ti, hi in enumerate(ds._cum_counts):
        starts = ds.valid_starts[ti].numpy()
        motion = horizon_motion(ds, ti, ctx)
        cand = np.arange(len(starts))
        if pick == "dynamic":
            thr = np.quantile(motion, 1.0 - dynamic_top)
            cand = cand[motion >= thr]
            q = np.quantile(motion, [0.25, 0.5, 0.75, 0.9])
            print(f"[select] {ds.tasks[ti]}: horizon motion p25/50/75/90 = "
                  + "/".join(f"{v:.1e}" for v in q)
                  + f"; keeping {len(cand)}/{len(starts)} windows >= {thr:.1e}")
        blocked = list((exclude or {}).get(ds.tasks[ti], []))
        picked = []                                   # local window indices
        for j in rng.permutation(cand):
            s = int(starts[j])
            if all(abs(s - int(starts[p])) > ds.T for p in picked) and all(abs(s - b) > ds.T for b in blocked):
                picked.append(int(j))
                if len(picked) >= n_per_task:
                    break
        if len(picked) < n_per_task:
            print(f"[select] WARNING {ds.tasks[ti]}: only {len(picked)} non-overlapping windows")
        for j in sorted(picked):
            out.append(lo + j)
            score_of[lo + j] = float(motion[j])
        lo = hi
    return out, score_of


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["reacher-easy", "cup-catch"])
    p.add_argument("--n_per_task", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="window selection, rollout noise and id shuffling")
    p.add_argument("--pick", choices=["random", "dynamic"], default="random",
                   help="random: the paper's uniform sampling (mostly static goal-reached windows); "
                        "dynamic: only windows with the most ground-truth motion in the imagined part")
    p.add_argument("--dynamic_top", type=float, default=0.2,
                   help="--pick dynamic: fraction of each task's windows (by horizon motion) eligible")
    p.add_argument("--exclude_keys", nargs="*", default=[],
                   help="key.csv of earlier exports: never share a frame with their sequences")
    p.add_argument("--exclude_idx", nargs="*", type=int, default=[],
                   help="dataset indices (same --tasks order and length) of windows already shown "
                        "to the annotator elsewhere, e.g. in metric videos; never shares a frame with them")
    p.add_argument("--id_prefix", default="s", help="sequence id prefix (keep ids unique across sets)")
    p.add_argument("--data_dir", default="./data/test")
    p.add_argument("--frames_dir", default="./data/test-shards")
    p.add_argument("--tasks_json", default="../tasks.json")
    p.add_argument("--tokenizer_ckpt", default="./checkpoints/combined/tokenizer.pt")
    p.add_argument("--dynamics_ckpt", default="./checkpoints/combined/dynamics.pt")
    p.add_argument("--out_dir", required=True)
    # the paper's rollout settings (hallucination_labels.py defaults)
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--lang_dim", type=int, default=512)
    p.add_argument("--ctx", type=int, default=8)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--max_ctx", type=int, default=24)
    p.add_argument("--eval_d", type=float, default=0.25)
    p.add_argument("--tau_ctx", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    args.extra_horizon = 0
    args.viz_per_task = 1                          # rollout_pass keeps the u8 frames

    if os.path.exists(os.path.join(args.out_dir, "manifest.js")):
        raise SystemExit(f"{args.out_dir} already holds an export; refusing to overwrite "
                         "(annotations refer to these exact frames)")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, tok_info = load_tokenizer_from_ckpt(args.tokenizer_ckpt, device)
    dyn, _r, _p, dyn_info = load_dynamics_from_ckpt(
        args.dynamics_ckpt, device=device, d_bottleneck=int(tok_info["d_bottleneck"]),
        n_latents=int(tok_info["n_latents"]), packing_factor=args.packing_factor)
    encoder, decoder = tok.encoder.eval(), tok.decoder.eval()
    dyn.eval()

    T = args.ctx + args.horizon
    ds = WMDataset(data_dir=args.data_dir, frames_dir=args.frames_dir, seq_len=T, img_size=224,
                   action_dim=16, lang_dim=args.lang_dim, tasks_json=args.tasks_json,
                   tasks=args.tasks, verbose=True)
    exclude = {}
    for kp in args.exclude_keys:            # starts only mean something within the same split
        with open(kp) as f:
            for r in csv.DictReader(f):
                if os.path.normpath(r.get("data_dir", "./data/test")) == os.path.normpath(args.data_dir):
                    exclude.setdefault(r["task"], []).append(int(r["start"]))
    for gi in args.exclude_idx:
        ti, st = ds._lookup(gi)
        exclude.setdefault(ds.tasks[ti], []).append(int(st))
    indices, motion_of = nonoverlapping_indices(ds, args.n_per_task, args.seed, ctx=args.ctx, pick=args.pick,
                                                dynamic_top=args.dynamic_top, exclude=exclude)
    print(f"[select] {len(indices)} non-overlapping windows over {ds.num_tasks} tasks (pick={args.pick})")
    rows0, z_gt, a_all, m_all, _l, viz = rollout_pass(args, ds, indices, encoder, decoder, dyn,
                                                      tok_info, dyn_info, device)

    # Anonymous, shuffled ids: the annotator sees tasks mixed, in random order.
    order = np.random.default_rng(args.seed + 1).permutation(len(rows0))
    ids = {int(i): f"{args.id_prefix}{k + 1:03d}" for k, i in enumerate(order)}

    H, W, C, patch = (int(tok_info[k]) for k in ("H", "W", "C", "patch"))
    hwc = lambda x: x.permute(0, 2, 3, 1).numpy()
    os.makedirs(os.path.join(args.out_dir, "data"), exist_ok=True)
    key = []
    for i, r0 in enumerate(rows0):
        sid = ids[i]
        real, wm = hwc(viz["real"][i]), hwc(viz["wm"][i])
        with torch.no_grad():                      # tokenizer ceiling, same one-pass decode
            rec = decode_packed_to_frames(decoder, z_packed=z_gt[i:i + 1, :T].to(device).float(), H=H, W=W, C=C,
                                          patch=patch, packing_factor=args.packing_factor)[0]
        rec = hwc((rec.clamp(0, 1) * 255).round().to(torch.uint8).cpu())
        ti, start = ds._lookup(r0["idx"])
        episode = int(ds.ep[ti][start])
        meta = {"id": sid, "task": r0["task"], "idx": int(r0["idx"]), "start": int(start),
                "episode": episode, "data_dir": args.data_dir, "pick": args.pick,
                "horizon_motion": motion_of[int(r0["idx"])],
                "ctx": args.ctx, "horizon": args.horizon, "seed": args.seed,
                "eval_d": args.eval_d, "tau_ctx": args.tau_ctx,
                "tokenizer_ckpt": os.path.abspath(args.tokenizer_ckpt),
                "dynamics_ckpt": os.path.abspath(args.dynamics_ckpt)}
        np.savez_compressed(os.path.join(args.out_dir, "data", f"{sid}.npz"), real=real, wm=wm, recon=rec,
                            actions=a_all[i, :T].numpy(), act_mask=m_all[i, :T].numpy(),
                            meta=json.dumps(meta))
        fdir = os.path.join(args.out_dir, "frames", sid)
        os.makedirs(fdir, exist_ok=True)
        for t in range(T):
            imageio.imwrite(os.path.join(fdir, f"gt_{t:02d}.png"), real[t])
            imageio.imwrite(os.path.join(fdir, f"wm_{t:02d}.png"), wm[t])
        key.append({k: meta[k] for k in ("id", "task", "idx", "start", "episode", "data_dir", "pick",
                                         "horizon_motion")})
        print(f"  [export] {sid} <- {r0['task']} idx {r0['idx']} start {start} ep {episode}", flush=True)

    key.sort(key=lambda r: r["id"])
    with open(os.path.join(args.out_dir, "key.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0].keys()))
        w.writeheader()
        w.writerows(key)
    manifest = {"set": os.path.basename(os.path.normpath(args.out_dir)), "ctx": args.ctx,
                "T": T, "size": [H, W], "seqs": [r["id"] for r in key]}
    with open(os.path.join(args.out_dir, "manifest.js"), "w") as f:
        f.write("window.MANIFEST = " + json.dumps(manifest) + ";\n")
    shutil.copy(TEMPLATE, os.path.join(args.out_dir, "annotate.html"))
    print(f"[out] {args.out_dir}: {len(key)} sequences; open annotate.html in a browser")


if __name__ == "__main__":
    main()
