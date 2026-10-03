# metric_io.py
"""
Shared input/reference code for the experimental hallucination metrics
(mask_metrics.py, dino_metrics.py): where the sequences come from, and the
paper's full-frame dPSNR per frame for plotting next to them.

Two sources, both giving per sequence uint8 (T,H,W,3) real / wm / recon:
  rollout   held-out test windows + the paper's rollout
            (hallucination_labels.rollout_pass: ctx 8 + horizon 16, d=0.25,
            tau_ctx 0.1, context + rollout decoded in one pass); recon = the
            one-pass decode of the GT latents
  npz       a frozen export (export_annotation_set.py, data/<id>.npz): use this
            for anything compared with the manual annotations
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch


def add_source_args(p):
    p.add_argument("--from_npz", default=None,
                   help="export dir (with data/<id>.npz) instead of a new rollout")
    p.add_argument("--ids", nargs="*", default=None, help="with --from_npz: only these ids")
    p.add_argument("--tasks", nargs="+", default=["reacher-easy", "cup-catch"])
    p.add_argument("--n_per_task", type=int, default=1)
    p.add_argument("--pick", choices=["random", "motion"], default="motion")
    p.add_argument("--motion_pct", type=float, default=25.0)
    p.add_argument("--exclude_keys", nargs="*",
                   default=["../../logs/annotation/dmc_v1/key.csv", "../../logs/annotation/dmc_v2/key.csv"],
                   help="rollout source: drop windows sharing frames with the sequences in these "
                        "annotation keys (keeps the annotator blind); missing files are ignored")
    p.add_argument("--data_dir", default="./data/test")
    p.add_argument("--frames_dir", default="./data/test-shards")
    p.add_argument("--tasks_json", default="../tasks.json")
    p.add_argument("--tokenizer_ckpt", default="./checkpoints/combined/tokenizer.pt")
    p.add_argument("--dynamics_ckpt", default="./checkpoints/combined/dynamics.pt")
    p.add_argument("--seed", type=int, default=0)
    # the paper's rollout settings (hallucination_labels.py defaults)
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--lang_dim", type=int, default=512)
    p.add_argument("--ctx", type=int, default=8)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--max_ctx", type=int, default=24)
    p.add_argument("--eval_d", type=float, default=0.25)
    p.add_argument("--tau_ctx", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)


def load_sequences(args, device):
    """List of dicts: name, task, idx, ctx, real, wm, recon (uint8 T,H,W,3)."""
    if args.from_npz:
        paths = sorted(glob.glob(os.path.join(args.from_npz, "data", "*.npz")))
        if args.ids:
            keep = set(args.ids)
            paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in keep]
        out = []
        for p in paths:
            d = np.load(p)
            meta = json.loads(str(d["meta"]))
            out.append({"name": f"{meta['id']}_{meta['task']}", "task": meta["task"], "idx": meta["idx"],
                        "ctx": int(meta["ctx"]), "real": d["real"], "wm": d["wm"], "recon": d["recon"]})
        print(f"[source] {len(out)} frozen sequences from {args.from_npz}")
        return out

    from hallucination_labels import motion_indices, rollout_pass, select_indices
    from interactive import load_dynamics_from_ckpt, load_tokenizer_from_ckpt
    from train_dynamics import decode_packed_to_frames
    from wm_dataset import WMDataset

    args.extra_horizon = 0
    args.viz_per_task = 1                          # rollout_pass keeps the u8 frames
    torch.manual_seed(args.seed)
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
    indices = (motion_indices(ds, args.n_per_task, args.motion_pct) if args.pick == "motion"
               else select_indices(ds, args.n_per_task, args.seed))
    keys = [k for k in (args.exclude_keys or []) if os.path.exists(k)]
    if keys:
        ann = defaultdict(list)
        for kp in keys:
            with open(kp) as f:
                for r in csv.DictReader(f):     # starts only mean something within the same split
                    if os.path.normpath(r.get("data_dir", "./data/test")) == os.path.normpath(args.data_dir):
                        ann[r["task"]].append(int(r["start"]))
        keep = []
        for i in indices:
            ti, st = ds._lookup(i)
            if all(abs(st - s) > T for s in ann.get(ds.tasks[ti], [])):
                keep.append(i)
        print(f"[source] excluded {len(indices) - len(keep)} window(s) overlapping {keys}")
        indices = keep
    rows0, z_gt, _a, _m, _l, viz = rollout_pass(args, ds, indices, encoder, decoder, dyn,
                                                tok_info, dyn_info, device)
    H, W, C, patch = (int(tok_info[k]) for k in ("H", "W", "C", "patch"))
    hwc = lambda x: x.permute(0, 2, 3, 1).numpy()
    out = []
    for i, r0 in enumerate(rows0):
        with torch.no_grad():
            rec = decode_packed_to_frames(decoder, z_packed=z_gt[i:i + 1, :T].to(device).float(),
                                          H=H, W=W, C=C, patch=patch,
                                          packing_factor=args.packing_factor)[0]
        out.append({"name": f"{r0['task']}_idx{r0['idx']}", "task": r0["task"], "idx": r0["idx"],
                    "ctx": args.ctx, "real": hwc(viz["real"][i]), "wm": hwc(viz["wm"][i]),
                    "recon": hwc((rec.clamp(0, 1) * 255).round().to(torch.uint8).cpu())})
    return out


def paper_dpsnr(real, wm, ctx):
    """The paper's scene metric on uint8 frames, per horizon frame and per
    sequence: PSNR of the WM and of the repeated last context frame vs real;
    dPSNR = 10 log10(sum mse_copy / sum mse_wm) (pooled over the horizon)."""
    r = real[ctx:].astype(np.float32) / 255.0
    w = wm[ctx:].astype(np.float32) / 255.0
    c = real[ctx - 1].astype(np.float32)[None] / 255.0
    mse_w = ((w - r) ** 2).mean(axis=(1, 2, 3))
    mse_c = ((c - r) ** 2).mean(axis=(1, 2, 3))
    psnr = lambda m: 10 * np.log10(1.0 / np.maximum(m, 1e-12))
    return psnr(mse_w), psnr(mse_c), float(psnr(mse_w.mean()) - psnr(mse_c.mean()))
