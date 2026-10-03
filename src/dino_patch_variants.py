# dino_patch_variants.py
"""
DINOv2 patch-score variants for the hallucination metric (OURS, experimental),
including PatchCore-style nearest-neighbour scoring, computed per imagined frame
on frozen annotation sets so they can be compared in eval_annotations.py.

Reference = the tokenizer round trip of the real frame (recon[t]), as in
dino_metrics.py "vs recon": both sides carry the tokenizer's artifacts.

Features (--model, default facebook/dinov2-with-registers-base), per layer in
--layers (hidden_states index; 12 = last block, taken after the final
LayerNorm, i.e. last_hidden_state), raw ("L<k>") or PatchCore's locally aware
version ("L<k>a": 3x3 average over the patch grid, stride 1).

Patch scores, all as cosine distance 1 - cos:
    same    WM patch (i,j) vs ref patch (i,j)                    (dino_metrics.py)
    loc1    min over ref patches within +-1 grid cell             (tolerates small shifts)
    loc2    min over ref patches within +-2 grid cells
    w2r     min over ALL ref patches (PatchCore: "does this WM patch look like
            anything in the real frame?") -> invented content
    r2w     min over ALL WM patches, per ref patch -> missed content
    sym     max(pool(w2r), pool(r2w))
    w2r3    w2r with a 3-frame bank recon[t-1], recon[t], recon[t+1] (timing tolerance)
Per frame, each score map is pooled as mean / top5 (mean of the top 5%) / max
(PatchCore's image score). Column: <feat>_<score>_<pool>, e.g. L12_same_mean,
L9a_w2r_max. Not used from PatchCore: coreset subsampling and density
reweighting (they make huge memory banks tractable; ours is one frame).

Output, per set: <root>/_hidden_metrics/dinovar_<set>/<id>_<task>.csv (one row
per imagined frame, plus the paper's per-frame PSNRs).

Run from src/ (GPU):
    python dino_patch_variants.py --sets dmc_v1 dmc_v2 mw_v1 mw_v2
"""
import argparse
import csv
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from metric_io import load_sequences, paper_dpsnr

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Features:
    def __init__(self, name, layers, res, device):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name, local_files_only=True).to(device).eval()
        self.layers, self.res, self.g, self.device = layers, res, res // 14, device

    @torch.no_grad()
    def __call__(self, frames_u8):
        """(N,H,W,3) uint8 -> {layer: (N, g, g, D) float32 on device}."""
        x = torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bicubic", align_corners=False)
        x = ((x - MEAN) / STD).to(self.device)
        out = self.model(pixel_values=x, output_hidden_states=True)
        n = self.g * self.g
        feats = {}
        for k in self.layers:
            h = out.last_hidden_state if k == len(out.hidden_states) - 1 else out.hidden_states[k]
            feats[k] = h[:, -n:].float().reshape(h.shape[0], self.g, self.g, -1)
        return feats


def aggregate3x3(f):
    """PatchCore's locally aware features: 3x3 average over the grid."""
    x = f.permute(0, 3, 1, 2)
    x = F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
    return x.permute(0, 2, 3, 1)


def window_mask(g, r, device):
    yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    p = torch.stack([yy.flatten(), xx.flatten()], 1).to(device)
    return (p[:, None, :] - p[None, :, :]).abs().max(-1).values <= r      # (g*g, g*g) Chebyshev


def pools(d):
    """(N, P) distances -> {pool: (N,)}."""
    k = max(1, int(round(0.05 * d.shape[1])))
    return {"mean": d.mean(1), "top5": d.topk(k, dim=1).values.mean(1), "max": d.max(1).values}


@torch.no_grad()
def frame_scores(f_wm, f_ref, masks):
    """f_wm, f_ref: (N, g, g, D) for the N imagined frames (aligned in time).
    Returns {score: (N, P) distance maps} plus 'w2r3' using neighbouring ref frames."""
    N, g, _, D = f_wm.shape
    a = F.normalize(f_wm.reshape(N, g * g, D), dim=-1)
    b = F.normalize(f_ref.reshape(N, g * g, D), dim=-1)
    sim = a @ b.transpose(1, 2)                                   # (N, P_wm, P_ref)
    out = {"same": 1 - sim.diagonal(dim1=1, dim2=2)}
    for r, m in masks.items():
        out[f"loc{r}"] = 1 - sim.masked_fill(~m, -2).max(2).values
    out["w2r"] = 1 - sim.max(2).values
    out["r2w"] = 1 - sim.max(1).values
    bank = []
    for t in range(N):                                            # recon[t-1], recon[t], recon[t+1]
        idx = [i for i in (t - 1, t, t + 1) if 0 <= i < N]
        bank.append((a[t] @ b[idx].reshape(-1, D).T).max(1).values)
    out["w2r3"] = 1 - torch.stack(bank)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", default=["dmc_v1", "dmc_v2", "mw_v1", "mw_v2"])
    p.add_argument("--root", default="../../logs/annotation")
    p.add_argument("--model", default="facebook/dinov2-with-registers-base")
    p.add_argument("--layers", nargs="+", type=int, default=[6, 9, 12])
    p.add_argument("--res", type=int, default=224)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fx = Features(args.model, args.layers, args.res, device)
    masks = {r: window_mask(fx.g, r, device) for r in (1, 2)}
    print(f"[dinovar] {args.model} layers {args.layers}, {fx.g}x{fx.g} patches, device {device}", flush=True)
    for s in args.sets:
        out_dir = os.path.join(args.root, "_hidden_metrics", f"dinovar_{s}")
        os.makedirs(out_dir, exist_ok=True)
        seqs = load_sequences(SimpleNamespace(from_npz=os.path.join(args.root, s), ids=None), device)
        for q in seqs:
            ctx = q["ctx"]
            fw, fr = fx(q["wm"][ctx:]), fx(q["recon"][ctx:])
            pw, pc, _ = paper_dpsnr(q["real"], q["wm"], ctx)
            cols = {}
            for k in args.layers:
                for agg in ("", "a"):
                    a, b = (fw[k], fr[k]) if not agg else (aggregate3x3(fw[k]), aggregate3x3(fr[k]))
                    sc = frame_scores(a, b, masks)
                    for name, d in sc.items():
                        for pool, v in pools(d).items():
                            cols[f"L{k}{agg}_{name}_{pool}"] = v.cpu().numpy()
                    pw2r, pr2w = pools(sc["w2r"]), pools(sc["r2w"])
                    for pool in pw2r:
                        cols[f"L{k}{agg}_sym_{pool}"] = torch.maximum(pw2r[pool], pr2w[pool]).cpu().numpy()
            T = q["real"].shape[0]
            with open(os.path.join(out_dir, q["name"] + ".csv"), "w", newline="") as f:
                w = csv.writer(f)
                keys = sorted(cols)
                w.writerow(["t", "paper_psnr_wm", "paper_psnr_copy"] + keys)
                for i, t in enumerate(range(ctx, T)):
                    w.writerow([t, f"{pw[i]:.6g}", f"{pc[i]:.6g}"] + [f"{cols[k][i]:.6g}" for k in keys])
        print(f"[dinovar] {s}: {len(seqs)} sequences -> {out_dir} ({len(cols)} columns)", flush=True)


if __name__ == "__main__":
    main()
