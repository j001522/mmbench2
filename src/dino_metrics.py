# dino_metrics.py
"""
DINOv2 patch-feature hallucination metric (OURS, experimental, not in the
paper); a parallel, mask-free alternative to mask_metrics.py. Raw values, to
be thresholded per task (no copy-baseline deltas). If useful it moves into the
[2] section of hallucination_labels.py.

Sequences: held-out test windows with the paper's rollout, or a frozen export
(--from_npz; use that for anything compared with the manual annotations). See
metric_io.py.

Per imagined frame t, DINOv2 patch tokens and, per grid location, the cosine
distance 1 - cos between the WM frame and a reference:
    vs real    reference = real[t]  (original version). Carries the tokenizer's
               own error: flat background patches light up even when the
               dynamics are right. Ceiling curve: d(recon[t], real[t]).
    vs recon   reference = recon[t], the tokenizer round trip of real[t] (fix a):
               both sides carry the tokenizer's artifacts, so what is left is
               mostly the dynamics.
for every model in --dino (base: facebook/dinov2-base; base_reg:
facebook/dinov2-with-registers-base, whose register tokens remove the
high-norm artifact patches DINOv2 puts in flat background). Frames are resized
to --res (multiple of 14; 224 -> 16x16 patches) and ImageNet-normalized.
Per frame the map is summarized as
    mean   over all patches
    fg     over patches overlapping the real or WM foreground
           (dmc_background_mask, DMControl only; nan when there is none)
    p95    95th percentile over patches (localized errors)
Per sequence: mean over the horizon. The paper's full-frame dPSNR is plotted
alongside for reference.

Outputs in --out_dir, per sequence:
    .mp4  simulation | world model, and the four distance maps
          (model x reference) over the real frame, with the curves underneath
    .png  the curves;  .csv  per frame;  _maps.npz  the distance maps
and summary.csv.

Weights come from the local Hugging Face cache ($HF_HOME); compute nodes are
offline, so fetch new models once on a login node.

Run from src/ (GPU):
    python dino_metrics.py --n_per_task 2 --out_dir ../../logs/dino_metrics/x
    python dino_metrics.py --from_npz ../../logs/annotation/dmc_v1 --no_video --out_dir ...
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from metric_io import add_source_args, load_sequences, paper_dpsnr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dmc_background_mask import foreground_mask, preset_for_task  # noqa: E402

HF_NAMES = {"small": "facebook/dinov2-small", "base": "facebook/dinov2-base",
            "large": "facebook/dinov2-large", "base_reg": "facebook/dinov2-with-registers-base"}
REFS = ("real", "recon")
SUMS = ("mean", "fg", "p95")
MODEL_COLORS = ["#2a78d6", "#e8710a", "#7b3fa0", "#1b9e4b"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ------------------------------------------------------------------ DINOv2

class Dino:
    """Patch tokens (N, g, g, D) of uint8 (N,H,W,3) frames (HF transformers)."""

    def __init__(self, name, res, device):
        from transformers import AutoModel
        assert res % 14 == 0, "--res must be a multiple of 14"
        self.name, self.res, self.grid, self.device = name, res, res // 14, device
        self.model = AutoModel.from_pretrained(HF_NAMES[name], local_files_only=True).to(device).eval()
        print(f"[dino] {HF_NAMES[name]}, input {res}px -> {self.grid}x{self.grid} patches")

    @torch.no_grad()
    def __call__(self, frames_u8, batch=32):
        out = []
        for s in range(0, len(frames_u8), batch):
            x = torch.from_numpy(np.ascontiguousarray(frames_u8[s:s + batch])).permute(0, 3, 1, 2).float() / 255.0
            if x.shape[-1] != self.res:
                x = F.interpolate(x, size=(self.res, self.res), mode="bicubic", align_corners=False)
            x = ((x - MEAN) / STD).to(self.device)
            h = self.model(pixel_values=x).last_hidden_state      # (B, 1 [+ registers] + g*g, D)
            tok = h[:, -self.grid * self.grid:]                  # patch tokens come last
            out.append(tok.float().reshape(tok.shape[0], self.grid, self.grid, -1).cpu())
        return torch.cat(out)


def cos_dist(a, b):
    """(N,g,g,D) x2 -> (N,g,g) cosine distance."""
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).numpy()


def patch_fg(mask, grid):
    """(H,W) bool -> (grid,grid) bool: patch overlaps the foreground."""
    H, W = mask.shape
    if H % grid or W % grid:
        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        return (F.adaptive_max_pool2d(m, grid)[0, 0] > 0).numpy()
    return mask.reshape(grid, H // grid, grid, W // grid).any(axis=(1, 3))


# ------------------------------------------------------------------ metrics

def sequence_metrics(seq, dinos, use_fg):
    """Per-frame rows for the imagined frames, a per-sequence dict and the
    distance maps {(model, ref): (T-ctx, g, g)}. Keys: <model>_<ref>_<summary>,
    plus <model>_ceil_<summary> = d(recon, real) (tokenizer-only error)."""
    real, wm, rec, ctx = seq["real"], seq["wm"], seq["recon"], seq["ctx"]
    T = real.shape[0]
    hz = np.arange(ctx, T)
    maps = {}
    for dn in dinos:
        f_real, f_rec, f_wm = dn(real[hz]), dn(rec[hz]), dn(wm[hz])
        maps[(dn.name, "real")] = cos_dist(f_wm, f_real)
        maps[(dn.name, "recon")] = cos_dist(f_wm, f_rec)
        maps[(dn.name, "ceil")] = cos_dist(f_rec, f_real)
    psnr_wm, psnr_copy, dpsnr = paper_dpsnr(real, wm, ctx)
    fg = {}
    if use_fg:
        pr = preset_for_task(seq["task"])
        for dn in dinos:
            fg[dn.grid] = [patch_fg(foreground_mask(real[t], pr) | foreground_mask(wm[t], pr), dn.grid) for t in hz]
    rows = []
    for i, t in enumerate(hz):
        r = {"t": int(t), "paper_psnr_wm": float(psnr_wm[i]), "paper_psnr_copy": float(psnr_copy[i])}
        for (mname, ref), m in maps.items():
            d = m[i]
            g = d.shape[0]
            r[f"{mname}_{ref}_mean"] = float(d.mean())
            r[f"{mname}_{ref}_p95"] = float(np.percentile(d, 95))
            f = fg.get(g)
            r[f"{mname}_{ref}_fg"] = float(d[f[i]].mean()) if (f is not None and f[i].any()) else np.nan
        rows.append(r)
    s = {"paper_delta_psnr": dpsnr}
    for k in rows[0]:
        if k != "t" and not k.startswith("paper_"):
            v = np.array([r[k] for r in rows], float)
            s[k] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    return rows, s, maps


# ------------------------------------------------------------------ figures

def figure(width_px, height_px, dpi, rows, s, models, ctx, T, title=None, fs=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2 = "#52514e"
    with plt.rc_context({"font.size": fs, "axes.titlesize": fs + 1, "legend.fontsize": fs - 1,
                         "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2}):
        fig, ax = plt.subplots(4, 1, figsize=(width_px / dpi, height_px / dpi), dpi=dpi, sharex=True)
        t = np.array([r["t"] for r in rows])
        a = ax[0]
        pw, pc = np.array([r["paper_psnr_wm"] for r in rows]), np.array([r["paper_psnr_copy"] for r in rows])
        a.fill_between(t, pw, pc, where=pw >= pc, interpolate=True, color="#1b9e4b", alpha=0.25, lw=0)
        a.fill_between(t, pw, pc, where=pw < pc, interpolate=True, color="#d32f2f", alpha=0.25, lw=0)
        a.plot(t, pw, color="#2a78d6", lw=1.6, label="world model")
        a.plot(t, pc, color=ink2, lw=1.0, ls="--", label="copy last ctx frame")
        div = s["paper_delta_psnr"] <= 0
        a.set_title(f"Paper: full-frame PSNR vs real (dB)   dPSNR = {s['paper_delta_psnr']:+.2f} dB -> "
                    f"{'DIVERGENT' if div else 'ok'}", loc="left", fontweight="bold",
                    color="#d32f2f" if div else "#1b9e4b")
        a.legend(loc="lower left", frameon=False, ncol=2)
        a.set_ylabel("dB")
        names = {"mean": "mean over patches", "fg": "mean over foreground patches", "p95": "95th pct over patches"}
        for a, summ in zip(ax[1:], SUMS):
            parts = []
            for mi, m in enumerate(models):
                col = MODEL_COLORS[mi % len(MODEL_COLORS)]
                for ref, ls, lw in (("recon", "-", 1.6), ("real", "--", 1.0)):
                    k = f"{m}_{ref}_{summ}"
                    a.plot(t, [r[k] for r in rows], color=col, ls=ls, lw=lw, label=f"{m} vs {ref}")
                    parts.append(f"{m}/{ref} {s[k]:.3f}")
                k = f"{m}_ceil_{summ}"
                a.plot(t, [r[k] for r in rows], color=col, ls=":", lw=1.0, label=f"{m} ceiling (recon vs real)")
            a.set_title(f"DINO 1-cos, {names[summ]}:  " + " | ".join(parts), loc="left")
            a.set_ylim(bottom=0)
        ax[1].legend(loc="upper left", frameon=False, ncol=3)
        for a in ax:
            a.axvspan(-0.5, ctx - 0.5, color="#e6e5e1", zorder=0)
            a.set_xlim(-0.5, T - 0.5)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        ax[0].text(ctx / 2 - 0.5, 0.9, "context", transform=ax[0].get_xaxis_transform(),
                   ha="center", va="top", color=ink2)
        ax[-1].set_xlabel("frame t")
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=fs + 3)
        fig.tight_layout(pad=0.4, h_pad=0.6)
    return fig, ax


def heat_overlay(frame, dmap, vmax):
    """Real frame dimmed to grey with the distance map (nearest-upsampled
    patches, inferno colormap) blended on top."""
    import matplotlib
    H, W, _ = frame.shape
    g = dmap.shape[0]
    up = np.kron(np.clip(dmap / vmax, 0, 1), np.ones((H // g + 1, W // g + 1)))[:H, :W]
    col = matplotlib.colormaps["inferno"](up)[..., :3] * 255
    grey = frame.astype(np.float32).mean(-1, keepdims=True).repeat(3, -1)
    a = 0.25 + 0.6 * up[..., None]
    return ((1 - a) * grey + a * col).clip(0, 255).astype(np.uint8)


def _label(img, texts, font, scale_pos):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for x0, txt in zip(scale_pos, texts):
        b = d.textbbox((x0, 4), txt, font=font)
        d.rectangle((b[0] - 3, b[1] - 2, b[2] + 3, b[3] + 2), fill=(0, 0, 0))
        d.text((x0, 4), txt, fill=(255, 255, 255), font=font)


def write_video(path, seq, rows, s, maps, models, fps, title):
    """Row 1: simulation | world model (2x). Row 2: the model x reference
    distance maps over the real frame (one colour scale for all). Curves below."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from PIL import Image, ImageFont
    real, wm, ctx = seq["real"], seq["wm"], seq["ctx"]
    T, H, W, _ = real.shape
    keys = [(m, ref) for m in models for ref in REFS]
    width = len(keys) * (W + 4) - 4
    width = max(width, 2 * (2 * W + 4))
    vmax = max(max(float(np.percentile(maps[k], 99)) for k in keys), 1e-3)
    fig, ax = figure(width, 620, 100, rows, s, models, ctx, T, title=title)
    bars = [a.axvline(0, color="red", lw=2) for a in ax]
    try:
        font = ImageFont.load_default(size=18)
        small = ImageFont.load_default(size=13)
    except TypeError:
        font = small = ImageFont.load_default()
    white = lambda h, w: np.full((h, w, 3), 255, np.uint8)
    frames = []
    for t in range(T):
        top = np.concatenate([np.repeat(np.repeat(real[t], 2, 0), 2, 1), white(2 * H, 8),
                              np.repeat(np.repeat(wm[t], 2, 0), 2, 1)], axis=1)
        top = Image.fromarray(np.concatenate([top, white(2 * H, width - top.shape[1])], axis=1))
        _label(top, [f"simulation  t={t}/{T - 1}",
                     "world model  " + ("context (recon)" if t < ctx else "imagined")], font, [6, 2 * W + 14])
        tiles = []
        for k in keys:
            tile = (heat_overlay(real[t], maps[k][t - ctx], vmax) if t >= ctx else
                    (real[t].astype(np.float32).mean(-1, keepdims=True).repeat(3, -1) * 0.5).astype(np.uint8))
            tiles += [tile, white(H, 4)]
        mid = np.concatenate(tiles[:-1], axis=1)
        mid = Image.fromarray(np.concatenate([mid, white(H, width - mid.shape[1])], axis=1))
        _label(mid, [f"{m} vs {ref}" for m, ref in keys] + ([f"scale 0..{vmax:.2f}"] if t >= ctx else []),
               small, [i * (W + 4) + 4 for i in range(len(keys))] + [4 + (len(keys) - 1) * (W + 4) + W // 2])
        for b in bars:
            b.set_xdata([t, t])
        fig.canvas.draw()
        panel = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        panel = np.asarray(Image.fromarray(panel).resize((width, panel.shape[0])))
        frames.append(np.concatenate([np.asarray(top), white(6, width), np.asarray(mid), panel], axis=0))
    plt.close(fig)
    if frames[0].shape[0] % 2:
        frames = [np.pad(f, ((0, 1), (0, 0), (0, 0)), constant_values=255) for f in frames]
    if frames[0].shape[1] % 2:
        frames = [np.pad(f, ((0, 0), (0, 1), (0, 0)), constant_values=255) for f in frames]
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", macro_block_size=1, quality=8)


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(p)
    p.add_argument("--dino", nargs="+", choices=list(HF_NAMES), default=["base", "base_reg"])
    p.add_argument("--res", type=int, default=224, help="DINOv2 input size (multiple of 14)")
    p.add_argument("--no_fg", action="store_true", help="skip the foreground-masked summary everywhere")
    p.add_argument("--fg_tasks", nargs="*", default=["reacher-easy", "reacher-hard", "cup-catch", "cup-spin", "mw-reach"],
                   help="tasks the colour foreground mask works on; fg = nan elsewhere")
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--video_top", type=int, default=0,
                   help="write videos only for the K highest and K lowest sequences by --video_key (0 = all)")
    p.add_argument("--video_key", default=None, help="summary key for --video_top (default <first model>_recon_mean)")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinos = [Dino(n, args.res, device) for n in args.dino]           # fail fast before the rollout
    seqs = load_sequences(args, device)
    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    results = []
    for seq in seqs:
        rows, s, maps = sequence_metrics(seq, dinos, use_fg=not args.no_fg and seq["task"] in args.fg_tasks)
        name = seq["name"]
        summary.append({"name": name, "task": seq["task"], "idx": seq["idx"], **s})
        results.append((seq, rows, s, maps))
        with open(os.path.join(args.out_dir, name + ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        np.savez_compressed(os.path.join(args.out_dir, name + "_maps.npz"),
                            **{f"{m}_{r}": v for (m, r), v in maps.items()})
        msg = " | ".join(f"{m}: vs real {s[f'{m}_real_mean']:.3f}, vs recon {s[f'{m}_recon_mean']:.3f}, "
                         f"ceil {s[f'{m}_ceil_mean']:.3f}" for m in args.dino)
        print(f"[{name}] paper dPSNR {s['paper_delta_psnr']:+.2f} | mean 1-cos  {msg}", flush=True)

    # Videos: all, or (--video_top K) only the K highest and K lowest by --video_key,
    # to look at the extremes of a large screened set.
    if not args.no_video:
        import matplotlib.pyplot as plt
        order = list(range(len(results)))
        if args.video_top > 0:
            key = args.video_key or f"{args.dino[0]}_recon_mean"
            order = sorted(order, key=lambda i: -results[i][2][key])
            order = order[:args.video_top] + [i for i in order[-args.video_top:] if i not in order[:args.video_top]]
            print(f"[video] {len(order)} of {len(results)} sequences: top/bottom {args.video_top} by {key}")
        for rank, i in enumerate(order):
            seq, rows, s, maps = results[i]
            tag = "" if args.video_top <= 0 else ("high_" if rank < args.video_top else "low_")
            name = tag + seq["name"]
            title = f"{seq['name']}  (DINOv2 {', '.join(args.dino)}; {dinos[0].grid}x{dinos[0].grid} patches)"
            fig, _ = figure(1300, 860, 110, rows, s, args.dino, seq["ctx"], seq["real"].shape[0],
                            title=title, fs=8)
            fig.savefig(os.path.join(args.out_dir, name + ".png"))
            plt.close(fig)
            write_video(os.path.join(args.out_dir, name + ".mp4"), seq, rows, s, maps, args.dino,
                        args.fps, title)

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"[out] {args.out_dir}")


if __name__ == "__main__":
    main()
