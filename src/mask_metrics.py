# mask_metrics.py
"""
Foreground-mask hallucination metrics (OURS, experimental, not in the paper),
DMControl and Meta-World tasks (the mask is a per-task colour rule, see
dreamer4/dmc_background_mask.py). Raw values, to be thresholded per task (no
copy-baseline deltas). If useful they move into the [2] section of
hallucination_labels.py.

Sequences: held-out test windows with the paper's rollout, or a frozen export
(--from_npz; use that for anything compared with the manual annotations). See
metric_io.py.

Per imagined frame t, candidate X in {wm: the rollout, recon: the tokenizer
round trip of real[t] = the ceiling a perfect dynamics model would reach}
against real[t], with M_X = foreground_mask(X), M_real = foreground_mask(real[t]):
    masked PSNR   over R_t = M_real | M_wm (the same pixels for both candidates);
                  per sequence from the masked MSE pooled over the horizon
    chamfer (px)  (mean dist of M_X to M_real + mean dist of M_real to M_X) / 2
    hausdorff     max of the two directed max distances (px); hd95 = the same
                  with the 95th percentile instead of the max (robust to specks)
    IoU           |M_X & M_real| / |M_X | M_real| (1 when both are empty)
    masked LPIPS  LPIPS(alex) spatial map of the full frames, averaged over R_t
An empty mask on one side counts as the image diagonal for the distances.
Per sequence: mean over the horizon (masked PSNR: pooled). The paper's
full-frame dPSNR is plotted alongside for reference.

Outputs in --out_dir, per sequence: .mp4 (simulation | world model | masks:
real green, WM magenta, both white; curves underneath), .png (curves), .csv
(per frame); and summary.csv.

Run from src/ (GPU):
    python mask_metrics.py --n_per_task 2 --out_dir ../../logs/mask_metrics/x
    python mask_metrics.py --from_npz ../../logs/annotation/dmc_v1 --no_video --out_dir ...
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from scipy import ndimage

from metric_io import add_source_args, load_sequences, paper_dpsnr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dmc_background_mask import foreground_mask, preset_for_task  # noqa: E402

CANDS = ("wm", "recon")
COLORS = {"wm": "#2a78d6", "recon": "#1b9e4b"}
STYLES = {"wm": "-", "recon": ":"}
LABELS = {"wm": "world model", "recon": "tokenizer recon (ceiling)"}


# ------------------------------------------------------------------ metrics

def distances(mx, mr, cap):
    """Directed distances (px) between two bool masks: d(M_X -> M_real) for
    every M_X pixel and d(M_real -> M_X) for every M_real pixel."""
    dt_r = np.minimum(ndimage.distance_transform_edt(~mr), cap) if mr.any() else np.full(mr.shape, cap)
    dt_x = np.minimum(ndimage.distance_transform_edt(~mx), cap) if mx.any() else np.full(mx.shape, cap)
    return dt_r[mx], dt_x[mr]


def shape_metrics(mx, mr, cap):
    if not mx.any() and not mr.any():
        return {"chamfer": 0.0, "hausdorff": 0.0, "hd95": 0.0, "iou": 1.0}
    a, b = distances(mx, mr, cap)
    stat = lambda v, f: float(f(v)) if v.size else 0.0
    inter, union = (mx & mr).sum(), (mx | mr).sum()
    return {"chamfer": (stat(a, np.mean) + stat(b, np.mean)) / 2,
            "hausdorff": max(stat(a, np.max), stat(b, np.max)),
            "hd95": max(stat(a, lambda v: np.percentile(v, 95)), stat(b, lambda v: np.percentile(v, 95))),
            "iou": float(inter / union)}


@torch.no_grad()
def lpips_maps(fn, a, b, device):
    """a, b: (N,H,W,3) uint8 -> (N,H,W) LPIPS spatial maps."""
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().div(127.5).sub(1.0).to(device)
    return fn(t(a), t(b)).squeeze(1).float().cpu().numpy()


KEYS = ("psnr_m", "chamfer", "hausdorff", "hd95", "iou", "lpips_m")


def sequence_metrics(seq, lpips_fn, device):
    real, ctx = seq["real"], seq["ctx"]
    cands = {"wm": seq["wm"], "recon": seq["recon"]}
    T, H, W, _ = real.shape
    cap = float(np.hypot(H, W))
    hz = np.arange(ctx, T)
    pr = preset_for_task(seq["task"])
    m_real = {t: foreground_mask(real[t], pr) for t in hz}
    m = {c: {t: foreground_mask(cands[c][t], pr) for t in hz} for c in CANDS}
    lp = {c: lpips_maps(lpips_fn, cands[c][hz], real[hz], device) for c in CANDS}
    psnr_wm, psnr_copy, dpsnr = paper_dpsnr(real, seq["wm"], ctx)

    rows, sse, npx = [], {c: 0.0 for c in CANDS}, 0
    for i, t in enumerate(hz):
        R = m_real[t] | m["wm"][t]
        r = {"t": int(t), "fg_real": int(m_real[t].sum()), "region": int(R.sum()),
             "paper_psnr_wm": float(psnr_wm[i]), "paper_psnr_copy": float(psnr_copy[i])}
        npx += int(R.sum())
        for c in CANDS:
            d2 = ((cands[c][t].astype(np.float32) - real[t].astype(np.float32)) / 255.0) ** 2
            e = float(d2.mean(-1)[R].sum()) if R.any() else 0.0
            sse[c] += e
            r[f"psnr_m_{c}"] = 10 * np.log10(1.0 / max(e / R.sum(), 1e-12)) if R.any() else np.nan
            r.update({f"{k}_{c}": v for k, v in shape_metrics(m[c][t], m_real[t], cap).items()})
            r[f"lpips_m_{c}"] = float(lp[c][i][R].mean()) if R.any() else np.nan
        rows.append(r)

    s = {"paper_delta_psnr": dpsnr}
    for c in CANDS:
        s[f"psnr_m_{c}"] = 10 * np.log10(1.0 / max(sse[c] / max(npx, 1), 1e-12)) if npx else np.nan
        for k in KEYS[1:]:
            v = np.array([r[f"{k}_{c}"] for r in rows], float)
            s[f"{k}_{c}"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    return rows, s


# ------------------------------------------------------------------ figures

PANELS = (("psnr_m", "masked PSNR (dB)"), ("chamfer", "chamfer (px)"),
          ("hausdorff", "Hausdorff (px): max (solid) and 95th pct (dashed)"),
          ("iou", "IoU"), ("lpips_m", "masked LPIPS"))


def figure(width_px, height_px, dpi, rows, s, ctx, T, title=None, fs=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2 = "#52514e"
    with plt.rc_context({"font.size": fs, "axes.titlesize": fs + 1, "legend.fontsize": fs - 1,
                         "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2}):
        fig, ax = plt.subplots(1 + len(PANELS), 1, figsize=(width_px / dpi, height_px / dpi), dpi=dpi,
                               sharex=True)
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
        for a, (k, name) in zip(ax[1:], PANELS):
            for c in CANDS:
                a.plot(t, [r[f"{k}_{c}"] for r in rows], color=COLORS[c], ls=STYLES[c],
                       lw=1.6 if c == "wm" else 1.1, label=LABELS[c])
                if k == "hausdorff":
                    a.plot(t, [r[f"hd95_{c}"] for r in rows], color=COLORS[c], ls="--", lw=1.0)
            extra = f"   hd95 WM {s['hd95_wm']:.3g} | recon {s['hd95_recon']:.3g}" if k == "hausdorff" else ""
            a.set_title(f"{name}:  WM {s[f'{k}_wm']:.3g} | recon ceiling {s[f'{k}_recon']:.3g}{extra}", loc="left")
        ax[1].legend(loc="lower left", frameon=False, ncol=2)
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


def mask_overlay(frame, m_real, m_wm):
    """Real frame dimmed to grey; real mask green, WM mask magenta, both white."""
    g = frame.astype(np.float32).mean(-1, keepdims=True).repeat(3, -1) * 0.45
    for m, col in ((m_real & ~m_wm, (40, 220, 60)), (m_wm & ~m_real, (240, 40, 220)),
                   (m_real & m_wm, (255, 255, 255))):
        g[m] = 0.2 * g[m] + 0.8 * np.array(col, np.float32)
    return g.clip(0, 255).astype(np.uint8)


def write_video(path, seq, rows, s, fps, title, scale=2):
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    real, wm, ctx = seq["real"], seq["wm"], seq["ctx"]
    pr = preset_for_task(seq["task"])
    T, H, W, _ = real.shape
    gap = np.full((H, 4, 3), 255, np.uint8)
    width = (3 * W + 8) * scale
    fig, ax = figure(width, 700, 100, rows, s, ctx, T, title=title)
    bars = [a.axvline(0, color="red", lw=2) for a in ax]
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    frames = []
    for t in range(T):
        top = np.concatenate([real[t], gap, wm[t], gap,
                              mask_overlay(real[t], foreground_mask(real[t], pr), foreground_mask(wm[t], pr))], axis=1)
        top = Image.fromarray(np.repeat(np.repeat(top, scale, 0), scale, 1))
        d = ImageDraw.Draw(top)
        for x0, txt in ((6, f"simulation  t={t}/{T - 1}"),
                        ((W + 4) * scale + 6, "world model  " + ("context (recon)" if t < ctx else "imagined")),
                        (2 * (W + 4) * scale + 6, "masks: real / WM / both")):
            b = d.textbbox((x0, 4), txt, font=font)
            d.rectangle((b[0] - 3, b[1] - 2, b[2] + 3, b[3] + 2), fill=(0, 0, 0))
            d.text((x0, 4), txt, fill=(255, 255, 255), font=font)
        for b in bars:
            b.set_xdata([t, t])
        fig.canvas.draw()
        panel = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        panel = np.asarray(Image.fromarray(panel).resize((width, panel.shape[0])))
        frames.append(np.concatenate([np.asarray(top), panel], axis=0))
    plt.close(fig)
    if frames[0].shape[0] % 2:
        frames = [np.pad(f, ((0, 1), (0, 0), (0, 0)), constant_values=255) for f in frames]
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", macro_block_size=1, quality=8)


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(p)
    p.add_argument("--lpips_net", default="alex")
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import lpips
    lpips_fn = lpips.LPIPS(net=args.lpips_net, spatial=True, verbose=False).to(device).eval()
    seqs = load_sequences(args, device)
    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for seq in seqs:
        rows, s = sequence_metrics(seq, lpips_fn, device)
        name = seq["name"]
        summary.append({"name": name, "task": seq["task"], "idx": seq["idx"], **s})
        with open(os.path.join(args.out_dir, name + ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        fg = np.mean([r["fg_real"] for r in rows]) / seq["real"][0, ..., 0].size
        title = f"{name}  (foreground {100 * fg:.1f}% of pixels)"
        if not args.no_video:
            import matplotlib.pyplot as plt
            fig, _ = figure(1300, 1000, 110, rows, s, seq["ctx"], seq["real"].shape[0], title=title, fs=8)
            fig.savefig(os.path.join(args.out_dir, name + ".png"))
            plt.close(fig)
            write_video(os.path.join(args.out_dir, name + ".mp4"), seq, rows, s, args.fps, title)
        print(f"[{name}] paper dPSNR {s['paper_delta_psnr']:+.2f} | WM: PSNR_m {s['psnr_m_wm']:.1f}, "
              f"chamfer {s['chamfer_wm']:.2f}, HD {s['hausdorff_wm']:.1f}, HD95 {s['hd95_wm']:.1f}, "
              f"IoU {s['iou_wm']:.3f}, LPIPS_m {s['lpips_m_wm']:.3f} | ceiling: IoU {s['iou_recon']:.3f}, "
              f"HD95 {s['hd95_recon']:.1f}", flush=True)

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"[out] {args.out_dir}")


if __name__ == "__main__":
    main()
