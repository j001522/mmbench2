# roc_report.py
"""
ROC report of the main hallucination metrics against the manual annotations,
on the combined annotated dataset (all sets, all tasks).

Metrics (per sequence: mean over the 16 imagined frames; paper dPSNR pooled
over the horizon as in the paper), each oriented so that higher = more
hallucinated for the ROC:
    paper_dpsnr   the paper's rollout dPSNR (lower = worse)
    hausdorff     foreground-mask Hausdorff distance, px
    hd95          foreground-mask 95th-percentile Hausdorff, px
    iou           foreground-mask IoU (lower = worse)
    dino          DINOv2-with-registers, same-location cosine distance to the
                  tokenizer recon, mean over patches
    clef, qwen    VLM judges (clef_metrics.py / vlm_judge_metrics.py): Clef 27B
                  and its base Qwen3.8-27B, 2x2 [real|wm] x (t-1, t) image,
                  expected 0-3 severity level per frame

Operating point: the threshold maximizing balanced accuracy (= Youden's J),
fit per task (the intended use) and, for comparison, one global threshold on
all tasks pooled. Labels: verdict yes = 1, no = 0, unsure dropped.

Outputs in --out:
    roc.png                 ROC curves: all tasks pooled + one panel per task,
                            operating points marked
    summary.csv             per scope x metric: n, positives, AUROC [95% CI,
                            cluster bootstrap over episodes], average precision,
                            threshold, TP/FP/TN/FN, precision, recall,
                            specificity, F1, accuracy, balanced accuracy, and
                            grouped 5-fold CV balanced accuracy
    examples/<metric>/      the 2 most confident false positives and false
                            negatives under the per-task thresholds: .mp4
                            (simulation | world model) and .png strip, captioned
                            with value, threshold and the annotation
    examples/index.csv

Run from src/:
    python roc_report.py --sets dmc_v1 dmc_v2 mw_v1 mw_v2 --out ../../logs/annotation/roc_report
    (VLM curves need the clef_<set> CSVs; roc_report_vlm/ is the run with them)
"""
import argparse
import csv
import os

import numpy as np

from eval_annotations import ROOT, auroc, bal_acc, best_threshold, cluster_boot, cv_bal_acc, load_set, scores

METRICS = [  # name, key in eval_annotations.scores(), sign (oriented = raw * sign), unit, label
    ("paper_dpsnr", ("paper_dpsnr", "pooled"), -1, "dB", "paper dPSNR (baseline)"),
    ("hausdorff", ("hausdorff", "mean"), +1, "px", "Hausdorff (mask)"),
    ("hd95", ("hd95", "mean"), +1, "px", "HD95 (mask)"),
    ("iou", ("iou", "mean"), -1, "", "IoU (mask)"),
    ("dino", ("dino_reg_recon", "mean"), +1, "", "DINOv2-reg same-location (vs recon)"),
    ("clef", ("clef:clef_stack_severity", "mean"), +1, "", "Clef-27B VLM (severity)"),
    ("qwen", ("clef:qwen27_stack_severity", "mean"), +1, "", "Qwen3.8-27B VLM (severity)"),
]
COLORS = {"paper_dpsnr": "#888780", "hausdorff": "#d32f2f", "hd95": "#e8710a", "iou": "#1b9e4b", "dino": "#2a78d6",
          "clef": "#7b3fb5", "qwen": "#8a5a2b"}
LINESTYLE = {"paper_dpsnr": "--", "clef": "-.", "qwen": "-."}
TASKS = ("reacher-easy", "cup-catch", "mw-reach")


# ------------------------------------------------------------------ statistics

def roc_curve(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp, fp = np.cumsum(y)[distinct], np.cumsum(1 - y)[distinct]
    tpr = np.r_[0, tp / max(y.sum(), 1)]
    fpr = np.r_[0, fp / max((1 - y).sum(), 1)]
    return fpr, tpr


def average_precision(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    if y.sum() == 0:
        return np.nan
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def confusion(pred, y):
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    tn, fn = int((~pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and np.isfinite(prec + rec) else np.nan
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, precision=prec, recall=rec, specificity=spec, f1=f1,
                accuracy=(tp + tn) / len(y), balanced_accuracy=np.nanmean([rec, spec]))


# ------------------------------------------------------------------ examples

def save_example(path_stem, real, wm, ctx, caption, scale=2, fps=4):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    T, H, W, _ = real.shape
    try:
        font, small = ImageFont.load_default(size=18), ImageFont.load_default(size=15)
    except TypeError:
        font = small = ImageFont.load_default()
    gap = np.full((H * scale, 8, 3), 255, np.uint8)
    up = lambda x: np.repeat(np.repeat(x, scale, 0), scale, 1)
    lines = caption.split("\n")
    cap_h = 8 + 20 * len(lines)
    frames = []
    for t in range(T):
        top = Image.fromarray(np.concatenate([up(real[t]), gap, up(wm[t])], axis=1))
        d = ImageDraw.Draw(top)
        for x0, txt in ((6, f"simulation  t={t}/{T - 1}"),
                        (W * scale + 14, "world model  " + ("context (recon)" if t < ctx else
                                                           f"imagined {t - ctx + 1}/{T - ctx}"))):
            b = d.textbbox((x0, 4), txt, font=font)
            d.rectangle((b[0] - 3, b[1] - 2, b[2] + 3, b[3] + 2), fill=(0, 0, 0))
            d.text((x0, 4), txt, fill=(255, 255, 255), font=font)
        cap = Image.new("RGB", (top.width, cap_h), (255, 255, 255))
        dc = ImageDraw.Draw(cap)
        for i, line in enumerate(lines):
            dc.text((8, 6 + 20 * i), line, fill=(20, 20, 20), font=small)
        frames.append(np.concatenate([np.asarray(top), np.asarray(cap)], axis=0))
    if frames[0].shape[0] % 2:
        frames = [np.pad(f, ((0, 1), (0, 0), (0, 0)), constant_values=255) for f in frames]
    imageio.mimwrite(path_stem + ".mp4", frames, fps=fps, codec="libx264", macro_block_size=1, quality=8)
    # strip: simulation (top) and world model (bottom) at 6 imagined frames
    ts = np.linspace(ctx, T - 1, 6).round().astype(int)
    g = np.full((H, 4, 3), 255, np.uint8)
    row = lambda a: np.concatenate(sum([[a[t], g] for t in ts], [])[:-1], axis=1)
    strip = np.concatenate([row(real), np.full((4, row(real).shape[1], 3), 255, np.uint8), row(wm)], axis=0)
    cap = Image.new("RGB", (strip.shape[1], cap_h + 22), (255, 255, 255))
    dc = ImageDraw.Draw(cap)
    dc.text((8, 4), "top: simulation, bottom: world model, frames t=" + ", ".join(map(str, ts)),
            fill=(80, 80, 80), font=small)
    for i, line in enumerate(lines):
        dc.text((8, 26 + 20 * i), line, fill=(20, 20, 20), font=small)
    imageio.imwrite(path_stem + ".png", np.concatenate([strip, np.asarray(cap)], axis=0))


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", default=["dmc_v1", "dmc_v2", "mw_v1", "mw_v2"])
    p.add_argument("--root", default=ROOT)
    p.add_argument("--out", default=os.path.join(ROOT, "roc_report"))
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--n_examples", type=int, default=2)
    args = p.parse_args()

    seqs = [q for s in args.sets for q in load_set(s, args.root)]
    tasks = [t for t in TASKS if any(q["task"] == t for q in seqs)]
    y = np.array([q["y"] for q in seqs])
    g = np.array([q["group"] for q in seqs])
    task = np.array([q["task"] for q in seqs])
    sc = [scores(q) for q in seqs]
    S = {m: np.array([d.get(k, np.nan) for d in sc]) for m, k, *_ in METRICS}
    print(f"[data] {len(seqs)} sequences from {args.sets}: " +
          ", ".join(f"{t} {int(y[task == t].sum())}/{int((task == t).sum())} yes" for t in tasks))
    os.makedirs(os.path.join(args.out, "examples"), exist_ok=True)

    # per-task thresholds (intended use) -> pooled predictions
    thr_task = {(m, t): best_threshold(S[m][task == t], y[task == t]) for m, *_ in METRICS for t in tasks}
    pred_task = {m: np.zeros(len(y), bool) for m, *_ in METRICS}
    for m, *_ in METRICS:
        for t in tasks:
            sel = task == t
            pred_task[m][sel] = S[m][sel] >= thr_task[(m, t)]

    rows = []
    scopes = [("all, global threshold", np.ones(len(y), bool))] + [(t, task == t) for t in tasks]
    for scope, sel in scopes:
        for m, key, sign, unit, label in METRICS:
            s, yy, gg = S[m][sel], y[sel], g[sel]
            thr = best_threshold(s, yy)
            lo, hi = cluster_boot(s, yy, gg, n=args.n_boot)
            rows.append({"scope": scope, "metric": m, "n": int(sel.sum()), "positives": int(yy.sum()),
                         "auroc": auroc(s, yy), "auroc_lo": lo, "auroc_hi": hi,
                         "average_precision": average_precision(s, yy),
                         "threshold": thr * sign, "rule": f"{'>=' if sign > 0 else '<='} threshold {unit}".strip(),
                         **confusion(s >= thr, yy), "cv_balanced_accuracy": cv_bal_acc(s, yy, gg)})
    for m, key, sign, unit, label in METRICS:      # pooled, each task at its own threshold
        cv = [r["cv_balanced_accuracy"] for r in rows if r["metric"] == m and r["scope"] in tasks]
        rows.append({"scope": "all, per-task thresholds", "metric": m, "n": len(y), "positives": int(y.sum()),
                     "auroc": np.nan, "auroc_lo": np.nan, "auroc_hi": np.nan, "average_precision": np.nan,
                     "threshold": "; ".join(f"{t}: {thr_task[(m, t)] * sign:.4g}" for t in tasks),
                     "rule": f"{'>=' if sign > 0 else '<='} threshold {unit}".strip(),
                     **confusion(pred_task[m], y), "cv_balanced_accuracy": float(np.mean(cv))})
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ROC figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("All tasks pooled (one global threshold)", np.ones(len(y), bool), "all, global threshold")] + \
             [(t, task == t, t) for t in tasks]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.9), dpi=120)
    for ax, (title, sel, scope) in zip(np.atleast_1d(axes), panels):
        for m, key, sign, unit, label in METRICS:
            r = next(r for r in rows if r["scope"] == scope and r["metric"] == m)
            fpr, tpr = roc_curve(S[m][sel], y[sel])
            ax.plot(fpr, tpr, color=COLORS[m], lw=2.0 if m != "paper_dpsnr" else 1.4,
                    ls=LINESTYLE.get(m, "-"),
                    label=f"{label}  {r['auroc']:.3f} [{r['auroc_lo']:.2f}, {r['auroc_hi']:.2f}]")
            ax.plot(1 - r["specificity"], r["recall"], "o", color=COLORS[m], ms=6, mec="white", mew=1)
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8, ls=":")
        n1 = int(y[sel].sum())
        ax.set_title(f"{title}\nn={int(sel.sum())}, {n1} hallucinating", fontsize=10)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=6.5, frameon=False, title="AUROC [95% CI]", title_fontsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Hallucination metrics vs manual annotations (dots: balanced-accuracy operating point)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "roc.png"))
    plt.close(fig)

    # examples: most confident errors under the per-task thresholds
    idx_rows = []
    for m, key, sign, unit, label in METRICS:
        d = os.path.join(args.out, "examples", m)
        os.makedirs(d, exist_ok=True)
        margin = np.array([S[m][i] - thr_task[(m, task[i])] for i in range(len(y))])   # >0 = flagged
        fp = [i for i in np.argsort(-margin) if pred_task[m][i] and y[i] == 0][:args.n_examples]
        fn = [i for i in np.argsort(margin) if not pred_task[m][i] and y[i] == 1][:args.n_examples]
        for kind, ids in (("FP", fp), ("FN", fn)):
            for rank, i in enumerate(ids, 1):
                q = seqs[i]
                data = np.load(os.path.join(args.root, q["set"], "data", f"{q['id']}.npz"))
                raw, thr_raw = S[m][i] * sign, thr_task[(m, q["task"])] * sign
                verdict = "hallucinating" if q["y"] else "not hallucinating"
                ann = f"annotation: {verdict}" + (f" ({', '.join(q['types'])}; onset t={q['onset']})"
                                                  if q["y"] else "")
                rule = ">=" if sign > 0 else "<="
                caption = (f"{kind} #{rank} for {label}: value {raw:.4g} {unit}, per-task threshold "
                           f"{rule} {thr_raw:.4g} {unit} -> {'flagged' if pred_task[m][i] else 'not flagged'}\n"
                           f"{q['task']}  {q['set']}/{q['id']}   {ann}")
                stem = os.path.join(d, f"{kind}{rank}_{q['id']}_{q['task']}")
                save_example(stem, data["real"], data["wm"], int(json_ctx(data)), caption)
                idx_rows.append({"metric": m, "kind": kind, "rank": rank, "set": q["set"], "id": q["id"],
                                 "task": q["task"], "value": f"{raw:.4g}", "threshold": f"{thr_raw:.4g}",
                                 "annotation": verdict, "types": " ".join(q["types"]), "onset": q["onset"],
                                 "files": os.path.relpath(stem, args.out) + ".{mp4,png}"})
    with open(os.path.join(args.out, "examples", "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(idx_rows[0].keys()))
        w.writeheader()
        w.writerows(idx_rows)

    print(f"\n{'scope':26s} {'metric':12s} {'AUROC [95% CI]':22s} {'AP':>5s} {'prec':>5s} {'rec':>5s} "
          f"{'spec':>5s} {'F1':>5s} {'balAcc':>6s} {'CV':>5s}")
    for r in rows:
        au = f"{r['auroc']:.3f} [{r['auroc_lo']:.2f}, {r['auroc_hi']:.2f}]" if np.isfinite(r["auroc"]) else "-"
        ap = f"{r['average_precision']:.2f}" if np.isfinite(r["average_precision"]) else "-"
        print(f"{r['scope']:26s} {r['metric']:12s} {au:22s} {ap:>5s} {r['precision']:5.2f} {r['recall']:5.2f} "
              f"{r['specificity']:5.2f} {r['f1']:5.2f} {r['balanced_accuracy']:6.3f} {r['cv_balanced_accuracy']:5.2f}")
    print(f"\n[out] {args.out}: roc.png, summary.csv, examples/ ({len(idx_rows)} clips)")


def json_ctx(data):
    import json
    return json.loads(str(data["meta"]))["ctx"]


if __name__ == "__main__":
    main()
