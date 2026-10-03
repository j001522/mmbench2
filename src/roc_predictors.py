# roc_predictors.py
"""
ROC curves of the paper's hallucination predictors against the manual
annotations: u_r, u_f, u_s raw and motion-normalized (u / max(motion, 1e-3)),
motion alone, and our Hausdorff (mask) metric as the reference. Per sequence:
mean over the 16 imagined frames (eval_annotations.scores, "mean"). One panel
per task, all annotation sets pooled. Inputs are the per-frame CSVs written by
paper_predictors.py and mask_metrics.py under logs/annotation/_hidden_metrics.

Run from src/:
    python roc_predictors.py --out ../../logs/annotation/roc_predictors
Writes roc_predictors.png and auroc.csv.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_annotations import load_set, scores
from plan_halluc import auroc

ROOT = "../../logs/annotation"
TASKS = ("reacher-easy", "cup-catch", "mw-reach")
# (score key, label, color, linestyle): color = predictor, dash = normalized
CURVES = (
    (("pred:u_r", "mean"), "u_r", "#2a78d6", "-"),
    (("pred:u_r_norm", "mean"), "u_r / motion", "#2a78d6", "--"),
    (("pred:u_f", "mean"), "u_f", "#eb6834", "-"),
    (("pred:u_f_norm", "mean"), "u_f / motion", "#eb6834", "--"),
    (("pred:u_s", "mean"), "u_s", "#1baf7a", "-"),
    (("pred:u_s_norm", "mean"), "u_s / motion", "#1baf7a", "--"),
    (("pred:motion", "mean"), "motion alone", "#eda100", ":"),
    (("hausdorff", "mean"), "Hausdorff (ours, reference)", "#3b3b3b", "-"),
)


def roc_curve(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    keep = np.r_[np.diff(s) != 0, True]                  # one point per distinct threshold
    tpr = np.r_[0.0, tp[keep] / max(y.sum(), 1)]
    fpr = np.r_[0.0, fp[keep] / max((1 - y).sum(), 1)]
    return fpr, tpr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=["dmc_v1", "dmc_v2", "mw_v1", "mw_v2"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = [s for name in a.sets for s in load_set(name, ROOT)]
    for s in seqs:
        s["sc"] = scores(s)

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#8a8a8a", "axes.linewidth": 0.6,
                         "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a"})
    fig, axes = plt.subplots(1, len(TASKS), figsize=(5.0 * len(TASKS), 5.0), sharey=True)
    rows, au_tab = [], {}
    for ax, task in zip(axes, TASKS):
        T = [s for s in seqs if s["task"] == task]
        y = [s["y"] for s in T]
        ax.plot([0, 1], [0, 1], color="#c8c8c8", lw=0.8, zorder=0)
        for key, lab, col, ls in CURVES:
            v = [s["sc"].get(key, np.nan) for s in T]
            if not np.isfinite(v).any():
                continue
            au = auroc(v, y)
            au_tab.setdefault(lab, {})[task] = au
            fpr, tpr = roc_curve(v, y)
            lw = 1.6 if key[0] == "hausdorff" else 2.2
            ax.plot(fpr, tpr, color=col, ls=ls, lw=lw)
            rows.append({"task": task, "metric": lab, "n": len(T), "positives": int(sum(y)), "auroc": f"{au:.4f}"})
        ax.set_title(f"{task}\n(n={len(T)}, {sum(y)} hallucinated)", fontsize=12)
        ax.set_xlabel("false positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(color="#ececec", lw=0.6)
    axes[0].set_ylabel("true positive rate")
    # One shared legend (same line styles in every panel); each entry carries
    # the AUROC on the three tasks, in panel order, in aligned columns.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=col, ls=ls, lw=2.2) for _k, _l, col, ls in CURVES]
    labels = [f"{lab:<28s}" + "".join(f"{au_tab[lab].get(t, np.nan):>7.2f}" for t in TASKS)
              for _k, lab, _c, _s in CURVES]
    header = f"{'AUROC':<28s}" + "".join(f"{t.split('-')[0][:7]:>7s}" for t in TASKS)
    # handle (3.0) + pad (0.8) em in front of each label, so the left-aligned
    # title header needs the same indent to sit over the AUROC columns.
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=False,
               title=" " * 7 + header, title_fontproperties={"family": "monospace", "size": 10.5},
               prop={"family": "monospace", "size": 10.5}, handlelength=3.0, handletextpad=0.8,
               alignment="left")
    fig.suptitle("Paper's hallucination predictors vs manual labels "
                 "(per-sequence mean over the 16 imagined frames)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(a.out, "roc_predictors.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.2)
    with open(os.path.join(a.out, "auroc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
