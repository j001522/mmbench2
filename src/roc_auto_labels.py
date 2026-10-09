# roc_auto_labels.py
"""
AUROC of the paper's hallucination predictors (u_r, u_f, u_s, raw, divided by
the rollout's per-step motion (u / motion, ours), divided by the per-task mean
motion (u / task motion, the paper's definition; roc_predictors.task_motion), and
motion alone) against three per-sequence labels, on the frozen annotation sets:
    paper_dpsnr  the paper's "scene divergent" label: rollout dPSNR <= 0, with
                 dPSNR = 10 log10(sum mse_copy / sum mse_wm) over the 16-frame
                 horizon (metric_io.paper_dpsnr), rebuilt from the per-frame PSNRs
    hausdorff    mean mask Hausdorff over the horizon >= the per-task threshold
                 (plan_halluc.THRESH; those thresholds were fit to the manual
                 labels of these same sequences)
    manual       the hand labels (reference; = roc_predictors.py)
Scopes:
    annotated test     dmc_v1 + mw_v2: hand-labeled windows sampled like the
                       paper's (uniform over the test split, ~50/task ~ the
                       paper's ~45)
    annotated val      dmc_v2 + mw_v1: the same sampling on the val split
                       (val windows move much more than test ones)
    annotated test + val   all 298 hand-labeled sequences
    auto test          auto_test_* (launch_auto_sets.sh): every non-overlapping
                       window the random packing finds in the test split, no
                       manual labels -> the paper's distribution, ~8x the size
    auto test + val    + auto_val_* (val windows move more than test ones)
Sets that do not exist yet are skipped. Per task, plus the mean over the tasks
with >= MIN_CLASS sequences in each class, plus all tasks pooled (one global
threshold; the only column where u / task motion can differ from u).
95% CI: bootstrap over sequences.
counts.csv: sequences and positives per scope, task and label.

Equal-motion AUROC (auroc_eq_motion): motion alone already predicts the labels,
so this asks whether a predictor separates hallucinated from good rollouts AT
THE SAME MOTION. Per task, the sequences are split into N_MBINS equal-count
bins of the rollout's latent motion (pred:motion, the same quantity as the
"motion alone" baseline); the AUROC inside each bin with >= MIN_BIN sequences
of each class, averaged weighted by bin size (eq_bins = bins used). Bins are
recomputed inside every bootstrap resample. Motion alone keeps a little AUROC
here only through the spread of motion inside a bin.

Run from src/ (CPU, seconds):
    python roc_auto_labels.py --out ../../logs/annotation/roc_auto_labels
Writes auroc.csv, counts.csv, auroc.png, auroc_equal_motion_*.png and
auroc_paper_setting.png / auroc_auto_hausdorff.png (FIG_TABLES: one scope x label as a table)
with their ROC curves in *_roc.png (first panel: all tasks pooled, one global threshold).
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval_annotations import fnum, load_set, scores  # noqa: E402
from plan_halluc import THRESH, auroc  # noqa: E402
from roc_predictors import CURVES, add_task_norm, roc_curve, task_motion  # noqa: E402

ROOT = "../../logs/annotation"
TASKS = ("reacher-easy", "cup-catch", "mw-reach")
AUTO_TEST = ("auto_test_reacher", "auto_test_cup", "auto_test_mw")
AUTO_VAL = ("auto_val_reacher", "auto_val_cup", "auto_val_mw")
SCOPES = {"annotated test": ("dmc_v1", "mw_v2"),
          "annotated val": ("dmc_v2", "mw_v1"),
          "annotated test + val": ("dmc_v1", "mw_v2", "dmc_v2", "mw_v1"),
          "auto test": AUTO_TEST,
          "auto test + val": AUTO_TEST + AUTO_VAL}
PREDICTORS = [("u_r", "pred:u_r"), ("u_f", "pred:u_f"), ("u_s", "pred:u_s"),
              ("u_r / motion", "pred:u_r_norm"), ("u_f / motion", "pred:u_f_norm"),
              ("u_s / motion", "pred:u_s_norm"),
              ("u_r / task motion", "pred:u_r_tnorm"), ("u_f / task motion", "pred:u_f_tnorm"),
              ("u_s / task motion", "pred:u_s_tnorm"), ("motion alone", "pred:motion")]
POOLED = "all tasks pooled"
LABELS = ("paper_dpsnr", "hausdorff", "manual")
N_BOOT = 1000
MIN_CLASS = 5
N_MBINS = 5          # equal-motion AUROC: motion bins per task
MIN_BIN = 3          # ... and sequences per class for a bin to count
FIG_EQ = [("auto test + val", "hausdorff"), ("annotated test + val", "manual")]   # auroc_equal_motion_*.png
FIG_TABLES = [("auto test", "paper_dpsnr", "auroc_paper_setting.png",
               "Paper setting: the paper's label (dPSNR <= 0), random test windows"),
              ("auto test + val", "hausdorff", "auroc_auto_hausdorff.png",
               "Hausdorff label, all random test + val windows")]


def seq_dpsnr(rows):
    """Paper dPSNR of one sequence from its per-frame PSNR columns (pooled MSE)."""
    w = np.array([fnum(r["paper_psnr_wm"]) for r in rows])
    c = np.array([fnum(r["paper_psnr_copy"]) for r in rows])
    mse = lambda p: np.mean(10 ** (-p / 10))
    return 10 * np.log10(mse(c) / mse(w))


def load_auto_set(name, root):
    """An unlabeled export (no annotations): key.csv + the hidden pred/mask metrics."""
    sdir = os.path.join(root, name)
    seqs = []
    for k in csv.DictReader(open(os.path.join(sdir, "key.csv"))):
        frames = {}
        for src in ("mask", "pred"):
            p = os.path.join(root, "_hidden_metrics", f"{src}_{name}", f"{k['id']}_{k['task']}.csv")
            if os.path.exists(p):
                frames[src] = list(csv.DictReader(open(p)))
        if "pred" in frames:
            seqs.append({"set": name, "id": k["id"], "task": k["task"], "y": None, "frames": frames})
    return seqs


def eq_motion_auroc(v, y, m):
    """AUROC within equal-count motion bins, averaged weighted by bin size; (auroc, bins used)."""
    edges = np.quantile(m, np.linspace(0, 1, N_MBINS + 1))
    b = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, N_MBINS - 1)
    aus, ws = [], []
    for k in range(N_MBINS):
        i = b == k
        if min(int(y[i].sum()), int((1 - y[i]).sum())) >= MIN_BIN:
            aus.append(auroc(v[i], y[i]))
            ws.append(int(i.sum()))
    return (float(np.average(aus, weights=ws)), len(aus)) if aus else (np.nan, 0)


def labels_of(s):
    rows = s["frames"].get("pred") or s["frames"].get("mask")
    hd = np.nanmean([fnum(r["hausdorff_wm"]) for r in s["frames"]["mask"]]) if "mask" in s["frames"] else np.nan
    return {"paper_dpsnr": int(seq_dpsnr(rows) <= 0),
            "hausdorff": int(hd >= THRESH[s["task"]]) if np.isfinite(hd) else None,
            "manual": s["y"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    names = dict.fromkeys(n for sets in SCOPES.values() for n in sets)
    seqs = {}
    for n in names:
        if not os.path.exists(os.path.join(ROOT, n, "key.csv")):
            print(f"[skip] {n}: not exported")
            continue
        seqs[n] = load_auto_set(n, ROOT) if n.startswith("auto_") else load_set(n, ROOT)
        print(f"[load] {n}: {len(seqs[n])} sequences")
    scopes = {sc: sets for sc, sets in SCOPES.items() if all(n in seqs for n in sets)}
    for ss in seqs.values():
        for s in ss:
            s["sc"], s["lab"] = scores(s), labels_of(s)
    m_task = task_motion(ROOT)
    print("[task motion]", {t: round(v, 4) for t, v in m_task.items()})
    for ss in seqs.values():
        add_task_norm(ss, m_task)

    rows, counts = [], []
    for scope, sets in scopes.items():
        pool = [s for n in sets for s in seqs[n]]
        for lab in LABELS:
            for t in TASKS:
                y = [s["lab"][lab] for s in pool if s["task"] == t and s["lab"][lab] is not None]
                counts.append({"scope": scope, "task": t, "label": lab, "n": len(y), "positives": int(sum(y)),
                               "positive_pct": round(100 * sum(y) / len(y), 1) if y else np.nan})
            if not any(c["n"] for c in counts[-len(TASKS):]):
                continue                                   # e.g. manual on the unlabeled sets
            for name, key in PREDICTORS:
                per = {}
                for t in TASKS:
                    T = [s for s in pool if s["task"] == t and s["lab"][lab] is not None]
                    y = np.array([s["lab"][lab] for s in T])
                    v = np.array([s["sc"].get((key, "mean"), np.nan) for s in T])
                    m = np.array([s["sc"].get(("pred:motion", "mean"), np.nan) for s in T])
                    if len(T) == 0:
                        continue
                    au = auroc(v, y)
                    eq, eq_bins = eq_motion_auroc(v, y, m)
                    idx = rng.integers(0, len(T), (N_BOOT, len(T)))
                    d = np.array([auroc(v[i], y[i]) for i in idx])
                    d_eq = np.array([eq_motion_auroc(v[i], y[i], m[i])[0] for i in idx]) if eq_bins else \
                        np.full(N_BOOT, np.nan)
                    per[t] = (au, d, eq, d_eq)
                    rows.append({"scope": scope, "label": lab, "predictor": name, "task": t, "n": len(T),
                                 "positives": int(y.sum()), "auroc": au,
                                 "ci_lo": np.nanpercentile(d, 2.5), "ci_hi": np.nanpercentile(d, 97.5),
                                 "auroc_eq_motion": eq, "eq_ci_lo": np.nanpercentile(d_eq, 2.5),
                                 "eq_ci_hi": np.nanpercentile(d_eq, 97.5), "eq_bins": eq_bins})
                # mean over the tasks with >= MIN_CLASS sequences in each class
                # (fewer: the AUROC is noise and most resamples are undefined)
                task_rows = [r for r in rows[-len(per):]]
                ok = [r["task"] for r in task_rows if min(r["positives"], r["n"] - r["positives"]) >= MIN_CLASS]
                d = np.mean([per[t][1] for t in ok], axis=0) if ok else np.full(N_BOOT, np.nan)
                ok_eq = [t for t in ok if np.isfinite(per[t][2])]
                d_eq = np.mean([per[t][3] for t in ok_eq], axis=0) if ok_eq else np.full(N_BOOT, np.nan)
                rows.append({"scope": scope, "label": lab, "predictor": name, "task": "mean over tasks",
                             "n": sum(r["n"] for r in task_rows), "positives": sum(r["positives"] for r in task_rows),
                             "tasks_in_mean": " ".join(ok),
                             "auroc": float(np.mean([per[t][0] for t in ok])) if ok else np.nan,
                             "ci_lo": np.nanpercentile(d, 2.5), "ci_hi": np.nanpercentile(d, 97.5),
                             "auroc_eq_motion": float(np.mean([per[t][2] for t in ok_eq])) if ok_eq else np.nan,
                             "eq_ci_lo": np.nanpercentile(d_eq, 2.5), "eq_ci_hi": np.nanpercentile(d_eq, 97.5),
                             "tasks_in_eq_mean": " ".join(ok_eq)})
                # all tasks pooled, one global threshold (the only column where the
                # per-task-constant normalization can differ from raw u)
                T = [s for s in pool if s["lab"][lab] is not None]
                y = np.array([s["lab"][lab] for s in T])
                v = np.array([s["sc"].get((key, "mean"), np.nan) for s in T])
                idx = rng.integers(0, len(T), (N_BOOT, len(T)))
                d = np.array([auroc(v[i], y[i]) for i in idx])
                rows.append({"scope": scope, "label": lab, "predictor": name, "task": POOLED, "n": len(T),
                             "positives": int(y.sum()), "auroc": auroc(v, y),
                             "ci_lo": np.nanpercentile(d, 2.5), "ci_hi": np.nanpercentile(d, 97.5),
                             "auroc_eq_motion": np.nan, "eq_bins": 0})
    with open(os.path.join(a.out, "auroc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(counts[0].keys()))
        w.writeheader()
        w.writerows(counts)
    for c in counts:
        print(f"[count] {c['scope']:16s} {c['task']:13s} {c['label']:12s} {c['positives']:4d}/{c['n']:4d}")

    # One heatmap per scope: predictors x (label, task), AUROC printed in each cell;
    # under each task: positives / sequences for that label.
    cnt = {(c["scope"], c["task"], c["label"]): c for c in counts}
    fig, axes = plt.subplots(len(scopes), 1, figsize=(16, 4.4 * len(scopes) + 0.6), squeeze=False)
    for ax, scope in zip(axes[:, 0], scopes):
        labs = [lab for lab in LABELS if any(r["scope"] == scope and r["label"] == lab for r in rows)]
        grp = (*TASKS, "mean over tasks", POOLED)
        cols = [(lab, t) for lab in labs for t in grp]
        Z = np.full((len(PREDICTORS), len(cols)), np.nan)
        for r in rows:
            if r["scope"] == scope:
                Z[[n for n, _ in PREDICTORS].index(r["predictor"]), cols.index((r["label"], r["task"]))] = r["auroc"]
        ax.imshow(Z, cmap="RdBu", vmin=0.0, vmax=1.0, aspect="auto")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(Z[i, j] - 0.5) > 0.3 else "#0b0b0b")
        n_lab = {lab: next(f"{r['positives']}/{r['n']} positive; mean over {len(r['tasks_in_mean'].split())} tasks"
                           for r in rows
                           if r["scope"] == scope and r["label"] == lab and r["task"] == "mean over tasks")
                 for lab in labs}
        ticks = []
        for lab, t in cols:
            if t == "mean over tasks":
                ticks.append("mean")
            elif t == POOLED:
                ticks.append("pooled")
            else:
                c = cnt[(scope, t, lab)]
                few = min(c["positives"], c["n"] - c["positives"]) < MIN_CLASS
                ticks.append(f"{t.split('-')[0]}{'*' if few else ''}\n{c['positives']}/{c['n']}")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(ticks, fontsize=8)
        for k, lab in enumerate(labs):
            ax.text(k * len(grp) + (len(grp) - 1) / 2, -0.75, f"label: {lab} ({n_lab[lab]})", ha="center", fontsize=9)
            if k:
                ax.axvline(k * len(grp) - 0.5, color="white", lw=3)
        ax.set_yticks(range(len(PREDICTORS)))
        ax.set_yticklabels([n for n, _ in PREDICTORS], fontsize=9)
        ax.set_title(f"{scope}: {' + '.join(scopes[scope])}", fontsize=10, loc="left", pad=24, fontweight="bold")
    fig.suptitle("AUROC of the paper's predictors per label (0.5 = chance; red = wrong direction)\n"
                 f"under each task: positives/sequences; * = fewer than {MIN_CLASS} in a class, left out of the mean",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(a.out, "auroc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"-> {path}")

    # Equal-motion AUROC, house bar style: tasks side by side (mean first), one bar
    # per predictor; dashed = the same predictor's AUROC over all sequences.
    from plan_figures import AVG, INK2, SERIES, bar_chart
    # per-task-constant normalization = raw u inside a task's motion bins: left out
    eq_preds = [(n, k) for n, k in PREDICTORS if not k.endswith("_tnorm")]
    colors = SERIES[:len(eq_preds) - 1] + ["#8a8985"]
    for scope, lab in FIG_EQ:
        sel = [r for r in rows if r["scope"] == scope and r["label"] == lab and r["task"] != POOLED
               and r["predictor"] in dict(eq_preds)]
        if not sel:
            continue
        gname = {t: t for t in TASKS} | {"mean over tasks": AVG}
        groups = [AVG] + [t for t in TASKS if any(r["task"] == t and r["eq_bins"] for r in sel)]
        values, raw = {}, {}
        for r in sel:
            g = gname[r["task"]]
            values[(g, r["predictor"])] = (r["auroc_eq_motion"], r["eq_ci_lo"], r["eq_ci_hi"])
            raw.setdefault(r["predictor"], {})[g] = r["auroc"]
        bars = [(n, c, None) for (n, _), c in zip(eq_preds, colors)]
        refs = [("AUROC over all sequences", INK2, raw[n], [n]) for n, _ in eq_preds]
        refs.append(("chance", "#c9c8c4", {g: 0.5 for g in groups}))
        tag = f"{scope}_{lab}".replace(" + ", "_").replace(" ", "_")
        bar_chart(os.path.join(a.out, f"auroc_equal_motion_{tag}.png"), groups, bars, values,
                  "AUROC within equal-motion bins",
                  f"Do the predictors detect hallucinations beyond motion? AUROC among rollouts with the same "
                  f"motion ({N_MBINS} equal-count bins per task), {lab} label, {scope}", ylim=(0, 1.05), refs=refs)
        print(f"-> auroc_equal_motion_{tag}.png")

    # Single-(scope, label) AUROC tables (FIG_TABLES), best per column in bold, + their ROC curves
    cell_names = [n for n, _ in PREDICTORS]
    for scope, lab, fname, title in FIG_TABLES:
        sel = [r for r in rows if r["scope"] == scope and r["label"] == lab]
        if not sel:
            continue
        cols = [t for t in TASKS if any(r["task"] == t and np.isfinite(r["auroc"]) for r in sel)] + ["mean over tasks", POOLED]
        cell = {(r["predictor"], r["task"]): r for r in sel}
        best = {c: max(cell[(n, c)]["auroc"] for n, _ in PREDICTORS) for c in cols}
        head = [f"{'mean' if c == 'mean over tasks' else 'pooled' if c == POOLED else c}\n"
                f"{cell[(PREDICTORS[0][0], c)]['positives']}/{cell[(PREDICTORS[0][0], c)]['n']} hallucinated"
                for c in cols]
        text = [[f"{cell[(n, c)]['auroc']:.2f} [{cell[(n, c)]['ci_lo']:.2f}, {cell[(n, c)]['ci_hi']:.2f}]"
                 for c in cols] for n, _ in PREDICTORS]
        fig, ax = plt.subplots(figsize=(2.2 + 2.0 * len(cols), 4.6))
        ax.axis("off")
        tb = ax.table(cellText=text, rowLabels=[n for n, _ in PREDICTORS], colLabels=head, loc="center",
                      cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(9)
        tb.scale(1, 1.6)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor("#c9c8c4")
            if i == 0:
                c.set_height(c.get_height() * 1.5)
                c.set_text_props(weight="bold")
            elif j >= 0 and cell[(PREDICTORS[i - 1][0], cols[j])]["auroc"] == best[cols[j]]:
                c.set_text_props(weight="bold")
        skipped = [t for t in TASKS if t not in cols]
        ax.set_title(f"{title}: AUROC [95% CI]"
                     + (f"\n({', '.join(skipped)}: no hallucinated window under this label, excluded)"
                        if skipped else ""), fontsize=10)
        fig.savefig(os.path.join(a.out, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {fname}")

        # ... and its ROC curves: all tasks pooled (one global threshold) + one panel per task
        pool = [s for n in scopes[scope] for s in seqs[n] if s["lab"][lab] is not None]
        panels = [(POOLED, pool)] + [(t, [s for s in pool if s["task"] == t]) for t in cols[:-2]]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 7.4), squeeze=False)
        for ax, (t, T) in zip(axes[0], panels):
            y = np.array([s["lab"][lab] for s in T])
            for key, name, col, ls in CURVES:
                if name not in cell_names:
                    continue
                v = np.array([s["sc"].get(key, np.nan) for s in T])
                r = cell[(name, t)]
                au, lo, hi = r["auroc"], r["ci_lo"], r["ci_hi"]
                fpr, tpr = roc_curve(v, y)
                ax.plot(fpr, tpr, color=col, ls=ls, lw=2.0, label=f"{name}  {au:.2f} [{lo:.2f}, {hi:.2f}]")
            ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8, ls=":")
            ax.set_title(f"{t}{' (one global threshold)' if t == POOLED else ''}\n"
                         f"n={len(T)}, {int(y.sum())} hallucinated", fontsize=10)
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_aspect("equal")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8, frameon=False,
                      title="AUROC [95% CI]", title_fontsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle(f"{title}: ROC" + (f" ({', '.join(skipped)}: no hallucinated window, only in the pooled "
                                         f"panel, as negatives)" if skipped else ""), fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        rname = fname.replace(".png", "_roc.png")
        fig.savefig(os.path.join(a.out, rname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {rname}")


if __name__ == "__main__":
    main()
