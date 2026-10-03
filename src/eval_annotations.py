# eval_annotations.py
"""
Evaluate hallucination metrics against the manual annotations.

Inputs, per annotation set <set> (logs/annotation/<set>):
    key.csv                       id -> task, episode, ...
    annotations/*.json            the annotator export(s); the newest row per id wins
    _hidden_metrics/mask_<set>/   per-frame CSVs from mask_metrics.py --from_npz
    _hidden_metrics/dino_<set>/   per-frame CSVs from dino_metrics.py --from_npz
    _hidden_metrics/clef_<set>/   per-frame CSVs from clef_metrics.py (optional)
Labels: verdict yes = 1, no = 0; unsure is dropped.

Per sequence, every per-frame metric is summarized over the imagined frames as
    mean    mean over all imagined frames
    last4   mean over the last 4 imagined frames
    worst   the worst frame (max, or min for metrics where lower = worse)
and oriented so that higher = more hallucinated. The paper's dPSNR (pooled over
the horizon, from the per-frame PSNRs) is included as a baseline, both as a
score and as its binary label (dPSNR <= 0).

Per task (all sets pooled, and per set):
    AUROC with a 95% CI from a cluster bootstrap over episodes (windows of the
          same episode are correlated)
    threshold: the one maximizing balanced accuracy on all data, and the
          balanced accuracy of a threshold chosen that way under grouped
          k-fold cross-validation (folds split by episode), i.e. an honest
          estimate of how well a per-task threshold transfers
    Spearman correlation between the metrics (mean summary)

Run from src/:
    python eval_annotations.py --sets dmc_v1 dmc_v2 --out ../../logs/annotation/eval
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

ROOT = "../../logs/annotation"
# per-frame column -> (source, direction): +1 = higher means more hallucinated
METRICS = {
    "iou":          ("mask", "iou_wm", -1),
    "hd95":         ("mask", "hd95_wm", +1),
    "hausdorff":    ("mask", "hausdorff_wm", +1),
    "chamfer":      ("mask", "chamfer_wm", +1),
    "psnr_masked":  ("mask", "psnr_m_wm", -1),
    "lpips_masked": ("mask", "lpips_m_wm", +1),
    "dino_reg_recon":    ("dino", "base_reg_recon_mean", +1),
    "dino_base_recon":   ("dino", "base_recon_mean", +1),
    "dino_reg_real":     ("dino", "base_reg_real_mean", +1),
    "dino_reg_recon_fg": ("dino", "base_reg_recon_fg", +1),
    "dino_reg_recon_p95": ("dino", "base_reg_recon_p95", +1),
}
AGGS = ("mean", "last4", "worst")
SIGN = {**{m: v[2] for m, v in METRICS.items()}, "paper_dpsnr": -1, "paper_label": +1}   # for display
N_BOOT = 1000                     # cluster-bootstrap draws (--n_boot)


# ------------------------------------------------------------------ loading

def load_set(name, root):
    sdir = os.path.join(root, name)
    key = {r["id"]: r for r in csv.DictReader(open(os.path.join(sdir, "key.csv")))}
    labels = {}
    for p in sorted(glob.glob(os.path.join(sdir, "annotations", "*.json"))):
        for r in json.load(open(p))["rows"]:
            old = labels.get(r["id"])
            if old is None or (r.get("updated") or "") >= (old.get("updated") or ""):
                labels[r["id"]] = r
    seqs = []
    for sid, k in key.items():
        a = labels.get(sid)
        if a is None or a.get("verdict") not in ("yes", "no"):
            continue
        frames = {}
        for src in ("mask", "dino", "dinovar", "pred", "clef"):
            p = os.path.join(root, "_hidden_metrics", f"{src}_{name}", f"{sid}_{k['task']}.csv")
            if os.path.exists(p):
                frames[src] = list(csv.DictReader(open(p)))
        if not frames:
            continue
        seqs.append({"set": name, "id": sid, "task": k["task"], "group": f"{name}:{k['episode']}",
                     "y": int(a["verdict"] == "yes"), "types": a.get("types", []),
                     "onset": a.get("onset"), "frames": frames})
    return seqs


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def scores(seq):
    """{(metric, agg): oriented score} for one sequence."""
    out = {}
    for m, (src, col, sign) in METRICS.items():
        rows = seq["frames"].get(src)
        if not rows or col not in rows[0]:
            continue
        v = np.array([fnum(r[col]) for r in rows]) * sign
        if not np.isfinite(v).any():
            continue
        out[(m, "mean")] = np.nanmean(v)
        out[(m, "last4")] = np.nanmean(v[-4:]) if np.isfinite(v[-4:]).any() else np.nan
        out[(m, "worst")] = np.nanmax(v)
    # Column-wise sources, every column higher = worse: dino_patch_variants.py ("dv:<col>"),
    # the paper's predictors from paper_predictors.py ("pred:<col>") and Clef VLM scores
    # from clef_metrics.py ("clef:<model>_<question>").
    for src, prefix in (("dinovar", "dv:"), ("pred", "pred:"), ("clef", "clef:")):
        rows = seq["frames"].get(src) or []
        for col in (rows[0] if rows else []):
            if col in ("t", "paper_psnr_wm", "paper_psnr_copy"):
                continue
            v = np.array([fnum(r[col]) for r in rows])
            out[(prefix + col, "mean")] = np.nanmean(v)
            out[(prefix + col, "last4")] = np.nanmean(v[-4:])
            out[(prefix + col, "worst")] = np.nanmax(v)
    rows = (seq["frames"].get("mask") or seq["frames"].get("dino") or seq["frames"].get("dinovar")
            or seq["frames"].get("pred"))
    pw = np.array([fnum(r["paper_psnr_wm"]) for r in rows])
    pc = np.array([fnum(r["paper_psnr_copy"]) for r in rows])
    mse = lambda p: np.mean(10 ** (-p / 10))
    dpsnr = 10 * np.log10(mse(pc) / mse(pw))
    out[("paper_dpsnr", "pooled")] = -dpsnr                    # lower dPSNR = worse
    out[("paper_label", "dpsnr<=0")] = float(dpsnr <= 0)
    return out


# ------------------------------------------------------------------ statistics

def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(s)                                         # average ranks for ties
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cluster_boot(s, y, g, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in groups}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(groups, len(groups), replace=True)])
        a = auroc(s[pick], y[pick])
        if np.isfinite(a):
            vals.append(a)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def bal_acc(pred, y):
    tpr = pred[y == 1].mean() if (y == 1).any() else np.nan
    tnr = (~pred[y == 0]).mean() if (y == 0).any() else np.nan
    return np.nanmean([tpr, tnr]), tpr, tnr


def best_threshold(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    cand = np.unique(s)
    if len(cand) > 400:
        cand = np.quantile(s, np.linspace(0, 1, 401))
    best = (-1, np.nan)
    for c in cand:
        b = bal_acc(s >= c, y)[0]
        if b > best[0]:
            best = (b, c)
    return best[1]


def cv_bal_acc(s, y, g, k=5, seed=0):
    """Grouped k-fold: threshold fit on the other folds, balanced accuracy on
    the held-out fold's predictions pooled over folds."""
    rng = np.random.default_rng(seed)
    groups = np.unique(g)
    rng.shuffle(groups)
    folds = np.array_split(groups, min(k, len(groups)))
    pred = np.zeros(len(s), bool)
    for f in folds:
        te = np.isin(g, f)
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            return np.nan
        thr = best_threshold(s[tr], y[tr])
        pred[te] = s[te] >= thr
    ok = np.isfinite(s)
    return bal_acc(pred[ok], y[ok])[0]


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return np.corrcoef(ra, rb)[0, 1] if ok.sum() > 2 else np.nan


# ------------------------------------------------------------------ main

def evaluate(seqs, title, out_rows):
    y = np.array([q["y"] for q in seqs])
    g = np.array([q["group"] for q in seqs])
    sc = [scores(q) for q in seqs]
    keys = sorted({k for s in sc for k in s}, key=lambda k: (k[0].startswith("paper"), k[0], k[1]))
    print(f"\n=== {title}: n={len(y)}  yes={y.sum()}  no={len(y) - y.sum()}  episodes={len(np.unique(g))}")
    if y.sum() == 0 or y.sum() == len(y):
        print("   single class: AUROC undefined")
        return {}
    res = {}
    for k in keys:
        s = np.array([d.get(k, np.nan) for d in sc])
        a = auroc(s, y)
        lo, hi = cluster_boot(s, y, g, n=N_BOOT)
        thr = best_threshold(s, y)
        ba, tpr, tnr = bal_acc(s >= thr, y) if np.isfinite(thr) else (np.nan,) * 3
        cv = cv_bal_acc(s, y, g)
        sign = SIGN.get(k[0], 1)
        res[k] = dict(auroc=a, lo=lo, hi=hi, thr=thr * sign if np.isfinite(thr) else np.nan,
                      bal_acc=ba, tpr=tpr, tnr=tnr, cv_bal_acc=cv, s=s)
        out_rows.append({"subset": title, "metric": k[0], "agg": k[1], "n": len(y), "yes": int(y.sum()),
                         "auroc": a, "auroc_lo": lo, "auroc_hi": hi,
                         "threshold_raw": res[k]["thr"], "direction": ">=" if sign > 0 else "<=",
                         "bal_acc_fit": ba, "tpr": tpr, "tnr": tnr, "bal_acc_cv": cv})
    ranked = sorted(res.items(), key=lambda kv: -np.nan_to_num(kv[1]["auroc"], nan=-1))
    print(f"   {'metric':20s} {'agg':9s} {'AUROC [95% CI]':22s} {'threshold':>14s} {'balAcc fit':>10s} {'TPR/TNR':>11s} {'balAcc CV':>9s}")
    for (m, agg), r in ranked:
        d = ">=" if SIGN.get(m, 1) > 0 else "<="
        thr = f"{d} {r['thr']:.4g}" if np.isfinite(r["thr"]) else "-"
        print(f"   {m:20s} {agg:9s} {r['auroc']:.3f} [{r['lo']:.2f}, {r['hi']:.2f}]      {thr:>14s} "
              f"{r['bal_acc']:10.3f} {r['tpr']:5.2f}/{r['tnr']:.2f} {r['cv_bal_acc']:9.3f}")
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", default=["dmc_v1", "dmc_v2"])
    p.add_argument("--root", default=ROOT)
    p.add_argument("--out", default=os.path.join(ROOT, "eval"))
    p.add_argument("--n_boot", type=int, default=1000, help="cluster-bootstrap draws for the AUROC CI")
    args = p.parse_args()
    global N_BOOT
    N_BOOT = args.n_boot

    seqs = [q for s in args.sets for q in load_set(s, args.root)]
    os.makedirs(args.out, exist_ok=True)
    out_rows, corr_rows = [], []
    for task in sorted({q["task"] for q in seqs}):
        tq = [q for q in seqs if q["task"] == task]
        res = evaluate(tq, f"{task} [{'+'.join(args.sets)}]", out_rows)
        for s in args.sets:
            sq = [q for q in tq if q["set"] == s]
            if sq and len(sq) != len(tq):
                evaluate(sq, f"{task} [{s}]", out_rows)
        # correlations between metrics (mean summary), pooled sets
        names = [k for k in res if k[1] in ("mean", "pooled")]
        if names:
            print(f"   Spearman ({task}, mean summary):")
            print("   " + " " * 20 + "".join(f"{k[0][:10]:>11s}" for k in names))
            for a in names:
                row = [spearman(res[a]["s"], res[b]["s"]) for b in names]
                corr_rows.append({"task": task, "metric": a[0], **{b[0]: v for b, v in zip(names, row)}})
                print(f"   {a[0]:20s}" + "".join(f"{v:11.2f}" for v in row))
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    if corr_rows:
        keys = sorted({k for r in corr_rows for k in r}, key=lambda k: (k not in ("task", "metric"), k))
        with open(os.path.join(args.out, "correlations.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(corr_rows)
    print(f"\n[out] {args.out}/results.csv, correlations.csv")


if __name__ == "__main__":
    main()
