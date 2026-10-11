# plan_figures.py
"""
Figures (PNG) + the exact plotted numbers (CSV) for the closed-loop planner
study, straight from the run directories (episode JSONs; CPU, seconds).

CONDITIONS come from a manifest, one per line: name, arm, run-dir globs
(relative to the current dir; '#' starts a comment). arm = the <arm>_viz
subdir holding the episodes (bc | cem):

    bc      bc   logs/plan/cup-catch_27497026 ...
    reward  cem  logs/plan/cup-catch_2754154[23] ...
    value   cem  logs/plan/cup-catch_2755523[35] ...

    cd dreamer4 && python mmbench2/src/plan_figures.py \
        --manifest logs/plan_figures/manifests/main.txt --out logs/plan_figures/main

Track A (WM planner vs the same CEM in the simulator; only conditions run with
--compare_sim). Per replan, regret_K = true reward the sim plan collects over
the K executed steps minus what the WM plan collects, both run in the
simulator from the same real state. It is reported as % of a fixed per-task
scale: the sim plan's mean K-step return over all of that task's replans
(every condition pooled; SCALE in the CSVs), so tasks share one axis. A fixed
scale, not the return inside each bin: cup-catch's sparse reward makes the
latter zero in many bins.
  A1_regret_vs_hausdorff   regret vs the Hausdorff of the WM plan's rollout
                           (equal-count bins per task), one line per condition
  A2_regret_by_label       regret of not-hallucinated vs hallucinated replans
  A3_predictors            uncertainty predictors on the WM plan's rollout:
                           AUROC vs the hallucination label | Spearman with regret
  A4_search_vs_baselines   regret of the WM plan vs the sim planner's own
                           seed-to-seed spread (|regret| of its rerun) and vs
                           the BC warm start (no search). Only runs that log
                           the rerun / BC plans contribute to those bars.
Track B (hallucination of the executed plan, every condition), per 16-step
chunk: hallucinated = mean Hausdorff (real frame vs WM prediction) >= task
threshold (plan_halluc.THRESH).
  B1_halluc_rate           chunk hallucination rate per condition
  B2_rate_vs_motion        rate vs the real latent motion (equal-count bins)
  B3_predictors            AUROC of the predictors vs the chunk label
Track C (task performance; only with --score_manifest, which adds conditions
scored here but kept out of A/B: the uniform-action floor and the exec=sim
oracles). Normalized score s = (R - R_floor) / (R_ceil - R_floor) per task,
floor = --floor (random policy), ceil = --ceil (the same CEM planning in the
simulator, executed): 1 - s of a WM planner = the share of the achievable
return lost to model error. Each condition is compared with the ceiling on
their shared env seeds (paired bootstrap); the floor is resampled on its own.
One scale for every condition: 0 = random, 1 = oracle_reward. --oracle_of
pairs each WM planner condition with the oracle of the same objective
(reward -> oracle_reward, value -> oracle_value).
  C1_normalized_score      score per condition; oracles as dashed segments
  C2_gain_vs_loss          per WM planner: gained by search over BC (solid,
                           BC -> WM) vs lost to the model (hatched, WM -> its
                           oracle), on seeds all three share
  C3_halluc_vs_score       episode score vs the episode's chunk hallucination
                           rate as a percentile within its task, all tasks
                           pooled (5 bins + episode dots); WM planners
                           (reward + value pooled) vs their oracles (oracle
                           falling too = hard states, not the WM's planning)
  C3b_halluc_vs_score_per_task  the same per task (terciles of the raw rate);
                           thin until there are more episodes / tasks
  C4_halluc_wm_vs_oracle   WM hallucination on the executed plan when the WM
                           chose it vs when its oracle chose it (paired seeds)
  C5_regret_vs_episode_loss  per episode, in score units: summed per-replan
                           regret_K of the WM run vs the return it actually
                           lost to its oracle (same seed). loss > regret =
                           errors compound; loss < regret = episodes recover
  C6_exploitation_same_state  per replan of the WM runs, from the same state:
                           WM hallucination on the plan the WM chose vs on the
                           plan the sim planner chose (paired)

    python mmbench2/src/plan_figures.py --manifest logs/plan_figures/manifests/main.txt \
        --score_manifest logs/plan_figures/manifests/score.txt --out logs/plan_figures/main

Bar charts: one panel, tasks side by side, the first group = the mean over
tasks (equal weight). All CIs: 95%, bootstrap over episodes (resampled within
each task; the average uses the same draws).
"""
import argparse
import csv
import glob
import itertools
import json
import os
import textwrap
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from plan_halluc import THRESH, auroc, episode_rows  # noqa: E402
from plan_regret import replan_rows  # noqa: E402

# Validated default categorical palette (light surface), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, MID = "#0b0b0b", "#52514e", "#e4e3df", "#e9e8e4"
HATCHES = ["", "///", "...", "xxx"]           # second encoding for the condition
PREDICTORS = ["u_r", "u_f", "u_s", "u_r_norm", "u_f_norm", "u_s_norm"]
N_BOOT = 2000
AVG = "average"


# ------------------------------------------------------------------ data
def read_manifest(path):
    conds = []
    for ln in open(path):
        parts = ln.split("#", 1)[0].split()
        if not parts:
            continue
        name, arm, pats = parts[0], parts[1], parts[2:]
        runs = sorted({d for p in pats for d in glob.glob(p) if os.path.isdir(d)})
        if not runs:
            raise SystemExit(f"condition {name!r}: no run directories match {pats}")
        conds.append((name, arm, runs))
    return conds


def load(conds):
    """-> replans (track A rows), chunks (track B rows), each tagged 'cond'."""
    replans, chunks = [], []
    for name, arm, runs in conds:
        for run_dir in runs:
            run = os.path.basename(os.path.normpath(run_dir))
            for jp in sorted(glob.glob(os.path.join(run_dir, f"{arm}_viz", "ep*_seed*.json"))):
                task = json.load(open(jp))["task"]
                for r in episode_rows(jp, run, arm, task, source="json"):
                    if r["halluc"] != "":
                        r["cond"] = name
                        chunks.append(r)
                if arm == "cem":
                    for r in replan_rows(jp, run):
                        r["cond"] = name
                        replans.append(r)
    return replans, chunks


def by_task_episode(rows, cond):
    """{task: [rows of one episode, ...]} for one condition."""
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["cond"] == cond:
            d[r["task"]][(r["run"], r["episode"])].append(r)
    return {t: list(eps.values()) for t, eps in d.items()}


def boot(groups, stat, tasks, seed=0):
    """groups: {task: [episode row lists]}; stat(list of rows) -> float.
    Returns {task or AVG: (mean, lo, hi)}; AVG = mean over tasks present,
    bootstrapped with the same draws."""
    rng = np.random.default_rng(seed)
    tasks = [t for t in tasks if groups.get(t)]
    point = {t: stat([r for e in groups[t] for r in e]) for t in tasks}
    draws = {t: [] for t in tasks}
    for _ in range(N_BOOT):
        for t in tasks:
            eps = groups[t]
            pick = [r for i in rng.integers(0, len(eps), len(eps)) for r in eps[i]]
            draws[t].append(stat(pick))
    out = {t: (point[t], *np.nanpercentile(draws[t], [2.5, 97.5])) for t in tasks}
    if tasks:
        avg = np.nanmean([draws[t] for t in tasks], axis=0)
        out[AVG] = (float(np.nanmean([point[t] for t in tasks])), *np.nanpercentile(avg, [2.5, 97.5]))
    return out


SCALE = {}   # task -> mean K-step return of the sim plan (set in track_a)


def pct(num):
    """100 * mean(num) / SCALE[task] (rows of one task)."""
    def f(rows):
        v = np.array([r.get(num, np.nan) for r in rows], float)
        v = v[np.isfinite(v)]
        return 100.0 * v.mean() / SCALE[rows[0]["task"]] if len(v) else np.nan
    return f


def mean_of(key):
    def f(rows):
        v = np.array([r.get(key, np.nan) for r in rows], float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else np.nan
    return f


# ------------------------------------------------------------------ plot helpers
def style(ax, ygrid=True, xgrid=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def write_csv(path, rows):
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def wrap_title(title, fig):
    """Wrap to the figure width (~0.075 in per character at fontsize 10), so a
    long title does not widen the saved canvas and squeeze the axes."""
    return "\n".join(textwrap.wrap(title, max(40, int(fig.get_figwidth() / 0.075))))


def legend_cols(handles):
    longest = max((len(h.get_label()) for h in handles), default=0)
    return min(len(handles), 4 if longest <= 18 else 2)


def bar_chart(path, groups, bars, values, ylabel, title, ylim=None, refs=None):
    """One panel. groups: x groups (AVG first); bars: [(label, color, hatch)];
    values[(group, bar_label)] = (m, lo, hi). refs: [(label, color, {group: y})]
    or (label, color, {group: y}, [bar labels]): dashed segments across a group's
    width, or only across the named bars."""
    n = len(bars)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(1.0 + 1.25 * len(groups) * max(1, n / 3), 3.8))
    for j, (lab, color, hatch) in enumerate(bars):
        xs, ms, err = [], [], [[], []]
        for i, g in enumerate(groups):
            m, lo, hi = values.get((g, lab), (np.nan,) * 3)
            if not np.isfinite(m):
                continue
            xs.append(i - 0.4 + w * (j + 0.5))
            ms.append(m)
            err[0].append(max(m - lo, 0) if np.isfinite(lo) else 0)
            err[1].append(max(hi - m, 0) if np.isfinite(hi) else 0)
        ax.bar(xs, ms, width=w * 0.92, color=color, hatch=hatch, edgecolor="white", linewidth=0, zorder=2)
        ax.errorbar(xs, ms, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
    labels = [b[0] for b in bars]
    for lab, color, ys, *span in refs or []:
        for i, g in enumerate(groups):
            if g in ys and np.isfinite(ys[g]):
                if span:
                    js = [labels.index(b) for b in span[0]]
                    x0, x1 = i - 0.4 + w * min(js), i - 0.4 + w * (max(js) + 1)
                else:
                    x0, x1 = i - 0.44, i + 0.44
                ax.plot([x0, x1], [ys[g]] * 2, color=color, lw=1.6, ls="--", zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g if g != AVG else "mean over tasks" for g in groups])
    if AVG in groups:
        ax.axvline(0.5, color=GRID, lw=1.2)
    ax.axhline(0, color=INK2, lw=0.8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(wrap_title(title, fig), color=INK, fontsize=10, loc="left")
    style(ax)
    handles = [Patch(facecolor=c, hatch=h, edgecolor="white", label=lab) for lab, c, h in bars]
    for lab, c in dict((r[0], r[1]) for r in refs or []).items():
        handles.append(plt.Line2D([], [], color=c, lw=1.6, ls="--", label=lab))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=legend_cols(handles),
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def line_panels(path, tasks, series, xlabel, ylabel, title, vlines=None, ylim=None):
    """series[task] = [(label, color, xs, ms, los, his)]. xlabel: str or
    {panel: str}. An AVG panel in tasks is drawn first, set off by a rule."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.7 * len(tasks) + 0.4, 3.4), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        if t == AVG:
            ax.set_facecolor("#f6f6f4")
        for lab, c, xs, ms, lo, hi in series.get(t, []):
            ax.fill_between(xs, lo, hi, color=c, alpha=0.15, lw=0)
            ax.plot(xs, ms, color=c, lw=2, label=lab)
            ax.scatter(xs, ms, s=40, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        if vlines and vlines.get(t) is not None:
            ax.axvline(vlines[t], color=INK2, lw=1, ls="--")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title("mean over tasks" if t == AVG else t, color=INK, fontsize=10)
        ax.set_xlabel(xlabel.get(t, xlabel.get("*", "")) if isinstance(xlabel, dict) else xlabel, color=INK2)
        style(ax, xgrid=True)
    axes[0][0].set_ylabel(ylabel, color=INK2)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(h), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def heatmaps(path, panels, title):
    """panels: [(subtitle, rows, cols, Z, vmin, vmax, col_labels)]."""
    cmap = LinearSegmentedColormap.from_list("div", [SERIES[1], MID, SERIES[0]])
    widths = [len(p[2]) for p in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(0.95 * sum(widths) + 2.2 * len(panels), 0.45 * len(panels[0][1]) + 2.0),
                             squeeze=False, gridspec_kw={"width_ratios": widths})
    for ax, (sub, rows, cols, Z, vmin, vmax, clabs) in zip(axes[0], panels):
        ax.imshow(np.ma.masked_invalid(Z), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i, j in itertools.product(range(len(rows)), range(len(cols))):
            if np.isfinite(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(clabs, fontsize=8, color=INK2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=9, color=INK2)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(sub, color=INK, fontsize=10)
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def binned_line(rows, conds, tasks, xkey, stat, n_bins, seed=0, avg_x=None):
    """Equal-count bins of xkey per task (pooled over conditions). Returns
    (series for line_panels, csv rows). stat(rows) -> float.
    Also an AVG series: bin b averaged over tasks, only where every task has
    >= 3 rows of that condition in bin b (otherwise the "average" would be one
    task's value, e.g. CEM has almost no low-motion chunks) (the bins
    are equal-count per task, so bin b is the same quantile everywhere), CI
    from one joint bootstrap; its x = mean over tasks of avg_x(task, bin
    median), or the bin number (1..n_bins) when avg_x is None."""
    series, out = defaultdict(list), []
    per_bin = defaultdict(dict)          # (cond, b) -> {task: (episode lists, x)}
    for t in tasks:
        tr = [r for r in rows if r["task"] == t and np.isfinite(r.get(xkey, np.nan))]
        if len(tr) < n_bins:
            continue
        edges = np.quantile([r[xkey] for r in tr], np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        for k, c in enumerate(conds):
            xs, ms, los, his = [], [], [], []
            for b in range(n_bins):
                g = [r for r in tr if r["cond"] == c and edges[b] <= r[xkey] < edges[b + 1]]
                if len(g) < 3:
                    continue
                eps = defaultdict(list)
                for r in g:
                    eps[(r["run"], r["episode"])].append(r)
                m, lo, hi = boot({t: list(eps.values())}, stat, [t], seed)[t]
                x = float(np.median([r[xkey] for r in g]))
                per_bin[(c, b)][t] = (list(eps.values()), x)
                xs.append(x), ms.append(m), los.append(lo), his.append(hi)
                out.append({"task": t, "condition": c, "bin": b, "bin_lo": edges[b], "bin_hi": edges[b + 1],
                            f"{xkey}_median": x, "n": len(g), "n_episodes": len(eps), "mean": m, "ci_lo": lo, "ci_hi": hi})
            if xs:
                series[t].append((c, SERIES[k % len(SERIES)], xs, ms, los, his))
    for k, c in enumerate(conds):
        xs, ms, los, his = [], [], [], []
        for b in range(n_bins):
            d = per_bin.get((c, b), {})
            if set(d) != {t for t in tasks if any(tt == t for (cc, _), v in per_bin.items() if cc == c for tt in v)}:
                continue
            m, lo, hi = boot({t: v[0] for t, v in d.items()}, stat, list(d), seed)[AVG]
            x = float(np.mean([avg_x(t, v[1]) for t, v in d.items()])) if avg_x else b + 1.0
            xs.append(x), ms.append(m), los.append(lo), his.append(hi)
            out.append({"task": AVG, "condition": c, "bin": b, "x_avg": x, "tasks": " ".join(d),
                        "mean": m, "ci_lo": lo, "ci_hi": hi})
        if xs:
            series[AVG].append((c, SERIES[k % len(SERIES)], xs, ms, los, his))
    return series, out


def predictor_panel(rows, conds, tasks, label_key, score):
    """Z[pred, (task, cond)] = score(predictor values, rows)."""
    cols = [(t, c) for t in tasks for c in conds
            if any(r["task"] == t and r["cond"] == c and np.isfinite(r.get("u_r", np.nan)) for r in rows)]
    Z = np.full((len(PREDICTORS), len(cols)), np.nan)
    out = []
    for j, (t, c) in enumerate(cols):
        g = [r for r in rows if r["task"] == t and r["cond"] == c and np.isfinite(r.get("u_r", np.nan))]
        for i, p in enumerate(PREDICTORS):
            Z[i, j] = score([r.get(p, np.nan) for r in g], g)
            out.append({"task": t, "condition": c, "predictor": p, label_key: Z[i, j], "n": len(g)})
    return cols, Z, out


# ------------------------------------------------------------------ figures
def track_a(replans, conds, tasks, out):
    conds = [c for c in conds if any(r["cond"] == c for r in replans)]
    if not conds:
        return []
    made = []
    groups = [AVG] + tasks
    for t in tasks:
        v = [r["Rsim_K_sim"] for r in replans if r["task"] == t and np.isfinite(r.get("Rsim_K_sim", np.nan))]
        SCALE[t] = float(np.mean(v))
    write_csv(os.path.join(out, "A_scale.csv"), [{"task": t, "scale_mean_Rsim_K_sim": SCALE[t]} for t in tasks])

    # A1: regret vs Hausdorff, lines.
    series, rows = binned_line(replans, conds, tasks, "hd_wm", pct("regret_K"), n_bins=5,
                               avg_x=lambda t, x: x / THRESH[t])
    write_csv(os.path.join(out, "A1_regret_vs_hausdorff.csv"), rows)
    p = os.path.join(out, "A1_regret_vs_hausdorff.png")
    line_panels(p, [AVG] + tasks, series,
                {AVG: "Hausdorff / task threshold (bin median)", "*": "Hausdorff of the WM plan's rollout (px, bin median)"},
                "regret (% of task scale)",
                "Regret of the WM plan vs how much its rollout hallucinates (dashed: hallucination threshold; "
                "task scale = sim plan's mean K-step return)",
                vlines={**THRESH, AVG: 1.0})
    made.append(p)

    # A2: regret by hallucination label, bars.
    vals, rows, bars = {}, [], []
    for k, c in enumerate(conds):
        for y, lab, color in ((0, "not hallucinated", SERIES[0]), (1, "hallucinated", SERIES[1])):
            name = f"{c}: {lab}"
            bars.append((name, color, HATCHES[k % len(HATCHES)]))
            sub = [r for r in replans if np.isfinite(r.get("halluc_wm", np.nan)) and r["halluc_wm"] == y]
            res = boot(by_task_episode(sub, c), pct("regret_K"), tasks)
            for g, v in res.items():
                vals[(g, name)] = v
                rows.append({"group": g, "condition": c, "halluc_wm": y, "mean": v[0], "ci_lo": v[1], "ci_hi": v[2],
                             "n_replans": sum(1 for r in sub if r["cond"] == c and (g == AVG or r["task"] == g))})
    write_csv(os.path.join(out, "A2_regret_by_label.csv"), rows)
    p = os.path.join(out, "A2_regret_by_label.png")
    bar_chart(p, groups, bars, vals, "regret (% of task scale)",
              "Do hallucinations change the planner's action choice? Return the WM plan loses vs the "
              "sim plan from the same state, split by whether its rollout hallucinated")
    made.append(p)

    # A3: predictors.
    lab_rows = [r for r in replans if np.isfinite(r.get("halluc_wm", np.nan))]
    cols, Za, ra = predictor_panel(lab_rows, conds, tasks, "auroc",
                                   lambda s, g: auroc(s, [r["halluc_wm"] for r in g]))

    def spear(s, g):
        a = np.array([[x, r.get("regret_K", np.nan)] for x, r in zip(s, g)], float)
        a = a[np.isfinite(a).all(1)]
        return float(spearmanr(a[:, 0], a[:, 1])[0]) if len(a) > 5 else np.nan
    cols_s, Zs, rs = predictor_panel(replans, conds, tasks, "spearman", spear)
    write_csv(os.path.join(out, "A3_predictors.csv"), ra + rs)
    p = os.path.join(out, "A3_predictors.png")
    heatmaps(p, [("AUROC vs hallucination label (0.5 = chance)", PREDICTORS, cols, Za, 0.2, 0.8,
                  [f"{t}\n{c}" for t, c in cols]),
                 ("Spearman correlation with regret (0 = none)", PREDICTORS, cols_s, Zs, -0.6, 0.6,
                  [f"{t}\n{c}" for t, c in cols_s])],
             "Uncertainty predictors on the WM plan's rollout (orange = wrong direction, blue = right direction)")
    made.append(p)

    # A4: search vs baselines.
    for r in replans:
        r["_abs_floor"] = abs(r.get("regret_floor_K", np.nan))
    vals, rows, bars = {}, [], []
    for k, c in enumerate(conds):
        for key, lab, color in (("regret_K", "WM plan", SERIES[0]),
                                ("_abs_floor", "sim planner vs its own rerun", SERIES[1]),
                                ("regret_bc_K", "BC warm start (no search)", SERIES[2])):
            name = f"{c}: {lab}"
            sub = [r for r in replans if np.isfinite(r.get(key, np.nan))]
            res = boot(by_task_episode(sub, c), pct(key), tasks)
            if not res:
                continue
            bars.append((name, color, HATCHES[k % len(HATCHES)]))
            for g, v in res.items():
                vals[(g, name)] = v
                rows.append({"group": g, "condition": c, "series": lab, "mean": v[0], "ci_lo": v[1], "ci_hi": v[2],
                             "n_replans": sum(1 for r in sub if r["cond"] == c and (g == AVG or r["task"] == g))})
    write_csv(os.path.join(out, "A4_search_vs_baselines.csv"), rows)
    p = os.path.join(out, "A4_search_vs_baselines.png")
    bar_chart(p, groups, bars, vals, "return lost (% of task scale)",
              "Return lost vs the sim planner: by the WM plan, by the sim planner rerun, by the BC plan")
    made.append(p)
    return made


def track_b(chunks, conds, tasks, out):
    made = []
    groups = [AVG] + tasks
    conds = [c for c in conds if any(r["cond"] == c for r in chunks)]

    vals, rows, bars = {}, [], []
    for k, c in enumerate(conds):
        bars.append((c, SERIES[k % len(SERIES)], ""))
        for g, v in boot(by_task_episode(chunks, c), mean_of("halluc"), tasks).items():
            vals[(g, c)] = v
            rows.append({"group": g, "condition": c, "rate": v[0], "ci_lo": v[1], "ci_hi": v[2],
                         "n_chunks": sum(1 for r in chunks if r["cond"] == c and (g == AVG or r["task"] == g))})
    write_csv(os.path.join(out, "B1_halluc_rate.csv"), rows)
    p = os.path.join(out, "B1_halluc_rate.png")
    bar_chart(p, groups, bars, vals, "fraction of 16-step chunks hallucinated",
              "Hallucination rate of the executed plan", ylim=(0, 1.02))
    made.append(p)

    series, rows = binned_line(chunks, conds, tasks, "motion_real", mean_of("halluc"), n_bins=5)
    write_csv(os.path.join(out, "B2_rate_vs_motion.csv"), rows)
    p = os.path.join(out, "B2_rate_vs_motion.png")
    line_panels(p, [AVG] + tasks, series,
                {AVG: "motion quintile of each task (1 = least)", "*": "real latent motion (bin median)"},
                "fraction hallucinated",
                "Hallucination rate vs how much the real scene moves", ylim=(0, 1.02))
    made.append(p)

    cols, Z, ra = predictor_panel(chunks, conds, tasks, "auroc", lambda s, g: auroc(s, [r["halluc"] for r in g]))
    write_csv(os.path.join(out, "B3_predictors.csv"), ra)
    p = os.path.join(out, "B3_predictors.png")
    npos = {(t, c): (sum(r["halluc"] for r in chunks if (r["task"], r["cond"]) == (t, c)),
                     sum(1 for r in chunks if (r["task"], r["cond"]) == (t, c))) for t, c in cols}
    heatmaps(p, [("AUROC vs chunk hallucination label (0.5 = chance)", PREDICTORS, cols, Z, 0.2, 0.8,
                  [f"{t}\n{c}\n{npos[(t, c)][0]}/{npos[(t, c)][1]} halluc." for t, c in cols])],
             "Uncertainty predictors on the executed plan's rollout (orange = wrong direction, blue = right direction)")
    made.append(p)
    return made


def load_returns(conds):
    """-> {cond: {task: {env_seed: episode return}}} from the episode JSONs."""
    out = defaultdict(lambda: defaultdict(dict))
    for name, arm, runs in conds:
        for run_dir in runs:
            for jp in sorted(glob.glob(os.path.join(run_dir, f"{arm}_viz", "ep*_seed*.json"))):
                ep = json.load(open(jp))
                seed = int(ep["env_seed"])
                if seed in out[name][ep["task"]]:
                    raise SystemExit(f"{name}: env_seed {seed} of {ep['task']} appears twice ({jp})")
                out[name][ep["task"]][seed] = float(ep["return"])
    return out


def paired_draws(rets, task, conds, floor, ceil, rng):
    """Normalized mean score of each of conds on the env seeds they all share
    with ceil (same initial states). Those seeds are resampled together; the
    floor is resampled on its own. -> (n_seeds, {cond: point}, {cond: draws})
    or None."""
    if any(task not in rets[c] for c in list(conds) + [floor, ceil]):
        return None
    seeds = sorted(set(rets[ceil][task]).intersection(*[rets[c][task] for c in conds]))
    if not seeds:
        return None
    rf = np.array(list(rets[floor][task].values()))
    rh = np.array([rets[ceil][task][s] for s in seeds])
    R = {c: np.array([rets[c][task][s] for s in seeds]) for c in conds}
    i = rng.integers(0, len(seeds), (N_BOOT, len(seeds)))
    j = rng.integers(0, len(rf), (N_BOOT, len(rf)))
    lo, hi = rf[j].mean(1), rh[i].mean(1)
    point = {c: (R[c].mean() - rf.mean()) / (rh.mean() - rf.mean()) for c in conds}
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = {c: (R[c][i].mean(1) - lo) / (hi - lo) for c in conds}
    return len(seeds), point, draws


def ci(d):
    return tuple(np.nanpercentile(d, [2.5, 97.5]))


def episode_rates(chunks, cond, task):
    """{env_seed: fraction of that episode's chunks hallucinated}."""
    eps = defaultdict(list)
    for r in chunks:
        if r["cond"] == cond and r["task"] == task:
            eps[int(r["env_seed"])].append(r["halluc"])
    return {s: float(np.mean(v)) for s, v in eps.items()}


def track_c(rets, chunks, replans, names, tasks, floor, ceil, oracle_of, baseline, out):
    """Normalized score s = (R - R_floor) / (R_ceil - R_floor), one scale for
    every condition (ceil = the reward-objective oracle). oracle_of maps a WM
    planner condition to the oracle with the same objective."""
    made = []
    rng = np.random.default_rng(0)
    groups = [AVG] + tasks
    color = {c: SERIES[k % len(SERIES)] for k, c in enumerate(names)}
    oracles = set(oracle_of.values()) | {ceil}
    scored = [c for c in names if c != floor and c not in oracles]
    planners = [c for c in scored if c in oracle_of]

    def avg_row(per):   # per: {task: (point, draws)}
        return (float(np.mean([m for m, _ in per.values()])),
                *ci(np.nanmean([d for _, d in per.values()], axis=0)))

    # C1: score per condition; oracles as reference segments
    vals, rows, refs = {}, [], []
    for c in scored + sorted(oracles - {ceil}):
        per = {}
        for t in tasks:
            res = paired_draws(rets, t, [c], floor, ceil, rng)
            if res is None:
                continue
            n, pt, dr = res
            per[t] = (pt[c], dr[c])
            vals[(t, c)] = (pt[c], *ci(dr[c]))
            rows.append({"group": t, "condition": c, "score": pt[c], "ci_lo": vals[(t, c)][1],
                         "ci_hi": vals[(t, c)][2], "n_seeds": n})
        if per:   # tasks a condition lacks are left out of its average (n_tasks)
            vals[(AVG, c)] = avg_row(per)
            rows.append({"group": AVG, "condition": c, "score": vals[(AVG, c)][0],
                         "ci_lo": vals[(AVG, c)][1], "ci_hi": vals[(AVG, c)][2], "n_tasks": len(per)})
    for t in tasks:
        if t in rets[floor] and t in rets[ceil]:
            rows.append({"group": t, "condition": f"anchors {floor} / {ceil}",
                         "R_floor_mean": float(np.mean(list(rets[floor][t].values()))),
                         "R_ceil_mean": float(np.mean(list(rets[ceil][t].values()))),
                         "n_floor": len(rets[floor][t]), "n_ceil": len(rets[ceil][t])})
    write_csv(os.path.join(out, "C1_normalized_score.csv"), rows)
    p = os.path.join(out, "C1_normalized_score.png")
    bars = [(c, color[c], "") for c in scored]
    refs = [(f"{ceil} (= 1)", color.get(ceil, INK2), {g: 1.0 for g in groups})]
    refs += [(o, color.get(o, INK2), {g: vals[(g, o)][0] for g in groups if (g, o) in vals})
             for o in sorted(oracles - {ceil})]
    bar_chart(p, groups, bars, vals, f"normalized score\n(0 = {floor}, 1 = {ceil})",
              "How well does each planner do? (dashed: oracle = the same CEM planning in the simulator)", refs=refs)
    made.append(p)

    # C2: BC -> WM planner -> its oracle, on seeds all three share
    rows, segs = [], {}
    for c in planners:
        o = oracle_of[c]
        per = {}
        for t in tasks:
            res = paired_draws(rets, t, [baseline, c, o], floor, ceil, rng)
            if res is None:
                continue
            n, pt, dr = res
            per[t] = (pt, dr, n)
        if not per:
            continue
        P = {k: float(np.mean([v[0][k] for v in per.values()])) for k in (baseline, c, o)}
        D = {k: np.nanmean([v[1][k] for v in per.values()], axis=0) for k in (baseline, c, o)}
        per[AVG] = (P, D, sum(v[2] for v in per.values()))
        for g, (pt, dr, n) in per.items():
            gain, loss = dr[c] - dr[baseline], dr[o] - dr[c]
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = loss / (dr[o] - dr[baseline])
            segs[(g, c)] = (pt[baseline], pt[c], pt[o], ci(dr[c]))
            rows.append({"group": g, "condition": c, "oracle": o, "n_seeds": n,
                         "s_" + baseline: pt[baseline], "s_wm": pt[c], "s_oracle": pt[o],
                         "gain_search": pt[c] - pt[baseline], "gain_ci_lo": ci(gain)[0], "gain_ci_hi": ci(gain)[1],
                         "lost_to_model": pt[o] - pt[c], "lost_ci_lo": ci(loss)[0], "lost_ci_hi": ci(loss)[1],
                         "frac_of_search_gain_lost": (pt[o] - pt[c]) / (pt[o] - pt[baseline])
                         if pt[o] != pt[baseline] else np.nan,
                         "frac_ci_lo": ci(frac)[0], "frac_ci_hi": ci(frac)[1]})
    write_csv(os.path.join(out, "C2_gain_vs_loss.csv"), rows)
    if segs:
        p = os.path.join(out, "C2_gain_vs_loss.png")
        n = len(planners)
        w = 0.8 / n
        fig, ax = plt.subplots(figsize=(1.0 + 1.25 * len(groups) * max(1, n / 3), 3.8))
        for j, c in enumerate(planners):
            for i, g in enumerate(groups):
                if (g, c) not in segs:
                    continue
                b, m, o, (lo, hi) = segs[(g, c)]
                x = i - 0.4 + w * (j + 0.5)
                ax.bar(x, m - b, bottom=b, width=w * 0.92, color=color[c], linewidth=0, zorder=2)
                ax.bar(x, o - m, bottom=m, width=w * 0.92, color="none", edgecolor=color[c],
                       hatch="///", linewidth=0.8, zorder=2)
                ax.plot([x - w * 0.46, x + w * 0.46], [b, b], color=INK, lw=1.2, zorder=3)
                ax.errorbar([x], [m], yerr=[[max(m - lo, 0)], [max(hi - m, 0)]], fmt="none",
                            ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g if g != AVG else "mean over tasks" for g in groups])
        ax.axvline(0.5, color=GRID, lw=1.2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel(f"normalized score\n(0 = {floor}, 1 = {ceil})", color=INK2)
        ax.set_title(wrap_title(f"How much of what search could gain does the model lose? (black tick = "
                                f"{baseline}; solid = gained, hatched = lost, up to the oracle)", fig),
                     color=INK, fontsize=10, loc="left")
        style(ax)
        handles = [Patch(facecolor=color[c], label=f"{c}: gained") for c in planners]
        handles += [Patch(facecolor="none", edgecolor=color[c], hatch="///", label=f"{c}: lost to model")
                    for c in planners]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=legend_cols(handles),
                  frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(p)

    # C3: episode score vs episode hallucination rate, equal-count bins per
    # task, WM planners (reward + value pooled) vs their oracles. A falling
    # oracle line too = states hard both to predict and to solve.
    def anchor_means(t):
        return np.mean(list(rets[floor][t].values())), np.mean(list(rets[ceil][t].values()))

    fams = [("WM planner", planners), ("oracle (same CEM in the simulator)",
                                       sorted({oracle_of[c] for c in planners}))]
    ep_rows = []
    for fam, cs in fams:
        for t in tasks:
            if t not in rets[floor] or t not in rets[ceil]:
                continue
            f_m, h_m = anchor_means(t)
            for c in cs:
                rates = episode_rates(chunks, c, t)
                ep_rows += [{"task": t, "cond": fam, "run": c, "episode": s, "env_seed": s, "rate": rates[s],
                             "score": (rets[c][t][s] - f_m) / (h_m - f_m)}
                            for s in sorted(rates) if s in rets.get(c, {}).get(t, {})]
    # Bins per family: oracle episodes hallucinate less, so shared bins would
    # leave each line with one or two points.
    series, rows = defaultdict(list), []
    for k, (fam, _) in enumerate(fams):
        ser, rw = binned_line([r for r in ep_rows if r["cond"] == fam], [fam], tasks, "rate",
                              mean_of("score"), n_bins=3)
        for t, lines in ser.items():
            series[t] += [(lab, SERIES[k], *rest) for lab, _, *rest in lines]
        rows += rw
    # C3b: per task (few episodes per bin for now; for when there are more tasks/seeds)
    write_csv(os.path.join(out, "C3b_halluc_vs_score_per_task.csv"), rows + ep_rows)
    if series:
        p = os.path.join(out, "C3b_halluc_vs_score_per_task.png")
        line_panels(p, [AVG] + tasks, series,
                    {AVG: "hallucination tercile (1 = least)", "*": "episode hallucination rate (bin median)"},
                    "episode normalized score",
                    f"Do episodes with more hallucination score lower? (score: 0 = {floor}, 1 = {ceil}; "
                    "episodes in equal-count bins per task and line)")
        made.append(p)

    # C3: all tasks pooled. x = the episode's hallucination percentile within
    # its task and line (ties share their mean rank), so tasks with different
    # base rates share one axis; bins = equal-count chunks of the episodes
    # sorted by percentile (ties sit at their shared mean rank).
    n_bins = 5
    for fam, _ in fams:
        for t in tasks:
            g = [r for r in ep_rows if r["cond"] == fam and r["task"] == t]
            if g:
                rk = rankdata([r["rate"] for r in g])
                for r, q in zip(g, rk):
                    r["pct"] = (q - 0.5) / len(g)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rows = []
    for k, (fam, _) in enumerate(fams):
        g = [r for r in ep_rows if r["cond"] == fam and "pct" in r]
        if not g:
            continue
        x = np.array([r["pct"] for r in g])
        y = np.array([r["score"] for r in g])
        ax.scatter(x, y, s=14, color=SERIES[k], alpha=0.25, linewidth=0, zorder=2)
        b = np.empty(len(x), int)
        for j, idx in enumerate(np.array_split(np.argsort(x, kind="stable"), n_bins)):
            b[idx] = j
        xs, ms, los, his = [], [], [], []
        for j in range(n_bins):
            v = y[b == j]
            if len(v) < 3:
                continue
            d = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
            xs.append(float(x[b == j].mean())), ms.append(float(v.mean()))
            los.append(ci(d)[0]), his.append(ci(d)[1])
            rows.append({"line": fam, "bin": j, "pct_mean": xs[-1], "n_episodes": int(len(v)),
                         "tasks": " ".join(sorted({r["task"] for r, bb in zip(g, b) if bb == j})),
                         "score": ms[-1], "ci_lo": los[-1], "ci_hi": his[-1]})
        rho = spearmanr(x, y)[0]
        rows.append({"line": fam, "bin": "spearman_all_episodes", "score": rho, "n_episodes": len(g)})
        lab = f"{fam} (n={len(g)} episodes, Spearman {rho:+.2f})"
        ax.fill_between(xs, los, his, color=SERIES[k], alpha=0.15, lw=0)
        ax.plot(xs, ms, color=SERIES[k], lw=2, label=lab, zorder=3)
        ax.scatter(xs, ms, s=40, color=SERIES[k], edgecolor="white", linewidth=1.5, zorder=4)
    write_csv(os.path.join(out, "C3_halluc_vs_score.csv"), rows)
    ax.set_xlim(0, 1)
    ax.set_xlabel("episode hallucination rate, percentile within its task (0 = least)", color=INK2)
    ax.set_ylabel(f"episode normalized score\n(0 = {floor}, 1 = {ceil})", color=INK2)
    ax.set_title(wrap_title("Do episodes with more hallucination score lower? (all tasks pooled; lines = "
                            f"{n_bins} equal-count bins, dots = episodes)", fig), color=INK, fontsize=10, loc="left")
    style(ax, xgrid=True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, "C3_halluc_vs_score.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    made.append(p)

    # C4: does the planner seek hallucination? WM rollout of the executed plan,
    # WM-chosen vs oracle-chosen, on shared seeds (paired)
    vals, rows, bars = {}, [], []
    for c in planners:
        o = oracle_of[c]
        bars += [(c, color[c], ""), (o, color[c], "///")]
        per = {}
        for t in tasks:
            rc, ro = episode_rates(chunks, c, t), episode_rates(chunks, o, t)
            seeds = sorted(set(rc) & set(ro))
            if not seeds:
                continue
            A, B = np.array([rc[s] for s in seeds]), np.array([ro[s] for s in seeds])
            i = rng.integers(0, len(seeds), (N_BOOT, len(seeds)))
            per[t] = ({c: A.mean(), o: B.mean()}, {c: A[i].mean(1), o: B[i].mean(1)}, len(seeds))
        if not per:
            continue
        per[AVG] = ({k: float(np.mean([v[0][k] for v in per.values()])) for k in (c, o)},
                    {k: np.mean([v[1][k] for v in per.values()], axis=0) for k in (c, o)},
                    sum(v[2] for v in per.values()))
        for g, (pt, dr, n) in per.items():
            for k in (c, o):
                vals[(g, k)] = (pt[k], *ci(dr[k]))
            diff = dr[c] - dr[o]
            rows.append({"group": g, "condition": c, "oracle": o, "n_seeds": n,
                         "rate_wm_plan": pt[c], "rate_oracle_plan": pt[o], "diff": pt[c] - pt[o],
                         "diff_ci_lo": ci(diff)[0], "diff_ci_hi": ci(diff)[1]})
    write_csv(os.path.join(out, "C4_halluc_wm_vs_oracle.csv"), rows)
    if rows:
        p = os.path.join(out, "C4_halluc_wm_vs_oracle.png")
        bar_chart(p, groups, bars, vals, "fraction of 16-step chunks hallucinated",
                  "Does the WM planner pick plans it hallucinates on? (WM's rollout of the executed plan: "
                  "WM-chosen solid, oracle-chosen hatched)",
                  ylim=(0, 1.02))
        made.append(p)

    # C5: per episode, the summed per-replan regret_K vs the return actually
    # lost to the oracle on the same seed, both in score units (paired seeds)
    vals, rows, bars, draws = {}, [], [], defaultdict(dict)
    for c in planners:
        o = oracle_of[c]
        labs = (f"{c}: summed per-replan regret", f"{c}: episode loss to its oracle")
        bars += [(labs[0], color[c], ""), (labs[1], color[c], "///")]
        for t in tasks:
            if t not in rets[floor] or t not in rets[ceil] or t not in rets[o] or t not in rets[c]:
                continue
            f_m, h_m = anchor_means(t)
            reg = defaultdict(list)
            for r in replans:
                if r["cond"] == c and r["task"] == t:
                    reg[int(r["env_seed"])].append(r.get("regret_K", np.nan))
            seeds = [s for s in sorted(reg) if s in rets[o][t] and s in rets[c][t]]
            if not seeds:
                continue
            x = np.array([np.nansum(reg[s]) for s in seeds]) / (h_m - f_m)
            y = np.array([rets[o][t][s] - rets[c][t][s] for s in seeds]) / (h_m - f_m)
            i = rng.integers(0, len(seeds), (N_BOOT, len(seeds)))
            for lab, v in zip(labs, (x, y)):
                draws[lab][t] = (float(v.mean()), v[i].mean(1))
                vals[(t, lab)] = (float(v.mean()), *ci(v[i].mean(1)))
            rho = spearmanr(x, y)[0] if len(seeds) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan
            rows.append({"group": t, "condition": c, "oracle": o, "n_seeds": len(seeds),
                         "regret_mean": vals[(t, labs[0])][0], "regret_ci_lo": vals[(t, labs[0])][1],
                         "regret_ci_hi": vals[(t, labs[0])][2], "loss_mean": vals[(t, labs[1])][0],
                         "loss_ci_lo": vals[(t, labs[1])][1], "loss_ci_hi": vals[(t, labs[1])][2],
                         "loss_over_regret": y.mean() / x.mean() if x.mean() else np.nan,
                         "spearman_per_seed": rho})
            rows += [{"group": t, "condition": c, "oracle": o, "env_seed": s, "regret": a, "loss": b,
                      "n_replans": len(reg[s]), "n_regret_nan": int(np.isnan(reg[s]).sum())}
                     for s, a, b in zip(seeds, x, y)]
    for lab, per in draws.items():
        vals[(AVG, lab)] = (float(np.mean([m for m, _ in per.values()])),
                            *ci(np.nanmean([d for _, d in per.values()], axis=0)))
    write_csv(os.path.join(out, "C5_regret_vs_episode_loss.csv"), rows)
    if rows:
        p = os.path.join(out, "C5_regret_vs_episode_loss.png")
        bar_chart(p, groups, bars, vals, "per episode, in score units\n(1 = oracle_reward - random)",
                  "Does per-replan regret account for the return actually lost? "
                  "(solid = summed regret, hatched = actual loss)")
        made.append(p)

    # C6: exploitation from the same state. Per replan of the WM runs: does the
    # WM hallucinate on the plan it chose (a_wm) more than on the plan the sim
    # planner chose from that very state (a_sim)? Paired, bootstrap over episodes.
    vals, rows, bars = {}, [], []
    for r in replans:
        r["_halluc_diff"] = r.get("halluc_wm", np.nan) - r.get("halluc_sim", np.nan)
    sub = [r for r in replans if r.get("exec") == "wm" and np.isfinite(r["_halluc_diff"])]
    for c in planners:
        labs = (f"{c}: WM-chosen plan", f"{c}: sim-chosen plan (same state)")
        bars += [(labs[0], color[c], ""), (labs[1], color[c], "///")]
        grp = by_task_episode(sub, c)
        res = {k: boot(grp, mean_of(k), tasks) for k in ("halluc_wm", "halluc_sim", "_halluc_diff")}
        for g in res["halluc_wm"]:
            vals[(g, labs[0])] = res["halluc_wm"][g]
            vals[(g, labs[1])] = res["halluc_sim"][g]
            d = res["_halluc_diff"][g]
            rows.append({"group": g, "condition": c, "p_halluc_wm": vals[(g, labs[0])][0],
                         "p_halluc_sim": vals[(g, labs[1])][0], "diff": d[0], "diff_ci_lo": d[1], "diff_ci_hi": d[2],
                         "n_replans": sum(1 for r in sub if r["cond"] == c and (g == AVG or r["task"] == g))})
    write_csv(os.path.join(out, "C6_exploitation_same_state.csv"), rows)
    if rows:
        p = os.path.join(out, "C6_exploitation_same_state.png")
        bar_chart(p, groups, bars, vals, "fraction of replans hallucinated",
                  "Does the planner seek hallucinations? From the same state, WM rollout of the plan the WM "
                  "chose (solid) vs of the plan the sim planner chose (hatched)", ylim=(0, 1.02))
        made.append(p)
    return made


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--score_manifest", default="",
                   help="extra conditions used only for track C (e.g. the random floor and "
                        "the exec=sim oracles); the main manifest's conditions are scored too")
    p.add_argument("--floor", default="random", help="track C: condition that scores 0")
    p.add_argument("--ceil", default="oracle_reward", help="track C: condition that scores 1")
    p.add_argument("--oracle_of", default="reward=oracle_reward,value=oracle_value",
                   help="track C: WM planner condition = oracle with the same objective")
    p.add_argument("--baseline", default="bc", help="track C (C2): the no-search baseline")
    a = p.parse_args()
    conds = read_manifest(a.manifest)
    os.makedirs(a.out, exist_ok=True)
    replans, chunks = load(conds)
    names = [c for c, _, _ in conds]
    tasks = sorted({r["task"] for r in chunks} | {r["task"] for r in replans})
    for c, arm, runs in conds:
        print(f"[{c}] arm={arm} {len(runs)} runs: {sum(r['cond'] == c for r in chunks)} chunks, "
              f"{sum(r['cond'] == c for r in replans)} compare replans")
    made = track_a(replans, names, tasks, a.out) + track_b(chunks, names, tasks, a.out)
    score_conds = read_manifest(a.score_manifest) if a.score_manifest else []
    extra = [c for c in score_conds if c[0] not in names]
    all_conds = conds + extra
    rets = load_returns(all_conds)
    if a.floor in rets and a.ceil in rets:
        chunks_x = load(extra)[1]   # their hallucination labels (C3, C4); kept out of A/B
        oracle_of = dict(kv.split("=") for kv in a.oracle_of.split(",") if kv)
        made += track_c(rets, chunks + chunks_x, replans, [c for c, _, _ in all_conds], tasks,
                        a.floor, a.ceil, oracle_of, a.baseline, a.out)
    else:
        print(f"[track C] skipped: needs conditions {a.floor!r} and {a.ceil!r} (--score_manifest)")
    with open(os.path.join(a.out, "manifest_used.txt"), "w") as f:
        for c, arm, runs in all_conds:
            f.write(f"{c} {arm} {' '.join(runs)}\n")
    print(f"-> {a.out}")
    for m in made:
        print(f"   {m}")


if __name__ == "__main__":
    main()
