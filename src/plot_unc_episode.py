"""
Plot the hallucination predictors along ONE executed episode, against env step.

Each replan logs u_r/u_f/u_s/motion over its whole imagined horizon, but only
the first exec_K steps of that plan are executed. This stitches those executed
prefixes together, so every env step t shows the signals the model assigned to
the action actually taken at t, next to the realized latent error e_z and the
real vs predicted reward. Dotted verticals mark replans: the imagined-step
counter restarts there, so jumps at those lines are expected.

plan_mpc_env.py --log_unc writes this as <episode>_unc.png next to each episode
JSON. To (re)plot saved episodes by hand:

    python plot_unc_episode.py ../../logs/plan/<run>/cem_viz/ep00*_seed*.json
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLORS = {
    "u_r": "#2a78d6", "u_f": "#eb6834", "u_s": "#1baf7a", "motion": INK_2,
    "motion_real": "#eda100", "e_z": "#4a3aa7", "reward": "#008300", "r_pred": "#e87ba4",
}
DASHED = (0, (4, 2))
# (title, [(key, label, linestyle)], y-range from 0?)
PANELS = [
    ("u_r  round-trip residual", [("u_r", "u_r", "-")], False),
    ("u_f  flow instability", [("u_f", "u_f", "-")], False),
    ("u_s  inter-seed variance", [("u_s", "u_s", "-")], False),
    ("Motion  (predicted vs real)",
     [("motion", "predicted m", DASHED), ("motion_real", "real", "-")], True),
    ("Realized latent error  e_z  (executed steps)", [("e_z", "e_z", "-")], True),
    # The gap between the two lines is the reward error e_r.
    ("Reward  (env vs reward head)",
     [("reward", "env", "-"), ("r_pred", "predicted", DASHED)], True),
]
KEYS = ["u_r", "u_f", "u_s", "motion", "motion_real", "e_z", "r_pred"]


def stitch(ep: dict):
    """Concatenate the executed prefix of each replan onto the env-step axis."""
    reps = [r for r in ep.get("replans", []) if "unc" in r]
    if not reps:
        raise ValueError("no replans with an 'unc' block -- was it run with --log_unc?")
    out = {k: [] for k in KEYS}
    bounds = []
    t = 0
    for r in reps:
        u = r["unc"]
        # e_z has exactly one entry per executed step (shorter on the final chunk).
        n = len(u.get("e_z", [])) or int(r["exec_K"])
        bounds.append(t)
        for k in KEYS:
            v = np.asarray(u.get(k, [np.nan] * n)[:n], dtype=np.float64)
            if v.size < n:
                v = np.concatenate([v, np.full(n - v.size, np.nan)])
            out[k].append(v)
        t += n
    s = {k: np.concatenate(v) for k, v in out.items()}
    s["reward"] = np.asarray(ep["rewards"][:t], dtype=np.float64)
    return s, bounds, t


def plot_episode(ep: dict, out: str) -> None:
    s, bounds, T = stitch(ep)
    x = np.arange(1, T + 1)

    # x ticks: minor tick at every replan boundary, labelled major ticks every
    # few replans (<= ~20 labels), so a step read off a plot can be found in
    # the video's "step i/T" counter.
    K = int(ep["replans"][0].get("exec_K") or 1) if ep.get("replans") else 1
    n_rep = max(1, len(bounds))
    major = K * int(np.ceil(n_rep / 20))
    minor = K if major > K else max(1, K // 4)           # short episodes: finer minor ticks

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9, 2.5 * len(PANELS)), sharex=True,
                             facecolor=SURFACE)
    for ax, (title, series, from_zero) in zip(axes, PANELS):
        for key, label, ls in series:
            v = s[key]
            if np.all(np.isnan(v)):
                continue
            ax.plot(x, v, color=COLORS[key], linewidth=1.6, linestyle=ls, label=label)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.xaxis.set_minor_locator(MultipleLocator(minor))
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.grid(axis="x", which="major", color=GRID, linewidth=0.6)
        ax.set_xlim(0, T + 1)
        if from_zero:
            ax.set_ylim(bottom=0)
        # Predictors keep a data-fit range, so relative changes stay visible.
        for b in bounds[1:]:
            ax.axvline(b + 0.5, color=INK_2, linewidth=0.8, linestyle=(0, (1, 3)))
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
    axes[-1].set_xlabel(f"env step (executed)  —  dotted lines = replans every {K} steps,"
                        f" minor ticks every {minor}", color=INK_2)

    head = (f"{ep.get('task')}  {ep.get('policy')}  ep {ep.get('episode')} "
            f"(seed {ep.get('env_seed')})   return {ep.get('return', float('nan')):.1f}"
            f"   success {ep.get('success', float('nan')):.0f}")
    fig.suptitle(head, x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episodes", nargs="+", help="episode JSONs (ep*.json)")
    a = p.parse_args()
    for path in a.episodes:
        with open(path) as f:
            ep = json.load(f)
        out = os.path.splitext(path)[0] + "_unc.png"
        plot_episode(ep, out)
        print(f"[plot] {out}")


if __name__ == "__main__":
    main()
