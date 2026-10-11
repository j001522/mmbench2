# predictor_cost_table.py
"""
Table figure of the predictors' deployment cost, from predictor_cost.py's cost.csv.
Per predictor: added time (and % overhead) when monitoring only the executed plan
(B = 1, % of one replan) and when scoring every CEM candidate (B = 64, % of one
batch rollout), plus peak GPU memory at B = 64.

Run from src/ (CPU):
    python predictor_cost_table.py --csv ../../logs/predictor_cost/27746570/cost.csv
Writes cost_table.png and cost_chart.png (the same numbers as bars) next to the csv.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROWS = [("motion", "motion", "latents already there"),
        ("u_f", "rollout+u_f", "sampler already\ncomputes x1_hat"),
        ("u_r", "u_r", "decode + re-encode\nevery frame"),
        ("u_s (N=2)", "rollout+u_s(N=2)", "N denoising seeds"),
        ("u_s (N=4)", "rollout+u_s(N=4)", "N denoising seeds"),
        ("u_s (N=8)", "rollout+u_s(N=8)", "N denoising seeds")]


def fmt_ms(ms):
    ms = max(ms, 0.0)          # negative = below run-to-run noise
    return f"+{ms / 1e3:.2f} s" if ms >= 1e3 else (f"+{ms:.0f} ms" if ms >= 1 else "< 1 ms")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    a = p.parse_args()
    with open(a.csv) as f:
        r = {(x["mode"], x["condition"]): x for x in csv.DictReader(f)}
    g = lambda m, c, k: float(r[(m, c)][k])  # noqa: E731

    text = [["(reference)", f"{g('monitor', 'rollout', 'ms') / 1e3:.2f} s / rollout",
             f"{g('batch', 'rollout', 'ms') / 1e3:.2f} s / rollout", f"{g('batch', 'rollout', 'peak_mem_gb'):.1f} GB"]]
    for _, c, why in ROWS:
        text.append([why] + [f"{fmt_ms(g(m, c, 'extra_ms'))}  ({max(g(m, c, 'overhead_pct'), 0):.1f}%)"
                             for m in ("monitor", "batch")] + [f"{max(g('batch', c, 'peak_mem_gb'), g('batch', 'rollout', 'peak_mem_gb')):.1f} GB"])
    head = ["what it adds", "monitor executed plan (B=1)\n+ time (% of a replan)",
            "score all CEM candidates (B=64)\n+ time (% of a rollout)", "peak GPU mem, rollout\n+ predictor (B=64)"]
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.axis("off")
    tb = ax.table(cellText=text, rowLabels=["rollout only"] + [n for n, _, _ in ROWS], colLabels=head,
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 2.0)
    tb.auto_set_column_width(list(range(len(head))))
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#c9c8c4")
        if i == 0:
            c.set_height(c.get_height() * 1.5)
            c.set_text_props(weight="bold")
        if i == 1:
            c.set_facecolor("#f1f0ec")
    ax.set_title("Deployment cost of the predictors on the planner's rollout (H100; H=32, 4 denoising steps, "
                 "latent-only rollout;\nreplan = 3 CEM iterations x 64 rollouts; median of 5 repeats;\nB=1 differences of tens of ms are within run-to-run noise)", fontsize=10, pad=24)
    out = os.path.join(os.path.dirname(a.csv), "cost_table.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"-> {out}")
    plt.close(fig)
    cost_chart(g, os.path.join(os.path.dirname(a.csv), "cost_chart.png"))


def cost_chart(g, out):
    """Same numbers as bars, one panel per measure: B=1 added time, B=64 time per
    rollout (rollout + what the predictor adds), B=64 peak GPU memory."""
    base, add, ink2 = "#c9c8c4", "#2a78d6", "#52514e"
    names = [n for n, _, _ in ROWS][::-1]                      # top-to-bottom = table order
    conds = [c for _, c, _ in ROWS][::-1]
    y = range(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)

    ax = axes[0]
    ms = [max(g("monitor", c, "extra_ms"), 0.0) for c in conds]
    ax.barh(y, ms, color=add, height=0.6)
    for i, (c, v) in enumerate(zip(conds, ms)):
        ax.text(v + 2, i, f"{fmt_ms(v)} ({max(g('monitor', c, 'overhead_pct'), 0):.1f}%)", va="center",
                fontsize=8, color=ink2)
    ax.set_xlim(0, max(ms) * 1.9)
    ax.set_xlabel("added time per replan (ms)")
    ax.set_title(f"Monitor the executed plan (B=1)\nreplan = {g('monitor', 'rollout', 'ref_ms') / 1e3:.1f} s; "
                 "tens of ms = run-to-run noise", fontsize=9, loc="left")

    ax = axes[1]
    r0 = g("batch", "rollout", "ms") / 1e3
    ex = [max(g("batch", c, "extra_ms"), 0.0) / 1e3 for c in conds]
    ax.barh(y, [r0] * len(conds), color=base, height=0.6, label="rollout")
    ax.barh(y, ex, left=r0, color=add, height=0.6, label="added by the predictor")
    for i, (c, v) in enumerate(zip(conds, ex)):
        ax.text(r0 + v + 1, i, f"+{max(g('batch', c, 'overhead_pct'), 0):.0f}%", va="center", fontsize=8, color=ink2)
    ax.set_xlim(0, (r0 + max(ex)) * 1.15)
    ax.set_xlabel("time per batch rollout (s)")
    ax.set_title("Score all CEM candidates (B=64)", fontsize=9, loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    ax = axes[2]
    m0 = g("batch", "rollout", "peak_mem_gb")
    mem = [max(g("batch", c, "peak_mem_gb"), m0) for c in conds]
    ax.barh(y, [m0] * len(conds), color=base, height=0.6)
    ax.barh(y, [m - m0 for m in mem], left=m0, color=add, height=0.6)
    for i, m in enumerate(mem):
        ax.text(m + 0.8, i, f"{m:.1f} GB", va="center", fontsize=8, color=ink2)
    ax.set_xlim(0, max(mem) * 1.18)
    ax.set_xlabel("peak GPU memory (GB)")
    ax.set_title("Peak GPU memory, rollout + predictor (B=64)", fontsize=9, loc="left")

    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(names)
    for ax in axes:
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Deployment cost of the predictors on the planner's rollout (H100; H=32, 4 denoising steps, "
                 "latent-only; median of 5 repeats)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
