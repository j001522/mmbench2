# plan_halluc.py
"""
Hallucination occurrence in closed-loop planning runs (plan_mpc_env.py), BC vs
CEM, scored with the Hausdorff mask metric and the per-task thresholds fitted
on the manual annotations (logs/annotation/roc_report/summary.csv).

Unit = one replan chunk: the K executed env steps of a replan, each real frame
against the world model's prediction of it (a fresh rollout of the executed
plan from the real context; the "WM" half of the compare video). Score = mean
over the chunk of the per-frame Hausdorff (px) between the foreground masks,
the same per-sequence summary the thresholds were fitted on (16 imagined
frames there, K = 16 here).

Per-frame Hausdorff source, per episode:
    json  replans[i].halluc.hausdorff, computed inline by plan_mpc_env.py on
          the raw frames (runs after 2026-10-03)
    mp4   recomputed from the compare video (older runs). Frames come back
          within ~0.4 dB PSNR of the raw ones, except the burned-in "step i/T"
          label (top-left of the real half). The label (black box, white text)
          adds no foreground, so it is left in: clearing its box in both masks
          (--stamp_box) can empty a mask when the cup-catch ball passes under
          it. Roundtrip of the 298 annotated sequences through this video
          format (logs/plan_halluc/roundtrip_check.py), AUROC raw -> mp4:
          reacher 0.938 -> 0.936, cup 0.962 -> 0.970, mw 0.901 -> 0.891.

Uncertainty predictors (only in --log_unc runs), over the executed steps of
the chunk: u_r, u_f, u_s, motion raw, and u_*_norm = u_* / max(motion, 1e-3)
per step (as paper_predictors.py), then the mean over the chunk.

Run from src/ (CPU):
    python plan_halluc.py --runs ../../logs/plan/cup-catch_27292399 ... \
        --out ../../logs/plan_halluc/x
Writes chunks.csv (one row per chunk) and summary.txt.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from dmc_background_mask import foreground_mask, preset_for_task  # noqa: E402

# Hausdorff (mask), per-sequence mean, ">= thr" = hallucination. Per-task
# balanced-accuracy thresholds, roc_report/summary.csv (2026-10-02).
THRESH = {"reacher-easy": 4.6960, "cup-catch": 2.6934, "mw-reach": 5.0050}
UNC_KEYS = ("u_r", "u_f", "u_s")
FRAME = 224
GAP = 4


# ------------------------------------------------------------------ metric

def hausdorff(real, wm, task, ignore=None):
    """Hausdorff distance (px) between the foreground masks of two (H,W,3)
    uint8 frames, as mask_metrics.shape_metrics. ignore: (y0, y1, x0, x1) box
    cleared in both masks."""
    pr = preset_for_task(task)
    mr, mx = foreground_mask(real, pr), foreground_mask(wm, pr)
    if ignore is not None:
        y0, y1, x0, x1 = ignore
        mr[y0:y1, x0:x1] = False
        mx[y0:y1, x0:x1] = False
    if not mr.any() and not mx.any():
        return 0.0
    cap = float(np.hypot(*mr.shape))
    dt_r = np.minimum(ndimage.distance_transform_edt(~mr), cap) if mr.any() else np.full(mr.shape, cap)
    dt_x = np.minimum(ndimage.distance_transform_edt(~mx), cap) if mx.any() else np.full(mx.shape, cap)
    a, b = dt_r[mx], dt_x[mr]
    return max(float(a.max()) if a.size else 0.0, float(b.max()) if b.size else 0.0)


def stamp_box(n_steps, comp_width, scale, pad=2):
    """Box (in FRAME px) that plan_mpc_env.stamp_steps covers on the real half
    of every video frame. stamp_steps runs on the final (possibly upscaled)
    video frame, so redo its layout at that width and map back."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=max(10, comp_width // 32))
    except TypeError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(Image.new("RGB", (comp_width, 64)))
    text = f"step {n_steps:>{len(str(n_steps))}d}/{n_steps}"
    x0, y0, x1, y1 = draw.textbbox((4, 3), text, font=font)
    x1, y1 = x1 + 3 + pad * scale, y1 + 2 + pad * scale
    return (0, int(np.ceil(y1 / scale)), 0, int(np.ceil(x1 / scale)))


def load_video(mp4):
    """(real, wm, scale, comp_width): uint8 (N, FRAME, FRAME, 3) each. Video
    frame i = after env step i; frame 0 holds the tokenizer round trip."""
    import imageio.v2 as imageio
    reader = imageio.get_reader(mp4)
    real, wm, scale, wt = [], [], None, None
    for f in reader:
        if scale is None:
            wt = f.shape[1]
            scale = next((c for c in (1, 2, 3, 4) if wt == (2 * FRAME + GAP) * c), None)
            if scale is None:
                raise ValueError(f"{mp4}: width {wt} is not a compare video")
        def down(x):
            if scale == 1:
                return x
            return x.reshape(FRAME, scale, FRAME, scale, 3).mean(axis=(1, 3)).round().astype(np.uint8)
        F = FRAME * scale
        real.append(down(f[:F, :F, :3]))
        wm.append(down(f[:F, F + GAP * scale:2 * F + GAP * scale, :3]))
    reader.close()
    return np.stack(real), np.stack(wm), scale, wt


# ------------------------------------------------------------------ episodes

def run_info(run_dir):
    """Header line of plan.log -> dict, plus value/policy ckpt lines."""
    info = {}
    p = os.path.join(run_dir, "plan.log")
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            if line.startswith("task=") and not info:
                for tok in line.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        info[k] = v.strip("'")
            elif line.startswith(("value_ckpt=", "policy_ckpt=")):
                info[line.split("=", 1)[0]] = line.split("=", 1)[1].split()[0]
    return info


def episode_rows(json_path, run, arm, task, source="auto", use_box=False):
    ep = json.load(open(json_path))
    replans = ep.get("replans", [])
    if not replans:
        return []
    have_json = all("hausdorff" in (r.get("halluc") or {}) for r in replans)
    use = "json" if (source in ("auto", "json") and have_json) else "mp4"
    hd_steps = None
    if use == "mp4":
        mp4 = os.path.splitext(json_path)[0] + ".mp4"
        if not os.path.exists(mp4):
            return []
        real, wm, scale, wt = load_video(mp4)
        n = len(real) - 1
        box = stamp_box(n, wt, scale) if use_box else None
        hd_steps = [hausdorff(real[t], wm[t], task, ignore=box) for t in range(1, len(real))]
    rows, t0 = [], 0
    ep_return = float(ep["return"])
    for i, r in enumerate(replans):
        k = int(r["exec_K"])
        n_exec = min(k, int(r["step"]) - t0)
        if use == "json":
            hd = np.asarray(r["halluc"]["hausdorff"][:n_exec], float)
        else:
            hd = np.asarray(hd_steps[t0:t0 + n_exec], float)
        row = {
            "task": task, "run": run, "arm": arm, "cost": ep.get("cost_kind"),
            "bc_head": ep.get("bc_head"), "episode": ep["episode"], "env_seed": ep["env_seed"],
            "tau_ctx": ep.get("tau_ctx", 0.0), "eval_d": ep.get("eval_d", 0.125),
            "ep_return": ep_return, "chunk": i, "start": t0, "n_exec": n_exec,
            "n_ctx": min(int(ep.get("ctx_frames", 4)), t0 + 1),
            "reward_chunk": r.get("reward_executed"), "source": use,
            "hausdorff_mean": float(np.mean(hd)) if hd.size else np.nan,
            "hausdorff_max": float(np.max(hd)) if hd.size else np.nan,
        }
        thr = THRESH.get(task)
        row["halluc"] = (int(row["hausdorff_mean"] >= thr) if thr is not None and np.isfinite(row["hausdorff_mean"])
                         else "")
        h = r.get("halluc") or {}
        if h.get("mse_wm"):
            d = 10 * np.log10(max(sum(h["mse_floor"]), 1e-12) / max(sum(h["mse_wm"]), 1e-12))
            row["paper_dpsnr"] = float(d)
        u = r.get("unc")
        if u:
            mot = np.asarray(u["motion"][:n_exec], float)
            row["motion"] = float(np.mean(mot))
            row["motion_real"] = float(np.mean(u["motion_real"][:n_exec])) if u.get("motion_real") else np.nan
            row["e_z"] = float(np.mean(u["e_z"][:n_exec])) if u.get("e_z") else np.nan
            for key in UNC_KEYS:
                if u.get(key) is None:
                    continue
                v = np.asarray(u[key][:n_exec], float)
                row[key] = float(np.nanmean(v))
                row[key + "_norm"] = float(np.nanmean(v / np.maximum(mot, 1e-3)))
        rows.append(row)
        t0 = int(r["step"])
    return rows


# ------------------------------------------------------------------ summary

def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    rk = rankdata(s)
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_rate(rows, n_boot=2000, seed=0):
    """Chunk-level hallucination rate with a 95% CI from a bootstrap over
    episodes (chunks of one episode are correlated)."""
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[(r["run"], r["episode"])].append(r["halluc"])
    eps = list(by_ep.values())
    rate = float(np.mean([v for e in eps for v in e]))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = [eps[i] for i in rng.integers(0, len(eps), len(eps))]
        bs.append(np.mean([v for e in pick for v in e]))
    return rate, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(eps)


def summarize(rows):
    out = []
    P = lambda s="": out.append(s)
    lab = [r for r in rows if r["halluc"] != ""]
    groups = defaultdict(list)
    for r in lab:
        groups[(r["task"], r["arm"], r["cost"], f'{r["bc_head"]} t{r["tau_ctx"]} d{r["eval_d"]}')].append(r)
    P("Hallucination rate per chunk (Hausdorff mean >= per-task threshold), 95% CI bootstrap over episodes")
    P(f"{'task':13s} {'arm':4s} {'cost':16s} {'bc_head':20s} {'eps':>4s} {'chunks':>6s} "
      f"{'rate':>6s} {'CI':>15s} {'HD mean':>8s} {'ep halluc>0':>11s} {'return':>7s}")
    for key in sorted(groups):
        g = groups[key]
        rate, lo, hi, n_ep = boot_rate(g)
        by_ep = defaultdict(list)
        for r in g:
            by_ep[(r["run"], r["episode"])].append(r)
        ep_any = np.mean([any(c["halluc"] for c in e) for e in by_ep.values()])
        ret = np.mean([e[0]["ep_return"] for e in by_ep.values()])
        P(f"{key[0]:13s} {key[1]:4s} {str(key[2]):16s} {str(key[3]):20s} {n_ep:4d} {len(g):6d} "
          f"{rate:6.3f} [{lo:.3f}, {hi:.3f}] {np.mean([r['hausdorff_mean'] for r in g]):8.2f} "
          f"{ep_any:11.2f} {ret:7.1f}")

    # motion strata (tertiles of the real latent motion, per task), where logged
    P()
    P("Rate by real-motion tertile (per task; chunks with --log_unc only)")
    for task in sorted({r["task"] for r in lab}):
        tr = [r for r in lab if r["task"] == task and np.isfinite(r.get("motion_real", np.nan))]
        if len(tr) < 9:
            continue
        q = np.quantile([r["motion_real"] for r in tr], [1 / 3, 2 / 3])
        for key in sorted({(r["arm"], r["cost"], r["bc_head"]) for r in tr}):
            g = [r for r in tr if (r["arm"], r["cost"], r["bc_head"]) == key]
            cells = []
            for b in range(3):
                lo_ = -np.inf if b == 0 else q[b - 1]
                hi_ = np.inf if b == 2 else q[b]
                c = [r["halluc"] for r in g if lo_ <= r["motion_real"] < hi_]
                cells.append(f"{np.mean(c):.2f} (n={len(c)})" if c else "  -  ")
            P(f"  {task:13s} {key[0]:4s} {str(key[1]):16s} {str(key[2]):20s} low {cells[0]:13s} "
              f"mid {cells[1]:13s} high {cells[2]}")

    P()
    P("Closed-loop AUROC of the predictors vs the Hausdorff label (chunks with --log_unc, arms pooled)")
    keys = [k + s for k in UNC_KEYS for s in ("", "_norm")] + ["motion", "e_z", "paper_dpsnr"]
    P(f"  {'task':13s} {'n':>4s} {'pos':>4s} " + " ".join(f"{k:>11s}" for k in keys))
    for task in sorted({r["task"] for r in lab}):
        tr = [r for r in lab if r["task"] == task and "u_r" in r]
        if not tr:
            continue
        y = [r["halluc"] for r in tr]
        vals = []
        for k in keys:
            s = [(-r[k] if k == "paper_dpsnr" else r[k]) if k in r else np.nan for r in tr]
            vals.append(auroc(s, y))
        P(f"  {task:13s} {len(tr):4d} {sum(y):4d} " + " ".join(f"{v:11.3f}" for v in vals))
    return "\n".join(out)


def _score(job):
    jp, run, arm, discount, source, use_box = job
    task = json.load(open(jp))["task"]
    rows = episode_rows(jp, run, arm, task, source, use_box)
    for r in rows:
        r["discount"] = discount
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--runs", nargs="+", required=True, help="logs/plan/<task>_<job> dirs")
    p.add_argument("--arms", nargs="+", default=["bc", "cem"])
    p.add_argument("--source", default="auto", choices=["auto", "json", "mp4"])
    p.add_argument("--stamp_box", action="store_true",
                   help="mp4 source: clear the step-label box in both masks (off: see docstring)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = []
    for run_dir in a.runs:
        run = os.path.basename(os.path.normpath(run_dir))
        info = run_info(run_dir)
        for arm in a.arms:
            for jp in sorted(glob.glob(os.path.join(run_dir, f"{arm}_viz", "ep*_seed*.json"))):
                jobs.append((jp, run, arm, info.get("discount", ""), a.source, a.stamp_box))
    rows = []
    with Pool(a.workers) as pool:
        for (jp, run, arm, *_), er in zip(jobs, pool.imap(_score, jobs)):
            rows += er
            if er:
                hr = [r["halluc"] for r in er if r["halluc"] != ""]
                print(f"[{run} {arm}] {os.path.basename(jp)}: {len(er)} chunks, src {er[0]['source']}, "
                      f"halluc {sum(hr)}/{len(hr)}", flush=True)
    if not rows:
        sys.exit("no episodes found")
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(os.path.join(a.out, "chunks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    s = summarize(rows)
    print(s)
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(s + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
