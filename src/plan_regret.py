# plan_regret.py
"""
WM planner vs true-model planner, per replan (plan_mpc_env.py --compare_sim).
At every replan both CEMs plan from the same real state with the same noise
draws: a_wm under the world model (reward head), a_sim in the simulator (true
reward). This script asks how the world model's hallucinations change the
planner's decisions.

Per replan (K = executed steps, H = horizon), from replans[i].compare:
    d_exec, d_first, w2   action distance a_wm vs a_sim (env action space)
    floor_sim / floor_wm  the same planner twice with independent noise: the
                          distance expected from CEM's own randomness
    d_avg, d_mu (+ _floor_sim)  distance of the K-step mean actions / of the
                          final elite means. d_exec compares two single CEM
                          samples, so it sits at the white-noise level
                          ~sqrt(2)*warm_start_init_std; these average it out
    regret_floor_K        R_sim(a_sim) - R_sim(a_sim2), a_sim2 = the sim floor
                          rerun: return spread of the true-model CEM itself
    regret_bc_K           R_sim(a_sim) - R_sim(a_bc), a_bc = the BC warm start
  --cost reward_value runs (both planners add the same value head; the sim side
  on h teacher-forced from the true frames):
    regret_J_K / _H       J_sim(a_sim) - J_sim(a_wm), J = R + gamma^t V(true
                          state) after t = K / H steps. regret_J_H is the main
                          regret for these runs (replaces regret_K below): J_H
                          is the objective the sim CEM maximizes, so J_K can be
                          negative (a_sim is not optimized for it, and V moves
                          by tens along a trajectory)
    regret_J_floor_H, regret_J_bc_H (and _K)   as above, on J
    v_optimism_K          V_wm(a_wm) - V_sim(a_wm) at K: value over-estimate
                          from the WM's imagined state vs the true one
    optimism_J_K          J_wm(a_wm) - J_sim(a_wm)
    regret_K / regret_H   R_sim(a_sim) - R_sim(a_wm): true return lost by
                          following the WM plan
    optimism_K            R_wm(a_wm) - R_sim(a_wm): the WM's over-prediction of
                          its own chosen plan
    wm_pref_K             R_wm(a_wm) - R_wm(a_sim): how much the WM prefers its
                          plan (truth: R_sim(a_wm) - R_sim(a_sim) = -regret_K)
    halluc_wm / _sim      Hausdorff-mean label (plan_halluc.THRESH) of the WM's
                          rollout of a_wm / a_sim against what the simulator
                          does with that plan. halluc_wm is the main label: the
                          rollout that made the planner pick a_wm.
    u_* (raw, _norm)      uncertainty predictors on the a_wm rollout (--log_unc)

Run from src/ (CPU):
    python plan_regret.py --runs ../../logs/plan/mw-reach_123 ... --out ../../logs/plan_regret/x
Writes replans.csv and summary.txt.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from plan_halluc import THRESH, auroc

UNC = ("u_r", "u_f", "u_s")


def replan_rows(json_path, run):
    ep = json.load(open(json_path))
    task = ep["task"]
    thr = THRESH.get(task)
    rows = []
    for i, r in enumerate(ep.get("replans", [])):
        c = r.get("compare")
        if not c:
            continue
        K = int(r["exec_K"])
        row = {"task": task, "run": run, "episode": ep["episode"], "env_seed": ep["env_seed"],
               "replan": i, "step": r["step"], "K": K, "exec": c.get("exec", "wm"),
               "ep_return": ep["return"]}
        for k in ("d_exec", "d_first", "d_H", "w2", "regret_K", "regret_H", "sim_seconds"):
            row[k] = c.get(k, np.nan)
        for k in ("d_avg", "d_mu"):
            row[k] = c.get(k, np.nan)
        for f in ("floor_sim", "floor_wm"):
            row[f] = c[f]["d_exec"] if f in c else np.nan
            for k in ("d_avg", "d_mu"):
                row[f"{k}_{f}"] = c[f].get(k, np.nan) if f in c else np.nan
        for k in ("Rsim_K_wm", "Rsim_K_sim", "Rsim_H_wm", "Rsim_H_sim",
                  "Rwm_K_wm", "Rwm_K_sim", "Rwm_H_wm", "Rwm_H_sim"):
            row[k] = c.get(k, np.nan)
        row["regret_floor_K"] = c["Rsim_K_sim"] - c["Rsim_K_sim2"] if "Rsim_K_sim2" in c else np.nan
        row["regret_bc_K"] = c["Rsim_K_sim"] - c["Rsim_K_bc"] if "Rsim_K_bc" in c else np.nan
        for k in ("regret_J_K", "regret_J_H"):
            row[k] = c.get(k, np.nan)
        g = lambda k: c.get(k, np.nan)
        for t in ("K", "H"):
            row[f"regret_J_floor_{t}"] = g(f"Jsim_{t}_sim") - g(f"Jsim_{t}_sim2")
            row[f"regret_J_bc_{t}"] = g(f"Jsim_{t}_sim") - g(f"Jsim_{t}_bc")
            row[f"v_optimism_{t}"] = g(f"Vwm_{t}_wm") - g(f"Vsim_{t}_wm")
            row[f"optimism_J_{t}"] = g(f"Jwm_{t}_wm") - g(f"Jsim_{t}_wm")
        row["Jsim_K_sim"], row["Jsim_K_wm"] = g("Jsim_K_sim"), g("Jsim_K_wm")
        row["optimism_K"] = row["Rwm_K_wm"] - row["Rsim_K_wm"]
        row["optimism_H"] = row["Rwm_H_wm"] - row["Rsim_H_wm"]
        row["wm_pref_K"] = row["Rwm_K_wm"] - row["Rwm_K_sim"]
        for k in ("wm", "sim"):
            hd = c.get("hausdorff_" + k) or []
            row[f"hd_{k}"] = float(np.mean(hd)) if hd else np.nan
            row[f"halluc_{k}"] = (int(row[f"hd_{k}"] >= thr) if thr is not None and np.isfinite(row[f"hd_{k}"])
                                  else np.nan)
        u = r.get("unc")
        if u and row["exec"] == "wm":                # unc is logged on the executed plan's rollout
            mot = np.asarray(u["motion"][:K], float)
            row["motion"] = float(mot.mean())
            for k in UNC:
                if u.get(k) is not None:
                    v = np.asarray(u[k][:K], float)
                    row[k] = float(np.nanmean(v))
                    row[k + "_norm"] = float(np.nanmean(v / np.maximum(mot, 1e-3)))
        rows.append(row)
    return rows


def boot_mean(rows, key, n_boot=2000, seed=0):
    """Mean of rows[key] with a 95% CI from a bootstrap over episodes."""
    by = defaultdict(list)
    for r in rows:
        v = r.get(key, np.nan)
        if np.isfinite(v):
            by[(r["run"], r["episode"])].append(v)
    eps = [v for v in by.values() if v]
    if not eps:
        return np.nan, np.nan, np.nan
    m = float(np.mean([x for e in eps for x in e]))
    rng = np.random.default_rng(seed)
    bs = [np.mean([x for i in rng.integers(0, len(eps), len(eps)) for x in eps[i]]) for _ in range(n_boot)]
    return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(t):
    m, lo, hi = t
    return f"{m:8.3f} [{lo:7.3f}, {hi:7.3f}]"


def summarize(rows):
    out = []
    P = out.append
    for task in sorted({r["task"] for r in rows}):
        T = [r for r in rows if r["task"] == task]
        n_ep = len({(r["run"], r["episode"]) for r in T})
        P(f"\n===== {task}: {len(T)} replans, {n_ep} episodes, exec={sorted({r['exec'] for r in T})} =====")
        P("Mean [95% CI, bootstrap over episodes]")
        for k, lab in (("d_exec", "action distance a_wm vs a_sim (K steps)"),
                       ("floor_sim", "  floor: sim CEM vs itself"),
                       ("floor_wm", "  floor: WM CEM vs itself"),
                       ("d_first", "first-action distance"),
                       ("w2", "W2 between final elite Gaussians"),
                       ("d_avg", "K-step mean-action distance"),
                       ("d_avg_floor_sim", "  floor: sim CEM vs itself"),
                       ("d_mu", "elite-mean distance"),
                       ("d_mu_floor_sim", "  floor: sim CEM vs itself"),
                       ("regret_K", "regret over K (true return lost)"),
                       ("regret_floor_K", "  floor: sim CEM vs itself"),
                       ("regret_bc_K", "  BC warm start (no search)"),
                       ("regret_H", "regret over H"),
                       ("regret_J_H", "regret on J = R + g^H V(true), H"),
                       ("regret_J_floor_H", "  floor: sim CEM vs itself"),
                       ("regret_J_bc_H", "  BC warm start (no search)"),
                       ("regret_J_K", "regret on J, K"),
                       ("regret_J_floor_K", "  floor: sim CEM vs itself"),
                       ("regret_J_bc_K", "  BC warm start (no search)"),
                       ("v_optimism_H", "WM value over-estimate on a_wm, H"),
                       ("v_optimism_K", "WM value over-estimate on a_wm, K"),
                       ("optimism_J_H", "WM optimism on J, a_wm, H"),
                       ("optimism_K", "WM optimism on a_wm over K"),
                       ("wm_pref_K", "WM preference a_wm over a_sim, K"),
                       ("hd_wm", "Hausdorff of the a_wm rollout"),
                       ("hd_sim", "Hausdorff of the a_sim rollout")):
            P(f"  {lab:40s} {fmt(boot_mean(T, k))}")
        both = [r for r in T if np.isfinite(r["d_exec"]) and np.isfinite(r["floor_sim"])]
        if both:
            P(f"  replans with d_exec > sim floor: {np.mean([r['d_exec'] > r['floor_sim'] for r in both]):.2f}"
              f"   with regret_K > 0: {np.mean([r['regret_K'] > 0 for r in T]):.2f}")
        trueK = np.array([r["Rsim_K_wm"] - r["Rsim_K_sim"] for r in T])
        prefK = np.array([r["wm_pref_K"] for r in T])
        ok = np.isfinite(trueK) & np.isfinite(prefK)
        if ok.sum() > 2:
            P(f"  WM ranks a_wm above a_sim in {np.mean(prefK[ok] > 0):.2f} of replans; truth agrees in "
              f"{np.mean(trueK[ok] > 0):.2f} (corr pref vs truth {np.corrcoef(prefK[ok], trueK[ok])[0, 1]:+.2f})")

        if task not in THRESH:
            continue
        # Main regret: on J when the run planned with the value head.
        REG = "regret_J_H" if any(np.isfinite(r["regret_J_H"]) for r in T) else "regret_K"
        OPT = "optimism_J_H" if REG == "regret_J_H" else "optimism_K"
        lab_rows = [r for r in T if np.isfinite(r["halluc_wm"])]
        P(f"\nBy the hallucination label of the a_wm rollout (thr {THRESH[task]} px)")
        for y in (0, 1):
            g = [r for r in lab_rows if r["halluc_wm"] == y]
            if not g:
                continue
            P(f"  halluc_wm={y}  n={len(g):4d}  {REG} {fmt(boot_mean(g, REG))}  "
              f"d_exec {boot_mean(g, 'd_exec')[0]:.3f}  {OPT} {boot_mean(g, OPT)[0]:8.3f}")
        # Does the planner pick plans the WM hallucinates on? Same state, the
        # WM's rollout of its own plan vs of the true-model plan.
        pr = [r for r in lab_rows if np.isfinite(r["halluc_sim"])]
        if pr:
            for r in pr:
                r["_dl"] = r["halluc_wm"] - r["halluc_sim"]
                r["_dhd"] = r["hd_wm"] - r["hd_sim"]
            P(f"\nExploitation (paired, same state): P(halluc | a_wm) {np.mean([r['halluc_wm'] for r in pr]):.3f} vs "
              f"P(halluc | a_sim) {np.mean([r['halluc_sim'] for r in pr]):.3f}")
            P(f"  label diff {fmt(boot_mean(pr, '_dl'))}   Hausdorff diff (px) {fmt(boot_mean(pr, '_dhd'))}")

        U = [r for r in lab_rows if "u_r" in r]
        if U:
            keys = [k + s for k in UNC for s in ("", "_norm")] + ["motion", "d_exec", OPT]
            big = np.nanpercentile([r[REG] for r in U], 67)
            P(f"\nAUROC on the a_wm rollout's chunk (n={len(U)}): target = halluc_wm | {REG} in top third "
              f"(> {big:.2f})")
            for k in keys:
                s = [r.get(k, np.nan) for r in U]
                a1 = auroc(s, [r["halluc_wm"] for r in U])
                a2 = auroc(s, [int(r[REG] > big) for r in U])
                P(f"  {k:12s} {a1:6.3f} | {a2:6.3f}")
            a = auroc([r["hd_wm"] for r in U], [int(r[REG] > big) for r in U])
            P(f"  {'hd_wm':12s}   -    | {a:6.3f}   (does the label itself flag costly decisions?)")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for run_dir in a.runs:
        run = os.path.basename(os.path.normpath(run_dir))
        for jp in sorted(glob.glob(os.path.join(run_dir, "cem_viz", "ep*_seed*.json"))):
            rows += replan_rows(jp, run)
    if not rows:
        raise SystemExit("no --compare_sim replans found")
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols and not k.startswith("_")]
    s = summarize(rows)
    with open(os.path.join(a.out, "replans.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(s)
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(s + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
