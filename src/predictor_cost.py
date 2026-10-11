# predictor_cost.py
"""
Deployment cost of the paper's hallucination predictors (u_r, u_f, u_s, motion)
on top of the planner's imagination rollout, timed on the exact code the
closed-loop planner runs (plan_cem.imagine_batch: latent-only, KV cache, reward
features h, no decoding; plan_mpc_env settings: ctx 4, max_ctx 24, H 32,
eval_d 0.25 -> 4 denoising steps, tau_ctx 0.1).

What each predictor adds at deployment:
    motion  RMS(z_t - z_{t-1}) on latents the rollout already has
    u_f     RMS change of the denoiser's x1_hat between Euler steps; the
            rollout already computes x1_hat, so only the deltas are extra
    u_s     N-1 extra denoising runs per imagined step (batch x N)
    u_r     decode + re-encode of every imagined latent (the only predictor
            that needs pixels)
Note: interactive.sample_one_timestep_packed always computes the pooled
instability (plus one .item() sync per step), so the "rollout" row already
contains u_f's deltas; u_f's own cost is timed in isolation (u_f_ops) on
tensors of the same shape.

Two deployment modes:
    batch    scoring every CEM candidate: one rollout of B = cem_samples x
             cem_n_rollouts = 64 plans (a replan = cem_iters = 3 of these)
    monitor  scoring only the plan about to be executed: B = 1, once per replan
Latents are random (timing depends on shapes only). Median and IQR of --reps
CUDA-synchronized repeats after --warmup; peak GPU memory per condition.

Run from src/ (GPU):
    python predictor_cost.py --out ../../logs/predictor_cost
"""
import argparse
import csv
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from plan_cem import build_models, encode_frames_to_packed, imagine_batch, load_act_meta, load_lang_emb
from plan_mpc_env import _rms, roundtrip_residual

SEEDS = (2, 4, 8)


def bench(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1e3 * (time.perf_counter() - t0))
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    return {"ms": med, "ms_q1": q1, "ms_q3": q3, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="reacher-easy")
    p.add_argument("--tasks_json", default="../tasks.json")
    p.add_argument("--tokenizer_ckpt", default="./checkpoints/combined/tokenizer.pt")
    p.add_argument("--dynamics_ckpt", default="./checkpoints/combined/dynamics.pt")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--ctx_frames", type=int, default=4)
    p.add_argument("--max_ctx", type=int, default=24)
    p.add_argument("--eval_d", type=float, default=0.25)
    p.add_argument("--tau_ctx", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=64, help="CEM rollouts per call (cem_samples x cem_n_rollouts)")
    p.add_argument("--cem_iters", type=int, default=3)
    p.add_argument("--rt_chunk", type=int, default=256, help="latents per decode/encode call for u_r")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    margs = SimpleNamespace(tokenizer_ckpt=a.tokenizer_ckpt, dynamics_ckpt=a.dynamics_ckpt, packing_factor=2,
                            schedule="shortcut", eval_d=a.eval_d)
    encoder, decoder, dyn, _rh, _ph, sched, info = build_models(margs, device)
    k_max = int(info["k_max"])
    _, act_mask = load_act_meta(a.tasks_json, a.task, device=device)
    lang = load_lang_emb(a.tasks_json, a.task, 512, device)
    H, W = int(info["H"]), int(info["W"])
    z1 = encode_frames_to_packed(encoder, torch.rand(1, 3, H, W, device=device), patch=int(info["patch"]),
                                 n_spatial=int(info["n_spatial"]), packing_factor=2, use_amp=True)
    S, D = z1.shape[1:]
    print(f"[setup] {torch.cuda.get_device_name()} | latent {S}x{D} | H={a.horizon} ctx={a.ctx_frames} "
          f"denoise steps={sched['K']} | batch={a.batch}", flush=True)

    rows = []

    def record(mode, B, name, what, r):
        rows.append({"mode": mode, "B": B, "condition": name, "what": what, **r})
        print(f"[{mode:7s} B={B:3d}] {name:22s} {r['ms']:9.1f} ms  (IQR {r['ms_q1']:.1f}-{r['ms_q3']:.1f})  "
              f"peak {r['peak_mem_gb']:.1f} GB", flush=True)

    for mode, B in (("monitor", 1), ("batch", a.batch)):
        z_ctx = z1.unsqueeze(1).expand(B, a.ctx_frames, S, D).contiguous() + 0.1 * torch.randn(
            B, a.ctx_frames, S, D, device=device)
        a_ctx = (torch.rand(B, a.ctx_frames, 16, device=device) * 2 - 1) * act_mask
        plan = (torch.rand(B, a.horizon, 16, device=device) * 2 - 1) * act_mask

        def roll(return_unc=False, seeds=1):
            return imagine_batch(dyn, z_start=z_ctx, action_plan=plan, horizon=a.horizon, sched=sched,
                                 k_max=k_max, act_mask_1d=act_mask, lang_emb=lang, use_amp=True, return_h=True,
                                 past_actions=a_ctx, max_ctx=a.max_ctx, tau_ctx=a.tau_ctx,
                                 return_unc=return_unc, unc_seeds=seeds)

        base = bench(roll, a.warmup, a.reps)
        record(mode, B, "rollout", "imagine_batch, latent only (= deployment planner)", base)
        z_seq = roll()[0]                                       # (B, 1+H, S, D)
        z_flat = z_seq[:, 1:].reshape(-1, S, D).contiguous()

        record(mode, B, "rollout+u_f", "return_unc, 1 seed (per-sample instability)",
               bench(lambda: roll(True, 1), a.warmup, a.reps))
        # u_f alone: the x1_hat deltas of the tail Euler steps, H steps, same shapes
        K = int(sched["K"])
        xs = [torch.randn(B, 1, S, D, device=device, dtype=torch.bfloat16) for _ in range(K)]

        def uf_ops():
            for _ in range(a.horizon):
                prev, d = None, []
                for x in xs:
                    xf = x.float()
                    if prev is not None:
                        d.append((xf - prev).pow(2).flatten(1).mean(1).sqrt())
                    prev = xf
                torch.stack(d)[len(d) // 2:].mean(0)
        record(mode, B, "u_f_ops", f"x1_hat deltas only ({K} Euler steps x H)", bench(uf_ops, a.warmup, a.reps))
        for n in SEEDS:
            record(mode, B, f"rollout+u_s(N={n})", f"return_unc, {n} denoising seeds",
                   bench(lambda: roll(True, n), a.warmup, a.reps))

        def u_r():
            for c0 in range(0, z_flat.shape[0], a.rt_chunk):
                roundtrip_residual(encoder, decoder, z_flat[c0:c0 + a.rt_chunk], info=info, use_amp=True)
        record(mode, B, "u_r", f"decode + encode of B x H = {z_flat.shape[0]} latents", bench(u_r, a.warmup, a.reps))

        def dec_only():
            from interactive import unpack_spatial_to_bottleneck
            for c0 in range(0, z_flat.shape[0], a.rt_chunk):
                z = unpack_spatial_to_bottleneck(z_flat[c0:c0 + a.rt_chunk].float().unsqueeze(1), k=2,
                                                 d_bottleneck=int(info["d_bottleneck"]))
                with torch.inference_mode(), torch.autocast("cuda"):
                    decoder(z)
        record(mode, B, "decode_only", "decoder alone on the same latents (reference)", bench(dec_only, a.warmup, a.reps))
        record(mode, B, "motion", "RMS of consecutive latents",
               bench(lambda: _rms((z_seq[:, 1:] - z_seq[:, :-1]).reshape(-1, S, D)), a.warmup, a.reps))

    # overhead: batch mode relative to its rollout call; monitor mode relative to one
    # replan (cem_iters batch rollouts), since it runs once per replan
    ref = {"batch": next(r["ms"] for r in rows if r["mode"] == "batch" and r["condition"] == "rollout")}
    ref["monitor"] = a.cem_iters * ref["batch"]
    for r in rows:
        base = next(x["ms"] for x in rows if x["mode"] == r["mode"] and x["condition"] == "rollout")
        extra = r["ms"] - base if r["condition"].startswith("rollout+") else r["ms"]
        if r["condition"] == "rollout":
            extra = 0.0
        r["extra_ms"] = extra
        r["ref"] = "one batch rollout" if r["mode"] == "batch" else f"one replan ({a.cem_iters} batch rollouts)"
        r["ref_ms"] = ref[r["mode"]]
        r["overhead_pct"] = 100 * extra / ref[r["mode"]]
    with open(os.path.join(a.out, "cost.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("\n[overhead]")
    for r in rows:
        print(f"  {r['mode']:7s} {r['condition']:22s} +{r['extra_ms']:8.1f} ms = {r['overhead_pct']:6.1f}% of {r['ref']}")
    print(f"-> {os.path.join(a.out, 'cost.csv')}")


if __name__ == "__main__":
    main()
