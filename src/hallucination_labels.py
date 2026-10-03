# hallucination_labels.py
"""
Per-sequence hallucination metrics and labels on held-out 24-frame sequences.
Two groups, kept separate throughout the code, CSV and summary:

  [1] PAPER metrics, exactly as defined in the MMBench2 paper ("Hallucination
      in World Models is Predictable and Preventable", Sec. 5 and App. E,
      Table 7) and its reference code. Do not change their definition.
  [2] OURS: variants that are NOT in the paper. Labels in this group carry a
      suffix (e.g. `_dec`) so they are never confused with the paper's.

========================= [1] PAPER METRICS =========================

  scene divergent : rollout dPSNR <= 0
      dPSNR = PSNR(open-loop rollout, GT) - PSNR(repeat-last-context-frame, GT)
      in pixel space, over the horizon only. ctx 8 + horizon 16 = 24 frames.
  action ignored  : action shuffle ratio <= 1.1
      ratio = one-step teacher-forced flow MSE with batch-shuffled actions
              / the same MSE with the true actions, in latent space.

Both metrics already exist in train_dynamics.py (run_dynamics_eval and
run_action_attribution_eval), but only as batch-pooled wandb scalars. This
script computes them PER SEQUENCE, with the same settings and conventions:
  - rollout: shortcut schedule d=0.25, tau_ctx=0.1   (--eval_d / --tau_ctx defaults)
  - shuffle: shortcut schedule d=1.0,  tau_ctx=0.1   (run_action_attribution_eval default)
  - shuffle pool = ALL loaded sequences (the paper's "pool-level" permutation,
    across tasks); per-(seq,t) noise and context corruption are shared between
    the true-action and shuffled-action passes, so the ratio differs only in
    the action tensor.
  - per-sequence ratio = mean_t MSE_shuf / mean_t MSE_real  (seq_ratios_b there)
  - the permutation is a derangement: randperm's fixed points (harmless in the
    pooled reference scalar) would give a per-sequence ratio of exactly 1
  - precision as in the reference: encoder/decoder and the shuffle-pass dynamics
    in fp32 (TF32), the rollout sampler in bf16 (sample_one_timestep_packed)
  - the decoder has causal time attention, so a decoded frame depends on the
    latents decoded before it in the same call. As in run_dynamics_eval, the
    whole [0, ctx+horizon) latent sequence (GT context + rollout) is decoded in
    one pass, never the horizon alone.

The paper's pool spans 200 tasks, so a shuffled action almost always comes
from a different embodiment, which is easy to detect and inflates the ratio.

====================== [2] OUR METRICS (not in paper) ======================

  scene divergent (decoded floor) : dPSNR_dec <= 0
      dPSNR_dec = PSNR(open-loop rollout, GT) - PSNR(repeat-DECODED-last-context-frame, GT)
      The paper's floor repeats the raw last context frame, which carries no
      tokenizer error while the rollout always does; nearly static windows are
      then labeled divergent almost by construction. Here the floor repeats the
      tokenizer round trip of that frame instead (frame ctx-1 of the same
      decode pass as the rollout), so both sides pay the reconstruction error
      and dPSNR_dec isolates the dynamics. Same window, threshold and rollout
      as [1]; only the floor frame differs.

--extra_horizon N (videos): the rollout and the one-step pass continue N frames
past the 24-frame label window with a sliding context of --max_ctx (24, the
training sequence length). Up to frame 23 every prediction sees exactly the
context it would see without it, and the labels are always computed on frames
[0, ctx+horizon) only, so they stay the paper's. Longer sequences do change
which windows are sampled (fewer valid starts near episode ends).

--viz_per_task N writes N label videos per task (<out>_viz/), taken from the
most dynamic third of the task's sequences (ground-truth frame motion), spread
over the label combinations.

Run from src/:
    python hallucination_labels.py --tasks reacher-easy mw-peg-insert-side \
        --out ../../logs/halluc/labels.csv
"""
import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from interactive import load_dynamics_from_ckpt, load_tokenizer_from_ckpt
from model import pack_bottleneck_to_spatial, temporal_patchify
from train_dynamics import (
    decode_packed_to_frames,
    make_tau_schedule,
    parse_batch,
    sample_one_timestep_packed,
)
from wm_dataset import WMDataset, collate_batch

# [1] paper thresholds
SCENE_DIVERGENT_DB = 0.0      # paper: dPSNR <= 0
ACTION_IGNORED_RATIO = 1.1    # paper: action shuffle ratio <= 1.1
# [2] ours
SCENE_DIVERGENT_DEC_DB = 0.0  # dPSNR_dec <= 0 (same threshold, decoded floor)


def select_indices(ds: WMDataset, n_per_task: int, seed: int):
    """Up to n_per_task random window indices per task, sorted, reproducible."""
    rng = np.random.default_rng(seed)
    out = []
    lo = 0
    for ti, hi in enumerate(ds._cum_counts):
        n = hi - lo
        k = min(n_per_task, n) if n_per_task > 0 else n
        pick = np.sort(rng.choice(n, size=k, replace=False)) + lo
        out.extend(int(i) for i in pick)
        lo = hi
    return out


def motion_indices(ds: WMDataset, n_per_task: int, pct: float):
    """Per task, the n_per_task windows that keep moving throughout: every
    valid window is scored by the pct-th percentile of its per-transition
    ground-truth motion (MSE between consecutive frames, 4x downsampled), so a
    single burst cannot carry a window that is otherwise static (e.g. goal
    already reached). Picked greedily by score with at most half-window overlap.
    Frames only, no model. For looking at the model, not for statistics."""
    T = ds.T
    out, lo = [], 0
    for ti, hi in enumerate(ds._cum_counts):
        n_frames = int(ds.seg_cum_frames[ti][-1])
        small = torch.cat([ds._get_frames(ti, s, min(2048, n_frames - s))[:, :, ::4, ::4]
                           for s in range(0, n_frames, 2048)])                 # uint8
        m = torch.empty(n_frames - 1)                     # m[i]: frame i -> i+1
        for s in range(0, n_frames - 1, 2048):
            e = min(s + 2048, n_frames - 1)
            m[s:e] = (small[s + 1:e + 1].float() - small[s:e].float()).pow(2).mean(dim=(1, 2, 3))
        starts = ds.valid_starts[ti].numpy()
        # frames used are obs[start : start+T], i.e. transitions start .. start+T-2
        win = np.lib.stride_tricks.sliding_window_view(m.numpy() / 255.0 ** 2, T - 1)
        score = np.percentile(win[starts], pct, axis=1)
        ep_of = ds.ep[ti]
        picked, per_ep = [], defaultdict(int)
        for j in np.argsort(-score):
            e = int(ep_of[int(starts[j])])
            if per_ep[e] < 2 and all(abs(int(starts[j]) - int(starts[p])) >= T // 2 for p in picked):
                picked.append(j)
                per_ep[e] += 1
            if len(picked) >= n_per_task:
                break
        print(f"[motion] {ds.tasks[ti]}: {len(starts)} windows scanned, picked {len(picked)} "
              f"(p{pct:g} motion {score[picked[-1]]:.2e} .. {score[picked[0]]:.2e}, "
              f"task median {np.median(score):.2e})", flush=True)
        out.extend(lo + int(j) for j in sorted(picked))
        lo = hi
    return out


def _window_actions(actions, act_mask, t0, t1):
    """actions/mask for frames [t0, t1) of a sliding window. Training windows
    (parse_batch) always carry a zero action with mask 0 on their first frame,
    so once the window has slid (t0 > 0) its first slot is cleared the same way."""
    a, m = actions[:, t0:t1], act_mask[:, t0:t1]
    if t0 > 0:
        a, m = a.clone(), m.clone()
        a[:, 0] = 0
        m[:, 0] = 0
    return a, m


def rollout_windowed(dyn, z_gt, *, ctx, n_steps, max_ctx, k_max, sched, actions,
                     act_mask, tau_ctx, lang_emb):
    """train_dynamics.sample_autoregressive_packed_sequence with the context
    capped at max_ctx-1 past frames (+1 predicted = the training seq_len).
    Identical to it while t < max_ctx; beyond that the oldest frames slide out
    (RoPE is relative, so a slid window looks like a mid-episode training one)."""
    outs = [z_gt[:, t] for t in range(ctx)]
    for t in range(ctx, ctx + n_steps):
        t0 = max(0, t - (max_ctx - 1))
        a_win, m_win = _window_actions(actions, act_mask, t0, t + 1)
        z_next = sample_one_timestep_packed(
            dyn, past_packed=torch.stack(outs[t0:t], dim=1), k_max=k_max, sched=sched,
            actions=a_win, act_mask=m_win, tau_ctx=tau_ctx, lang_emb=lang_emb,
        )
        outs.append(z_next)
    return torch.stack(outs, dim=1)


@torch.no_grad()
def rollout_pass(args, ds, indices, encoder, decoder, dyn, tok_info, dyn_info, device):
    """Per-sequence rollout dPSNR (mirrors run_dynamics_eval, but per sequence),
    [1] against the raw last context frame (paper) and [2] against its decoded
    version (ours). Also returns the packed GT latents + actions for the
    shuffle pass (on CPU)."""
    H, W, C, patch = (int(tok_info[k]) for k in ("H", "W", "C", "patch"))
    k_max = int(dyn_info["k_max"])
    sched = make_tau_schedule(k_max=k_max, schedule="shortcut", d=args.eval_d)
    ctx, hor = args.ctx, args.horizon
    hor_all = hor + args.extra_horizon

    loader = DataLoader(torch.utils.data.Subset(ds, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)
    rows, z_all, a_all, m_all, l_all = [], [], [], [], []
    viz = {"real": [], "wm": [], "psnr_pred_t": [], "psnr_floor_t": []}
    keep_frames = args.viz_per_task > 0
    k = 0
    for batch in loader:
        frames, actions, act_mask, _r, lang = parse_batch(batch, use_rewards=False, device=device)
        B, T = frames.shape[:2]
        assert T >= ctx + hor_all, f"seq_len {T} < ctx+horizon {ctx + hor_all}"

        # fp32 (TF32) encoder/decoder, as in run_dynamics_eval (called outside autocast).
        z_btLd, _ = encoder(temporal_patchify(frames, patch))
        n_sp = z_btLd.shape[2] // args.packing_factor
        z_gt = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_sp, k=args.packing_factor)

        z_pred = rollout_windowed(
            dyn, z_gt, ctx=ctx, n_steps=hor_all, max_ctx=args.max_ctx, k_max=k_max,
            sched=sched, actions=actions, act_mask=act_mask, tau_ctx=args.tau_ctx, lang_emb=lang,
        )
        # Decode the whole sequence (GT context latents + rollout) in one pass,
        # as run_dynamics_eval does: the decoder's time attention is causal, so
        # decoding the horizon alone would change the predicted frames.
        pred_all = decode_packed_to_frames(
            decoder, z_packed=z_pred[:, :ctx + hor_all], H=H, W=W, C=C, patch=patch,
            packing_factor=args.packing_factor,
        ).float()                                                        # (B, ctx+hor_all, ...)
        pred_h = pred_all[:, ctx:]                                       # (B, hor_all, ...)
        gt_h = frames[:, ctx:ctx + hor_all].float()
        psnr = lambda mse: 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))

        mse_pred_t = (pred_h - gt_h).pow(2).mean(dim=(2, 3, 4))         # (B, hor_all)
        # Labels: the paper's window only (first `hor` horizon frames).
        mse_pred = mse_pred_t[:, :hor].mean(dim=1)                       # (B,)
        psnr_pred = psnr(mse_pred)

        # ---- [1] PAPER: floor = raw last context frame, repeated ----
        floor_h = frames[:, ctx - 1:ctx].float().expand_as(gt_h)
        mse_floor_t = (floor_h - gt_h).pow(2).mean(dim=(2, 3, 4))
        mse_floor = mse_floor_t[:, :hor].mean(dim=1)
        psnr_floor = psnr(mse_floor)

        # ---- [2] OURS: floor = DECODED last context frame, repeated ----
        # Frame ctx-1 of the same decode pass (z_pred[:, ctx-1] is the GT latent).
        floor_dec_h = pred_all[:, ctx - 1:ctx].expand_as(gt_h)
        mse_floor_dec_t = (floor_dec_h - gt_h).pow(2).mean(dim=(2, 3, 4))
        mse_floor_dec = mse_floor_dec_t[:, :hor].mean(dim=1)
        psnr_floor_dec = psnr(mse_floor_dec)
        # Ground-truth scene motion over the whole rolled-out window, used to
        # pick dynamic sequences for the videos.
        seq = frames[:, ctx - 1:ctx + hor_all].float()
        gt_motion = (seq[:, 1:] - seq[:, :-1]).pow(2).mean(dim=(1, 2, 3, 4))

        viz["psnr_pred_t"].append(psnr(mse_pred_t).cpu())
        viz["psnr_floor_t"].append(psnr(mse_floor_t).cpu())
        if keep_frames:
            to_u8 = lambda x: (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            viz["real"].append(to_u8(frames[:, :ctx + hor_all]))
            viz["wm"].append(to_u8(pred_all))

        for b in range(B):
            idx = indices[k + b]
            ti, start = ds._lookup(idx)
            rows.append({
                "task": ds.tasks[ti], "idx": idx, "start": start,
                "gt_motion": float(gt_motion[b]),
                # [1] paper
                "psnr_pred": float(psnr_pred[b]), "psnr_floor": float(psnr_floor[b]),
                "delta_psnr": float(psnr_pred[b] - psnr_floor[b]),
                "mse_pred": float(mse_pred[b]), "mse_floor": float(mse_floor[b]),
                # [2] ours
                "psnr_floor_dec": float(psnr_floor_dec[b]),
                "delta_psnr_dec": float(psnr_pred[b] - psnr_floor_dec[b]),
                "mse_floor_dec": float(mse_floor_dec[b]),
            })
        k += B
        z_all.append(z_gt.cpu())
        a_all.append(actions.cpu()); m_all.append(act_mask.cpu()); l_all.append(lang.cpu())
        print(f"  [rollout] {k}/{len(indices)}", flush=True)

    viz = {key: (torch.cat(v) if v else None) for key, v in viz.items()}
    return rows, torch.cat(z_all), torch.cat(a_all), torch.cat(m_all), torch.cat(l_all), viz


@torch.no_grad()
def shuffle_pass(args, dyn, dyn_info, z_pool, a_pool, m_pool, l_pool, device):
    """Per-(sequence, step) one-step MSE with true and batch-shuffled actions.
    The inner loop is the same as train_dynamics.run_action_attribution_eval.
    _pred_mse; the differences are that it is chunked over sequences (the pool
    can be large), keeps the per-step MSEs, and caps the context at max_ctx
    (only matters past frame 23, i.e. with --extra_horizon). The permutation is
    drawn over the whole pool."""
    k_max = int(dyn_info["k_max"])
    sched = make_tau_schedule(k_max=k_max, schedule="shortcut", d=args.attr_d)
    K, tau_s, tau_i_s = int(sched["K"]), sched["tau"], sched["tau_idx"]
    dt, e_step = float(sched["dt"]), int(sched["e"])
    emax = int(round(math.log2(k_max)))
    ctx_sig_idx = min(int(round((1.0 - args.tau_ctx) * k_max)), k_max) if args.tau_ctx > 0 else k_max

    BP, TP, S, D = z_pool.shape
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 12345)
    # Derangement (uniform, by rejection): a fixed point would keep its own
    # actions and get ratio == 1 exactly, i.e. a spurious "action ignored" label.
    while True:
        perm = torch.randperm(BP, generator=gen)
        if BP < 2 or not (perm == torch.arange(BP)).any():
            break

    def pred_mse(z, z_past, noise, act, msk, lang):
        """Teacher-forced one-step MSE per (seq, t), t in [1, TP-1] -> (b, TP-1)."""
        b = z.shape[0]
        out = []
        for t in range(1, TP):
            t0 = max(0, t - (args.max_ctx - 1))
            n = t - t0 + 1
            a_win, m_win = _window_actions(act, msk, t0, t + 1)
            zt = noise[:, t - 1:t].clone()
            step_idxs = torch.full((b, n), emax, device=device, dtype=torch.long)
            step_idxs[:, -1] = e_step
            sig_idxs = torch.full((b, n), ctx_sig_idx, device=device, dtype=torch.long)
            for i in range(K):
                sig_idxs[:, -1] = int(tau_i_s[i])
                # fp32 (TF32), as in run_action_attribution_eval (called outside autocast).
                x1_hat, _ = dyn(a_win, step_idxs, sig_idxs,
                                torch.cat([z_past[:, t0:t], zt], dim=1),
                                act_mask=m_win, agent_tokens=None, lang_emb=lang)
                x1 = x1_hat[:, -1:].float()
                zt = zt + (x1 - zt) / max(1e-4, 1.0 - float(tau_s[i])) * dt
            out.append((zt[:, 0] - z[:, t]).pow(2).mean(dim=(1, 2)))
        return torch.stack(out, dim=1)

    mse_real, mse_shuf = [], []
    for s in range(0, BP, args.batch_size):
        sl = slice(s, min(s + args.batch_size, BP))
        b = sl.stop - sl.start
        z = z_pool[sl].to(device).float()
        # Shared noise for both interventions, drawn per chunk from a
        # chunk-seeded generator so results do not depend on batch order.
        g = torch.Generator(device="cpu").manual_seed(args.seed * 1_000_003 + s)
        noise = torch.randn((b, TP - 1, S, D), generator=g).to(device)
        if args.tau_ctx > 0:
            ctx_noise = torch.randn(z.shape, generator=g).to(device)
            z_past = (1.0 - args.tau_ctx) * z + args.tau_ctx * ctx_noise
        else:
            z_past = z
        lang = l_pool[sl].to(device)
        mse_real.append(pred_mse(z, z_past, noise, a_pool[sl].to(device), m_pool[sl].to(device), lang).cpu())
        p = perm[sl]
        mse_shuf.append(pred_mse(z, z_past, noise, a_pool[p].to(device), m_pool[p].to(device), lang).cpu())
        print(f"  [shuffle] {sl.stop}/{BP}", flush=True)

    return torch.cat(mse_real), torch.cat(mse_shuf), perm


def select_for_viz(rows, n_per_task: int, dynamic_third: bool = True):
    """Up to n_per_task sequences per task, round-robin over the label
    combinations (both, divergent only, ignored only, neither), most dynamic
    first within each. With random windows, only the most dynamic third (by
    mean ground-truth motion) is eligible; --pick motion windows are all."""
    order = [(1, 1), (1, 0), (0, 1), (0, 0)]
    by_task = defaultdict(list)
    for i, r in enumerate(rows):
        by_task[r["task"]].append(i)
    out = []
    for ids in by_task.values():
        ids = sorted(ids, key=lambda i: -rows[i]["gt_motion"])
        if dynamic_third:
            ids = ids[:max(n_per_task, len(ids) // 3)]
        pools = {c: [i for i in ids if (rows[i]["scene_divergent"], rows[i]["action_ignored"]) == c]
                 for c in order}
        picked = []
        while len(picked) < n_per_task and any(pools.values()):
            for c in order:
                if pools[c] and len(picked) < n_per_task:
                    picked.append(pools[c].pop(0))
        out.extend(picked)
    return out


def write_label_video(path, real, wm, psnr_pred_t, psnr_floor_t, ratio_t, row, ctx, hor, fps):
    """Top: real (left) | world model (right), 2x upscaled; the WM side is the
    tokenizer reconstruction for the ctx context frames and the open-loop
    rollout after. Bottom (flat): the per-frame metrics behind the two labels,
    green where the model behaves (WM beats copy-last-frame / actions matter),
    red where it does not, and a red bar at the current frame.
      real, wm: (T, 3, H, W) uint8. psnr_*_t: (T-ctx,) over the rollout.
      ratio_t: (T-1,) per-step shuffle ratio; entry j predicts frame j+1.
      The labels use frames [ctx, ctx+hor) and steps [1, ctx+hor)."""
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, MultipleLocator, NullFormatter, NullLocator
    from PIL import Image, ImageDraw, ImageFont

    GREEN, RED, INK, INK2, GRID = "#1b9e4b", "#d32f2f", "#0b0b0b", "#52514e", "#e6e5e1"
    T, _, H, W = real.shape
    s, gap = 2, 8
    top_w = 2 * W * s + gap
    plot_h = 340
    t_win = ctx + hor                          # end of the paper's label window
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(top_w / 100, plot_h / 100), dpi=100, sharex=True)
    div, ign = bool(row["scene_divergent"]), bool(row["action_ignored"])

    # --- scene: WM rollout vs copy-last-frame, both against the real frame ---
    t_h = np.arange(ctx, T)
    good = psnr_pred_t >= psnr_floor_t
    a1.fill_between(t_h, psnr_pred_t, psnr_floor_t, where=good, interpolate=True,
                    color=GREEN, alpha=0.25, lw=0)
    a1.fill_between(t_h, psnr_pred_t, psnr_floor_t, where=~good, interpolate=True,
                    color=RED, alpha=0.25, lw=0)
    a1.plot(t_h, psnr_pred_t, color="#2a78d6", lw=1.8, label="world model")
    a1.plot(t_h, psnr_floor_t, color=INK2, lw=1.2, ls="--", label="copy last context frame")
    # Context: the tokenizer round trip, i.e. the best PSNR the WM can reach.
    rec = [10 * np.log10(255.0 ** 2 / max(np.mean((real[t].astype(np.float64)
                                                    - wm[t].astype(np.float64)) ** 2), 1e-9))
           for t in range(ctx)]
    a1.plot(np.arange(ctx), rec, color="#2a78d6", lw=1.2, ls=":", label="tokenizer recon (ceiling)")
    a1.set_ylabel("PSNR vs real\n(dB)", fontsize=8, color=INK2)
    a1.set_title(f"Scene   dPSNR(t={ctx}-{t_win - 1}) = {row['delta_psnr']:+.2f} dB   ->   "
                 f"{'DIVERGENT' if div else 'ok'}", fontsize=9, loc="left",
                 color=RED if div else GREEN, fontweight="bold")
    a1.legend(fontsize=7, loc="lower left", frameon=False, ncol=3)

    # --- actions: how much worse one-step prediction gets with shuffled actions ---
    t_s = np.arange(1, T)
    r = np.clip(ratio_t, 1e-3, None)
    thr = np.full_like(r, ACTION_IGNORED_RATIO)
    a2.fill_between(t_s, r, thr, where=r > thr, interpolate=True, color=GREEN, alpha=0.25, lw=0)
    a2.fill_between(t_s, r, thr, where=r <= thr, interpolate=True, color=RED, alpha=0.25, lw=0)
    a2.plot(t_s, r, color="#eb6834", lw=1.8, label="shuffled / true one-step error")
    a2.axhline(ACTION_IGNORED_RATIO, color=INK2, ls="--", lw=1.0, label="threshold 1.1")
    a2.set_yscale("log")
    lo, hi = min(r.min(), 0.9), max(r.max(), 2.0)
    a2.set_ylim(lo / 1.1, hi * 1.1)
    ticks = [v for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200) if lo / 1.1 <= v <= hi * 1.1]
    a2.yaxis.set_major_locator(FixedLocator(ticks))
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a2.yaxis.set_minor_locator(NullLocator())
    a2.yaxis.set_minor_formatter(NullFormatter())
    a2.set_ylabel("action shuffle\nratio", fontsize=8, color=INK2)
    a2.set_xlabel("frame t", fontsize=8, color=INK2)
    a2.set_title(f"Actions   ratio(t=1-{t_win - 1}) = {row['shuffle_ratio']:.2f}   ->   "
                 f"{'IGNORED' if ign else 'ok'}   (shuffled actions from: "
                 f"{row['shuffle_donor_task']})", fontsize=9, loc="left",
                 color=RED if ign else GREEN, fontweight="bold")
    a2.legend(fontsize=7, loc="upper left", frameon=False, ncol=2)

    a1.axvspan(-0.5, ctx - 0.5, color=GRID, zorder=0)
    a1.text(ctx / 2 - 0.5, 0.92, "context", transform=a1.get_xaxis_transform(),
            fontsize=7, ha="center", va="top", color=INK2)
    for a in (a1, a2):
        a.set_xlim(-0.5, T - 0.5)
        if T > t_win:                       # beyond the paper's label window
            a.axvspan(t_win - 0.5, T - 0.5, color="#f4f3ef", zorder=0)
            a.axvline(t_win - 0.5, color=INK2, lw=0.8, ls=":")
        a.xaxis.set_major_locator(MultipleLocator(4 if T > 32 else 2))
        a.xaxis.set_minor_locator(MultipleLocator(1))
        a.tick_params(labelsize=7, colors=INK2)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    if T > t_win:
        a1.text(t_win, 0.92, " beyond label window", transform=a1.get_xaxis_transform(),
                fontsize=7, va="top", color=INK2)
    bars = [a.axvline(0, color="red", lw=2) for a in (a1, a2)]
    fig.tight_layout(pad=0.4, h_pad=0.6)

    up = lambda x: np.repeat(np.repeat(x.transpose(1, 2, 0), s, 0), s, 1)
    frames = []
    for t in range(T):
        for b in bars:
            b.set_xdata([t, t])
        fig.canvas.draw()
        plot = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).resize((top_w, plot_h))
        top = Image.fromarray(np.concatenate(
            [up(real[t]), np.zeros((H * s, gap, 3), np.uint8), up(wm[t])], axis=1))
        d = ImageDraw.Draw(top)
        phase = "context (recon)" if t < ctx else "imagined"
        for x0, text in ((6, f"real  t={t}/{T - 1}"), (W * s + gap + 6, f"world model  {phase}")):
            box = d.textbbox((x0, 4), text, font=font)
            d.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
            d.text((x0, 4), text, fill=(255, 255, 255), font=font)
        frames.append(np.concatenate([np.asarray(top), np.asarray(plot)], axis=0))
    plt.close(fig)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", macro_block_size=1, quality=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--data_dir", default="./data/test")
    p.add_argument("--frames_dir", default="./data/test-shards")
    p.add_argument("--tasks_json", default="../tasks.json")
    p.add_argument("--tokenizer_ckpt", default="./checkpoints/combined/tokenizer.pt")
    p.add_argument("--dynamics_ckpt", default="./checkpoints/combined/dynamics.pt")
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--lang_dim", type=int, default=512)
    p.add_argument("--n_per_task", type=int, default=45,
                   help="sequences per task; paper: ~9k over 200 tasks = 45. 0 = all windows")
    p.add_argument("--pick", choices=["random", "motion"], default="random",
                   help="random: paper-style sample (use for statistics). motion: the windows "
                        "that keep moving throughout (use for looking at the model)")
    p.add_argument("--motion_pct", type=float, default=25.0,
                   help="--pick motion scores a window by this percentile of its per-frame motion")
    p.add_argument("--ctx", type=int, default=8)
    p.add_argument("--horizon", type=int, default=16, help="label window: frames [ctx, ctx+horizon)")
    p.add_argument("--extra_horizon", type=int, default=0,
                   help="roll out this many more frames past the label window (videos only)")
    p.add_argument("--max_ctx", type=int, default=24,
                   help="sliding context length (training seq_len); only binds past frame 23")
    p.add_argument("--eval_d", type=float, default=0.25, help="rollout shortcut step size")
    p.add_argument("--attr_d", type=float, default=1.0, help="shuffle-ratio shortcut step size")
    p.add_argument("--tau_ctx", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--viz_per_task", type=int, default=0,
                   help="write this many label videos per task (<out>_viz/). Keeps the decoded "
                        "frames in RAM (~1.3 GB per 200 seqs at 24 frames)")
    p.add_argument("--viz_fps", type=int, default=6)
    p.add_argument("--out", required=True, help="per-sequence CSV; a .json summary is written next to it")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    tok, tok_info = load_tokenizer_from_ckpt(args.tokenizer_ckpt, device)
    dyn, _rew, _pol, dyn_info = load_dynamics_from_ckpt(
        args.dynamics_ckpt, device=device, d_bottleneck=int(tok_info["d_bottleneck"]),
        n_latents=int(tok_info["n_latents"]), packing_factor=args.packing_factor)
    encoder, decoder = tok.encoder.eval(), tok.decoder.eval()
    dyn.eval()

    n_lab = args.ctx + args.horizon                    # paper label window
    ds = WMDataset(data_dir=args.data_dir, frames_dir=args.frames_dir,
                   seq_len=n_lab + args.extra_horizon, img_size=224, action_dim=16,
                   lang_dim=args.lang_dim, tasks_json=args.tasks_json, tasks=args.tasks,
                   verbose=True)
    if args.pick == "motion":
        indices = motion_indices(ds, args.n_per_task, args.motion_pct)
        print("[labels] NOTE: --pick motion selects dynamic windows on purpose; the per-task "
              "fractions below are not representative of the task.")
    else:
        indices = select_indices(ds, args.n_per_task, args.seed)
    print(f"[labels] {len(indices)} sequences over {ds.num_tasks} tasks")
    if ds.num_tasks < 10:
        print(f"[labels] note: shuffle pool spans {ds.num_tasks} task(s) vs 200 in the paper, so "
              "shuffled actions come from other embodiments less often.")

    rows, z_pool, a_pool, m_pool, l_pool, viz = rollout_pass(
        args, ds, indices, encoder, decoder, dyn, tok_info, dyn_info, device)
    mse_real, mse_shuf, perm = shuffle_pass(args, dyn, dyn_info, z_pool, a_pool, m_pool, l_pool, device)

    for i, r in enumerate(rows):
        # ---- [1] PAPER labels ----
        # One-step predictions of frames 1..n_lab-1 (the paper's window).
        mr, ms = float(mse_real[i, :n_lab - 1].mean()), float(mse_shuf[i, :n_lab - 1].mean())
        r["attr_mse_real"], r["attr_mse_shuf"] = mr, ms
        r["shuffle_ratio"] = ms / max(mr, 1e-12)
        r["shuffle_donor_task"] = rows[int(perm[i])]["task"]
        r["scene_divergent"] = int(r["delta_psnr"] <= SCENE_DIVERGENT_DB)
        r["action_ignored"] = int(r["shuffle_ratio"] <= ACTION_IGNORED_RATIO)
        # ---- [2] OUR labels ----
        r["scene_divergent_dec"] = int(r["delta_psnr_dec"] <= SCENE_DIVERGENT_DEC_DB)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    summary = {"args": vars(args), "seconds": time.time() - t0, "tasks": {}}
    print(f"\n{'':29s} | {'[1] PAPER':^48s} | {'[2] OURS':^20s}")
    print(f"{'task':24s} {'n':>4s} | {'dPSNR med':>9s} {'divergent':>9s} "
          f"{'ratio med':>9s} {'pool ratio':>10s} {'ignored':>7s} | {'dPSNRdec med':>12s} {'div_dec':>7s}")
    for task, rs in by_task.items():
        d = np.array([r["delta_psnr"] for r in rs])
        q = np.array([r["shuffle_ratio"] for r in rs])
        dd = np.array([r["delta_psnr_dec"] for r in rs])
        pool = (sum(r["attr_mse_shuf"] for r in rs) / max(sum(r["attr_mse_real"] for r in rs), 1e-12))
        s = {"n": len(rs),
             # [1] paper
             "delta_psnr_median": float(np.median(d)), "delta_psnr_mean": float(d.mean()),
             "frac_scene_divergent": float(np.mean([r["scene_divergent"] for r in rs])),
             "shuffle_ratio_median": float(np.median(q)), "shuffle_ratio_pool": float(pool),
             "frac_action_ignored": float(np.mean([r["action_ignored"] for r in rs])),
             # [2] ours
             "delta_psnr_dec_median": float(np.median(dd)), "delta_psnr_dec_mean": float(dd.mean()),
             "frac_scene_divergent_dec": float(np.mean([r["scene_divergent_dec"] for r in rs]))}
        summary["tasks"][task] = s
        print(f"{task:24s} {s['n']:4d} | {s['delta_psnr_median']:9.2f} {s['frac_scene_divergent']:9.2f} "
              f"{s['shuffle_ratio_median']:9.2f} {s['shuffle_ratio_pool']:10.2f} {s['frac_action_ignored']:7.2f} | "
              f"{s['delta_psnr_dec_median']:12.2f} {s['frac_scene_divergent_dec']:7.2f}")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.viz_per_task > 0:
        viz_dir = os.path.splitext(args.out)[0] + "_viz"
        ratio_t = (mse_shuf / mse_real.clamp_min(1e-12)).numpy()     # (N, T-1)
        for i in select_for_viz(rows, args.viz_per_task, dynamic_third=args.pick == "random"):
            r = rows[i]
            path = os.path.join(viz_dir, f"{r['task']}_idx{r['idx']}_div{r['scene_divergent']}"
                                         f"_ign{r['action_ignored']}.mp4")
            write_label_video(path, viz["real"][i].numpy(), viz["wm"][i].numpy(),
                              viz["psnr_pred_t"][i].numpy(), viz["psnr_floor_t"][i].numpy(),
                              ratio_t[i], r, args.ctx, args.horizon, args.viz_fps)
            print(f"  [viz] {path}", flush=True)
    print(f"\n[labels] {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
