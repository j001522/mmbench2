# plan_mpc_env.py
"""
Closed-loop MPC evaluation: plan in imagination, act in the real environment.

Each control cycle:
  - encode the last --ctx_frames real observations into packed latents;
  - warm-start CEM's mean with a BC rollout of the integrated policy head;
  - run CEM over action sequences, scoring every candidate by rolling it out
    through the world model (predicted return from the reward head, plus an
    optional terminal latent-distance-to-goal term standing in for a value
    function);
  - execute the first --replan_every actions of the winning plan in the env;
  - repeat from the new real observation.

Unlike plan_cem.py (offline, open-loop, scored against recorded episodes),
everything reported here is ground truth from the environment: real return and
real success rate. The planner components are imported from plan_cem.py.

Typical usage:
    python plan_mpc_env.py --task cup-catch \
        --tokenizer_ckpt ./checkpoints/combined/tokenizer.pt \
        --dynamics_ckpt  ./checkpoints/combined/dynamics.pt \
        --n_episodes 5 --output ./logs/mpc_cup-catch.csv
"""
import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')

import argparse
import csv
import json
import sys
import time
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_cem import (
    ALL_COST_KINDS,
    bc_rollout_integrated,
    build_models,
    cem_plan,
    decode_packed_sequence,
    encode_frames_to_packed,
    imagine_batch,
    list_episode_ids,
    load_act_meta,
    load_episode,
    load_lang_emb,
    goal_scale,
    latent_cost,
    needs_goal,
    needs_reward_head,
    needs_value_head,
    predict_rewards,
    reward_cost,
    _tile_row,
)
from interactive import make_tau_schedule, sample_one_timestep_packed, unpack_spatial_to_bottleneck
from model import temporal_unpatchify
from plan_halluc import hausdorff as mask_hausdorff
from sim_planner import SimHandle, sim_cem_plan, to_env_actions
from train_heads_offline import extract_h

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -------------------- goal latent -------------------- #

def load_goal_latent(args, encoder, info, device, use_amp, rng) -> Optional[torch.Tensor]:
    """Encode the last frame of a demo episode as the goal state. Returns None
    when the cost does not use a goal."""
    if not needs_goal(args.cost):
        return None
    ep_ids = list_episode_ids(args.task, args.data_dir)
    if not ep_ids:
        raise SystemExit(f"no demo episodes for task {args.task} under {args.data_dir}")
    ep_id = int(args.goal_ep) if args.goal_ep >= 0 else int(rng.choice(ep_ids))
    epi = load_episode(args.task, args.data_dir, args.frames_dir, ep_id,
                       shard_size=int(args.shard_size))
    z = encode_frames_to_packed(
        encoder, epi.frames[-1:].to(device), patch=int(info["patch"]),
        n_spatial=int(info["n_spatial"]), packing_factor=int(info["packing_factor"]),
        use_amp=use_amp,
    )
    print(f"[mpc] goal = last frame of episode {ep_id} (len {epi.frames.shape[0]})")
    return z[0]


# -------------------- render backend diagnostics -------------------- #

def describe_render_backend() -> str:
    """Which EGL device THIS process renders on (egl_probe.py only tests fresh
    subprocesses). dm_control and gymnasium/mujoco each keep a module-level
    EGL_DISPLAY; eglGetPlatformDisplayEXT returns the same handle for the same
    device, so matching handles recovers the device index. Vendor 'NVIDIA' vs
    'Mesa Project' separates GPU from software rendering. Call after the first
    render, since the displays are created lazily."""
    import ctypes
    parts = [f"MUJOCO_GL={os.environ.get('MUJOCO_GL', 'unset')}",
             f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID', 'unset')}"]
    try:
        from mujoco.egl import egl_ext as EGL

        def _ptr(x):
            return ctypes.cast(x, ctypes.c_void_p).value

        devices = EGL.eglQueryDevicesEXT()
        for name in ("dm_control._render.pyopengl.egl_renderer", "mujoco.egl"):
            disp = getattr(sys.modules.get(name), "EGL_DISPLAY", None)
            if disp is None:
                continue
            idx = next((i for i, dv in enumerate(devices) if _ptr(EGL.eglGetPlatformDisplayEXT(
                EGL.EGL_PLATFORM_DEVICE_EXT, dv, None)) == _ptr(disp)), None)
            vendor = EGL.eglQueryString(disp, EGL.EGL_VENDOR)
            parts.append(f"{name.split('.')[0]}: egl_device={idx} of {len(devices)} "
                         f"vendor={vendor.decode() if vendor else '?'}")
    except Exception as e:
        parts.append(f"query failed ({type(e).__name__}: {e})")
    return " ".join(parts)


# -------------------- hallucination labels during planning -------------------- #
# The two labels of hallucination_labels.py (MMBench2 paper), per replan chunk:
#   scene divergent: the chunk's open-loop WM prediction is no better than
#     copying the real frame the chunk started from (chunk dPSNR <= 0).
#   action ignored: one-step prediction of each executed frame from the real
#     history gets <= 1.1x worse when the actions are replaced. Closed loop has
#     no batch of other sequences to shuffle in, so the replacement is uniform
#     random actions (every slot of the window, like the paper's shuffle).

def _psnr_u8(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(mse in [0,1] units, psnr dB) between two uint8 frames."""
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)) / 255.0 ** 2
    return mse, 10.0 * math.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def shuffle_mse_executed(dyn, z_hist, a_hist, frames_idx, *, act_dim, act_mask_1d, k_max,
                         lang_emb, max_ctx, use_amp, gen, seed, tau_ctx=0.1):
    """Teacher-forced one-step MSE of each executed frame s (z_hist[s]) from the
    real window before it, with the executed actions and with uniform random
    ones; the same noise for both (forked RNG), so they differ only in the
    actions. Settings as the paper's shuffle ratio (d=1.0 shortcut, tau_ctx 0.1);
    window = max_ctx frames incl. the predicted one, first slot action-free as
    in training."""
    device = z_hist[0].device
    sched1 = make_tau_schedule(k_max=k_max, schedule="shortcut", d=1.0)
    m1 = act_mask_1d.view(1, 1, 16)
    out_true, out_rand = [], []
    for s in frames_idx:
        t0 = max(0, s - (max_ctx - 1))
        past = torch.stack(z_hist[t0:s], dim=0).unsqueeze(0)
        a_true = torch.stack(a_hist[t0:s + 1], dim=0).unsqueeze(0).clone()
        mask = m1.expand(1, s + 1 - t0, 16).clone()
        a_rand = torch.zeros_like(a_true)
        a_rand[..., :act_dim] = torch.rand((1, s + 1 - t0, act_dim), generator=gen,
                                           device=device) * 2 - 1
        a_rand = a_rand * m1
        for a in (a_true, a_rand):
            a[:, 0] = 0
        mask[:, 0] = 0
        res = []
        for a in (a_true, a_rand):
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(seed * 100_003 + s)
                z, _ = sample_one_timestep_packed(
                    dyn, past_packed=past, k_max=k_max, sched=sched1, actions=a,
                    act_mask=mask, use_amp=use_amp, tau_ctx=tau_ctx, lang_emb=lang_emb)
            res.append(float((z.float()[0] - z_hist[s].float()).pow(2).mean()))
        out_true.append(res[0])
        out_rand.append(res[1])
    return out_true, out_rand


def halluc_chunks(replans: List[dict]) -> List[dict]:
    """Per replan chunk: [start, end) env steps, dPSNR, shuffle ratio, labels."""
    out, t = [], 0
    for r in replans:
        h = r.get("halluc")
        if not h:
            continue
        n = len(h["mse_wm"])
        d = (10.0 * math.log10(max(sum(h["mse_floor"]), 1e-12) / max(sum(h["mse_wm"]), 1e-12))
             if h["mse_wm"] else float("nan"))
        q = (sum(h["shuf_mse_rand"]) / max(sum(h["shuf_mse_true"]), 1e-12)
             if h.get("shuf_mse_true") else float("nan"))
        out.append({"start": t, "end": t + n, "delta_psnr": d, "shuffle_ratio": q,
                    "scene_divergent": int(d <= 0.0), "action_ignored": int(q <= 1.1)})
        t += n
    return out


def halluc_panel(replans: List[dict], n_steps: int, width: int, height: int = 300):
    """Whole-episode chart for under the video: per-step PSNR of the WM
    prediction and of copy-the-chunk's-start-frame (green where the WM wins,
    red where it loses), and the per-chunk action shuffle ratio vs 1.1.
    Returns (background uint8 HxWx3, step -> pixel column, (y0, y1) rows the
    moving bar spans)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    GREEN, RED, INK2 = "#1b9e4b", "#d32f2f", "#52514e"
    p_wm, p_fl = [], []
    for r in replans:
        h = r.get("halluc") or {}
        p_wm += h.get("psnr_wm", [])
        p_fl += h.get("psnr_floor", [])
    x = np.arange(1, len(p_wm) + 1)
    p_wm, p_fl = np.array(p_wm), np.array(p_fl)
    ch = halluc_chunks(replans)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(width / 100, height / 100), dpi=100, sharex=True)
    if len(x):
        good = p_wm >= p_fl
        a1.fill_between(x, p_wm, p_fl, where=good, interpolate=True, color=GREEN, alpha=0.3, lw=0)
        a1.fill_between(x, p_wm, p_fl, where=~good, interpolate=True, color=RED, alpha=0.3, lw=0)
        a1.plot(x, p_wm, color="#2a78d6", lw=1.0, label="world model")
        a1.plot(x, p_fl, color=INK2, lw=0.8, ls="--", label="copy frame at replan")
    n_div = sum(c["scene_divergent"] for c in ch)
    a1.set_title(f"Scene: WM prediction vs real frame   (divergent replans {n_div}/{len(ch)})",
                 fontsize=8, loc="left", color=INK2)
    a1.set_ylabel("PSNR (dB)", fontsize=7, color=INK2)
    a1.legend(fontsize=6, loc="lower left", frameon=False, ncol=2)

    qs = np.array([c["shuffle_ratio"] for c in ch], dtype=float)
    if len(ch) and np.isfinite(qs).any():
        xs = np.array([c["start"] for c in ch] + [ch[-1]["end"]], dtype=float) + 0.5
        q = np.clip(np.append(qs, qs[-1]), 1e-3, None)
        for c, v in zip(ch, q):                # one block per replan chunk
            a2.fill_between([c["start"] + 0.5, c["end"] + 0.5], [v, v], [1.1, 1.1],
                            color=GREEN if v > 1.1 else RED, alpha=0.3, lw=0)
        a2.plot(xs, q, drawstyle="steps-post", color="#eb6834", lw=1.2)
        a2.axhline(1.1, color=INK2, ls="--", lw=0.8)
        a2.set_yscale("log")
        lo, hi = min(np.nanmin(q), 0.9), max(np.nanmax(q), 2.0)
        a2.set_ylim(lo / 1.1, hi * 1.1)
        ticks = [v for v in (0.5, 1, 2, 5, 10, 20, 50, 100) if lo / 1.1 <= v <= hi * 1.1]
        a2.yaxis.set_major_locator(FixedLocator(ticks))
        a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a2.yaxis.set_minor_locator(NullLocator())
    n_ign = sum(c["action_ignored"] for c in ch)
    a2.set_title(f"Actions: one-step error, random / executed actions, per replan   "
                 f"(ignored replans {n_ign}/{len(ch)}; threshold 1.1)",
                 fontsize=8, loc="left", color=INK2)
    a2.set_ylabel("ratio", fontsize=7, color=INK2)
    a2.set_xlabel("env step", fontsize=7, color=INK2)
    for a in (a1, a2):
        a.set_xlim(0, max(n_steps, 1))
        a.tick_params(labelsize=6, colors=INK2)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3, h_pad=0.4)
    fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba())[..., :3][:, :width].copy()
    if bg.shape[1] < width:                    # figsize rounding can lose a pixel
        bg = np.pad(bg, ((0, 0), (0, width - bg.shape[1]), (0, 0)), constant_values=255)
    H = bg.shape[0]
    b1, b2 = a1.get_window_extent(), a2.get_window_extent()
    y0, y1 = int(H - max(b1.y1, b2.y1)), int(H - min(b1.y0, b2.y0))
    to_px = lambda t: int(round(a1.transData.transform((t, 0))[0]))
    plt.close(fig)
    return bg, to_px, (y0, y1)


def add_halluc_panel(vid: List[np.ndarray], replans: List[dict], scale: int = 2) -> List[np.ndarray]:
    """Upscale the (real | WM) video frames and put the whole-episode chart
    under them with a red bar at the current step (frame i = after step i)."""
    up = [np.repeat(np.repeat(f, scale, 0), scale, 1) for f in vid]
    width = up[0].shape[1]
    bg, to_px, (y0, y1) = halluc_panel(replans, len(vid) - 1, width)
    out = []
    for i, f in enumerate(up):
        panel = bg.copy()
        c = min(max(to_px(i), 1), panel.shape[1] - 2)
        panel[y0:y1, c - 1:c + 1] = (220, 30, 30)
        out.append(np.concatenate([f, panel], axis=0))
    return out


# -------------------- rollout video -------------------- #

def save_strip(path: str, frames_u8: List[np.ndarray], max_frames: int = 16) -> None:
    """Save a horizontal strip of evenly spaced real frames from an episode."""
    if not frames_u8:
        return
    T = len(frames_u8)
    idx = sorted(set(np.linspace(0, T - 1, num=min(max_frames, T), dtype=int).tolist()))
    sel = torch.stack([torch.from_numpy(frames_u8[i]) for i in idx]).float() / 255.0
    row = _tile_row(sel)
    arr = (row.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(arr).save(path)


def compose_frame(real_chw: np.ndarray, wm_chw: Optional[np.ndarray], gap_px: int = 4) -> np.ndarray:
    """(3,H,W) uint8 frames -> one (H, W[+gap+W], 3) uint8 video frame, real
    environment on the left, world-model prediction for the same step on the
    right."""
    real_hwc = np.transpose(real_chw, (1, 2, 0))
    if wm_chw is None:
        return np.ascontiguousarray(real_hwc)
    wm_hwc = np.transpose(wm_chw, (1, 2, 0))
    gap = np.zeros((real_hwc.shape[0], gap_px, 3), dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate([real_hwc, gap, wm_hwc], axis=1))


def stamp_steps(frames_hwc: List[np.ndarray]) -> List[np.ndarray]:
    """Burn 'step i/T' into the top-left corner of each video frame. Frame i is
    the observation after env step i (frame 0 = initial obs), so i matches the
    x axis of the _unc.png plot."""
    from PIL import ImageDraw, ImageFont
    T = len(frames_hwc) - 1
    w = frames_hwc[0].shape[1]
    try:
        font = ImageFont.load_default(size=max(10, w // 32))
    except TypeError:                                    # Pillow < 10.1: fixed-size bitmap font
        font = ImageFont.load_default()
    out = []
    for i, f in enumerate(frames_hwc):
        img = Image.fromarray(f)
        draw = ImageDraw.Draw(img)
        text = f"step {i:>{len(str(T))}d}/{T}"
        x0, y0, x1, y1 = draw.textbbox((4, 3), text, font=font)
        draw.rectangle((x0 - 3, y0 - 2, x1 + 3, y1 + 2), fill=(0, 0, 0))
        draw.text((4, 3), text, fill=(255, 255, 255), font=font)
        out.append(np.asarray(img))
    return out


def save_video(path_base: str, frames_hwc: List[np.ndarray], *, fps: int, kind: str) -> List[str]:
    """Write the episode as .mp4 and/or .gif. GIF is the safe option for a
    headless VS Code session: its image preview plays animated GIFs with no
    extension, while .mp4 needs one."""
    if not frames_hwc or kind == "none":
        return []
    os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
    frames_hwc = stamp_steps(frames_hwc)
    written = []

    if kind in ("mp4", "both"):
        try:
            import imageio.v2 as imageio  # imageio-ffmpeg ships its own ffmpeg binary
            path = path_base + ".mp4"
            imageio.mimwrite(path, frames_hwc, fps=fps, codec="libx264",
                             macro_block_size=1, quality=8)
            written.append(path)
        except Exception as e:
            print(f"  [video] mp4 failed ({type(e).__name__}: {e}); gif still written")
            kind = "both" if kind == "mp4" else kind

    if kind in ("gif", "both"):
        path = path_base + ".gif"
        imgs = [Image.fromarray(f) for f in frames_hwc]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(1, fps)), loop=0, optimize=True)
        written.append(path)

    return written


def _nanmean(values: List[float]) -> float:
    """Mean of the finite entries; nan when there are none (np.nanmean warns)."""
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


# -------------------- hallucination predictors (--log_unc) -------------------- #

def _rms(x: torch.Tensor) -> torch.Tensor:
    """RMS over every dim but the first: (N, ...) -> (N,)."""
    return x.float().pow(2).flatten(1).mean(1).sqrt()


@torch.inference_mode()
def roundtrip_residual(encoder, decoder, z: torch.Tensor, *, info: dict,
                       use_amp: bool) -> torch.Tensor:
    """u_r = RMS(z - encode(decode(z))) per latent: (N, S, D) -> (N,).

    Same quantity as uncertainty.URNormScorer. Each latent is decoded as its own
    one-frame clip and quantized to uint8 before re-encoding, matching how the
    real observations are encoded here, so u_r is on the same footing as e_z.
    """
    patch = int(info["patch"])
    pf = int(info["packing_factor"])
    z_btLd = unpack_spatial_to_bottleneck(
        z.float().unsqueeze(1), k=pf, d_bottleneck=int(info["d_bottleneck"]))  # (N,1,L,Db)
    dev = z.device
    with torch.autocast(device_type=dev.type, enabled=(use_amp and dev.type == "cuda")):
        patches = decoder(z_btLd)
    frames = temporal_unpatchify(
        patches, int(info["H"]), int(info["W"]), int(info["C"]), patch)[:, 0]
    frames = (frames.float().clamp(0, 1) * 255.0).round() / 255.0
    z_rt = encode_frames_to_packed(
        encoder, frames, patch=patch, n_spatial=int(info["n_spatial"]),
        packing_factor=pf, use_amp=use_amp)
    return _rms(z.float() - z_rt)


def _tolist(x: Optional[torch.Tensor]) -> list:
    return [] if x is None else [float(v) for v in x.detach().float().cpu()]


def plan_distance(a: torch.Tensor, b: torch.Tensor, K: int, act_mask_1d: torch.Tensor,
                  real_dim: int, mu_std: Optional[Tuple[dict, dict]] = None) -> dict:
    """Distance between two (H, act_dim) plans in the env actions actually
    executed (padded, masked, sliced; range [-1, 1]), RMS over steps and dims:
    d_exec over the K executed steps, d_first the first action, d_H the whole
    horizon, per_step the RMS over dims at each of the K steps. With
    mu_std=(cem_out_a, cem_out_b), also w2: the 2-Wasserstein distance between
    the final CEM elite Gaussians over the K steps (diagonal: per dim
    sqrt((mu_a-mu_b)^2 + (s_a-s_b)^2), RMS). Not KL: the elite std is clamped at
    --cem_min_std, so KL would mostly measure the std ratio.
    best_plan is one CEM sample (white noise at --warm_start_init_std around
    the warm start), so d_exec of two different samples sits at
    ~sqrt(2)*std whatever the planners prefer. d_avg (RMS over dims of the
    difference of the K-step mean actions) averages that noise down by
    sqrt(K); d_mu is d_exec on the final elite means."""
    env_a = lambda x: to_env_actions(x[None], act_mask_1d, real_dim)[0]
    ea, eb = env_a(a), env_a(b)
    d2 = (ea - eb) ** 2                                               # (H, real_dim)
    out = {"d_exec": float(np.sqrt(d2[:K].mean())), "d_first": float(np.sqrt(d2[0].mean())),
           "d_H": float(np.sqrt(d2.mean())), "per_step": np.sqrt(d2[:K].mean(axis=1)).tolist(),
           "d_avg": float(np.sqrt(((ea[:K].mean(0) - eb[:K].mean(0)) ** 2).mean()))}
    if mu_std is not None and all("mu" in o for o in mu_std):
        (oa, ob) = mu_std
        dm = (env_a(oa["mu"]) - env_a(ob["mu"])) ** 2
        w = dm + (env_a(oa["std"]) - env_a(ob["std"])) ** 2
        out["w2"] = float(np.sqrt(w[:K].mean()))
        out["d_mu"] = float(np.sqrt(dm[:K].mean()))
    return out


# -------------------- one closed-loop episode -------------------- #

def run_episode(
    *, args, env, encoder, decoder, dyn, rew_head, policy_head, sched, info,
    device, use_amp, act_dim, act_mask_1d, lang_emb, z_goal, rng, ep_index: int,
    env_seed: int = 0, out_dir: str = "", value_head=None,
) -> dict:
    patch = int(info["patch"])
    n_spatial = int(info["n_spatial"])
    packing_factor = int(info["packing_factor"])
    k_max = int(info["k_max"])
    real_dim = int(env._real_action_dim)
    record = str(args.video) != "none"
    compare = record and bool(args.video_compare)
    # Random actions for the per-replan action-shuffle check (--halluc).
    halluc_gen = torch.Generator(device=device).manual_seed(int(env_seed))
    # --policy random: uniform actions, reproducible per episode (score floor).
    random_gen = torch.Generator(device=device).manual_seed(int(env_seed) + 7_919)

    def decode(z_seq: torch.Tensor) -> np.ndarray:
        """(T, S, D) packed latents -> (T, 3, H, W) uint8."""
        frames = decode_packed_sequence(
            decoder, z_seq, H=int(info["H"]), W=int(info["W"]), C=int(info["C"]),
            patch=patch, packing_factor=packing_factor,
            d_bottleneck=int(info["d_bottleneck"]), use_amp=use_amp,
        )
        return (frames.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()

    t_reset = time.time()
    obs_dict, _info = env.env.reset()
    frame = env._extract_rgb(obs_dict)                       # (3, H, W) uint8
    if ep_index == 0:
        print(f"[render] reset+first render {time.time() - t_reset:.2f}s | "
              f"{describe_render_backend()}", flush=True)

    # PLAN_RENDER_PROBE=1: time 3 extra renders (same call as the Pixels
    # wrapper) after each stage of the first replan of episode 0, to find the
    # stage after which rendering stalls. Diagnostic only; off by default.
    probe_on = os.environ.get("PLAN_RENDER_PROBE") == "1" and ep_index == 0
    pix = env.env
    while probe_on and type(pix).__name__ != "Pixels":
        pix = pix.env

    def render_probe(tag: str) -> None:
        if not probe_on:
            return
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t = time.time()
            pix.env.render(width=pix._size, height=pix._size)
            ts.append(time.time() - t)
        print(f"[probe] {tag:28s} render {1e3 * min(ts):9.1f} / {1e3 * max(ts):9.1f} ms "
              f"(min/max of 3)", flush=True)

    render_probe("after reset")

    # --compare_sim: CEM in the real env between save/restore (sim_planner.py),
    # and renders of counterfactual plans through the Pixels wrapper's env.
    sim = SimHandle(env.env) if bool(args.compare_sim) else None
    render_fn = None
    if sim is not None:
        pix_r = env.env
        while type(pix_r).__name__ != "Pixels":
            pix_r = pix_r.env
        render_fn = lambda: env._extract_rgb(pix_r.env.render(width=pix_r._size, height=pix_r._size))

    def encode(frame_u8: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(frame_u8).float().div(255.0).unsqueeze(0).to(device)
        return encode_frames_to_packed(
            encoder, t, patch=patch, n_spatial=n_spatial,
            packing_factor=packing_factor, use_amp=use_amp,
        )[0]

    # Context under the led-to convention: acts[i] is the action that produced
    # z_hist[i], so the first frame pairs with the zero action.
    z_hist: List[torch.Tensor] = [encode(frame)]
    a_hist: List[torch.Tensor] = [torch.zeros(16, device=device)]
    frames_u8: List[np.ndarray] = [frame]
    render_probe("after encode(frame0)")

    # Video: environment on the left, the model's prediction for the same step
    # on the right. The first video frame has no prediction yet, so it shows the
    # tokenizer round-trip of the initial observation instead.
    vid: List[np.ndarray] = []
    if record:
        wm0 = decode(z_hist[0].unsqueeze(0))[0] if compare else None
        vid.append(compose_frame(frame, wm0))
        render_probe("after decode(wm0)")

    rewards: List[float] = []
    success = 0.0
    replans: List[dict] = []
    step = 0
    done = False
    t_ep = time.time()

    while step < args.max_steps and not done:
        n_ctx = max(1, int(args.ctx_frames))
        z_ctx = torch.stack(z_hist[-n_ctx:], dim=0)          # (t, S, D)
        a_ctx = torch.stack(a_hist[-n_ctx:], dim=0)          # (t, 16)

        H_plan = min(int(args.horizon), args.max_steps - step)
        K_exec = max(1, min(int(args.replan_every), H_plan))
        bc_only = str(args.policy) == "bc"
        random_only = str(args.policy) == "random"
        if bc_only or random_only:
            # Nothing searches over the extra horizon, so only imagine the
            # actions that will actually be executed.
            H_plan = K_exec

        # BC rollout of the integrated policy head from the real context: CEM's
        # initial mean, or the executed plan itself when --policy bc.
        init_mu = None
        init_std = float(args.cem_init_std)
        t_bc = time.time()
        if bc_only or args.warm_start == "bc":
            _z_bc, init_mu = bc_rollout_integrated(
                policy_head, dyn, z_start=z_ctx, horizon=H_plan, act_dim=act_dim,
                act_mask_1d=act_mask_1d, sched=sched, k_max=k_max,
                lang_emb=lang_emb, use_amp=use_amp,
                past_actions=a_ctx, max_ctx=int(args.max_ctx), tau_ctx=float(args.tau_ctx),
            )
            init_mu = init_mu.clamp(-1.0, 1.0)
            init_std = float(args.warm_start_init_std)
        dt_bc = time.time() - t_bc
        if step == 0:
            render_probe("after bc_rollout")

        # u_r of CEM samples: chunked decode + re-encode of every imagined latent.
        ur_fn = None
        if str(args.unc_kind).startswith("u_r"):
            def ur_fn(z_flat: torch.Tensor) -> torch.Tensor:
                return torch.cat([
                    roundtrip_residual(encoder, decoder, z_flat[i:i + int(args.ur_chunk)],
                                       info=info, use_amp=use_amp)
                    for i in range(0, z_flat.shape[0], int(args.ur_chunk))])

        def wm_cem(gen):
            return cem_plan(
                dyn, z_start=z_ctx,
                z_goal=(z_goal if z_goal is not None else torch.zeros_like(z_hist[-1])),
                horizon=H_plan, act_dim=act_dim, act_mask_1d=act_mask_1d,
                sched=sched, k_max=k_max, lang_emb=lang_emb,
                cost_kind=args.cost,
                n_samples=args.cem_samples, n_elites=args.cem_elites,
                n_iters=args.cem_iters, init_std=init_std,
                min_std=args.cem_min_std, rollout_batch=args.rollout_batch,
                noise_beta=args.noise_beta,
                device=device, use_amp=use_amp, generator=gen,
                init_mu=init_mu, n_rollouts_elite=int(args.cem_n_rollouts),
                rew_head=rew_head, value_head=value_head,
                discount=float(args.discount),
                goal_weight=float(args.goal_weight),
                past_actions=a_ctx, max_ctx=int(args.max_ctx), tau_ctx=float(args.tau_ctx),
                unc_weight=float(args.unc_weight), log_pop=bool(args.log_cem_pop), log_K=K_exec,
                unc_kind=str(args.unc_kind), ur_fn=ur_fn,
                unc_mode=str(args.unc_mode), unc_thr=float(args.unc_thr),
                unc_thr_pct=float(args.unc_thr_pct), unc_rate=float(args.unc_rate),
            )

        def sim_value_score(plans: torch.Tensor, T: int):
            """--cost reward_value under --compare_sim: the true-model
            counterpart of value_cost. plans (N, >=T, act_dim) model space,
            executed for T steps in the simulator. Returns numpy (R_T, V_T, J_T),
            J_T = R_T + gamma^T V_T, where V_T is the SAME value head on h from
            the dynamics teacher-forced on the encoded TRUE frames, over the
            window imagination would attend to at that position (max_ctx past
            frames + the new one; real context frames fill it when T is short).
            h_mode "clean" (all slots at full signal): plan-time h is read off
            the last denoising iteration, so this is the closest deterministic
            match. A rollout that ends the episode early gets V = 0."""
            gam = float(args.discount)
            N = plans.shape[0]
            W = int(args.max_ctx) + 1 if int(args.max_ctx) > 0 else z_ctx.shape[0] + T
            n_roll = min(T, W)
            n_real = min(W - n_roll, z_ctx.shape[0])
            p16 = torch.zeros((N, T, 16), device=device)
            p16[..., :act_dim] = plans[:, :T]
            p16 = p16 * act_mask_1d.view(1, 1, 16)
            R, frames, n_steps = sim.returns_last_frames(
                to_env_actions(plans[:, :T], act_mask_1d, real_dim), render_fn, n_roll, discount=gam)
            V = np.zeros(N)
            ok = np.flatnonzero(n_steps == T)
            for s in range(0, len(ok), 32):
                idx = ok[s:s + 32]
                fr = torch.from_numpy(np.stack([f for i in idx for f in frames[i]])).float().div(255.0).to(device)
                z_roll = torch.cat([encode_frames_to_packed(
                    encoder, fr[c:c + 256], patch=patch, n_spatial=n_spatial,
                    packing_factor=packing_factor, use_amp=use_amp) for c in range(0, len(fr), 256)])
                z_roll = z_roll.view(len(idx), n_roll, *z_roll.shape[1:])
                a_roll = p16[torch.as_tensor(idx, device=device), T - n_roll:]
                if n_real > 0:
                    z_roll = torch.cat([z_ctx[-n_real:].unsqueeze(0).expand(len(idx), -1, -1, -1), z_roll], 1)
                    a_roll = torch.cat([a_ctx[-n_real:].unsqueeze(0).expand(len(idx), -1, -1), a_roll], 1)
                h = extract_h(dyn, z1=z_roll, actions=a_roll,
                              act_mask=act_mask_1d.view(1, 1, 16).expand(len(idx), z_roll.shape[1], 16),
                              lang_emb=None if lang_emb is None else lang_emb.expand(len(idx), -1),
                              k_max=k_max, mode="clean", use_amp=use_amp)
                with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                    V[idx] = value_head.predict(h)[:, -1].float().cpu().numpy()
            return R, V, R + gam ** T * V

        sim_score = None
        if sim is not None and value_head is not None:
            sim_score = lambda samples: sim_value_score(samples, H_plan)[2]

        def sim_cem(gen):
            return sim_cem_plan(
                sim, horizon=H_plan, act_dim=act_dim, act_mask_1d=act_mask_1d, real_dim=real_dim,
                n_samples=args.cem_samples, n_elites=args.cem_elites, n_iters=args.cem_iters,
                init_std=init_std, min_std=args.cem_min_std, noise_beta=args.noise_beta,
                device=device, generator=gen, init_mu=init_mu, discount=float(args.discount),
                score_fn=sim_score)

        def cmp_gen(k: int) -> torch.Generator:
            # --compare_sim: the same CEM noise draws for both planners at this
            # replan (k=0); k=1 is an independent draw for the noise floors.
            return torch.Generator(device=device).manual_seed(
                int(env_seed) * 1_000_003 + len(replans) * 7 + k)

        t_cem = time.time()
        if bc_only:
            # Baseline: execute the policy head's own actions, no search.
            out = {"best_plan": init_mu, "best_cost": float("nan"), "history": []}
        elif random_only:
            # Floor of the normalized score: uniform actions in [-1, 1].
            rand_plan = torch.rand((H_plan, act_dim), device=device, generator=random_gen) * 2.0 - 1.0
            out = {"best_plan": rand_plan, "best_cost": float("nan"), "history": []}
        else:
            out = wm_cem(cmp_gen(0) if sim is not None else rng)
        dt_cem = time.time() - t_cem
        plan = out["best_plan"]                               # (H_plan, act_dim)

        # --compare_sim: plan the same decision under the true model. Same start
        # state, same objective (--cost reward vs the true reward), same warm
        # start and the same CEM noise draws, so the two plans differ only
        # through the model they were planned under.
        cmp: Optional[dict] = None
        out_sim = None
        if sim is not None:
            t_sim = time.time()
            out_sim = sim_cem(cmp_gen(0))
            cmp = {"exec": str(args.exec), "sim_seconds": time.time() - t_sim,
                   "sim_best_return": -float(out_sim["best_cost"]),
                   "wm_best_return": -float(out["best_cost"])}
            cmp.update(plan_distance(out["best_plan"], out_sim["best_plan"], K_exec,
                                     act_mask_1d, real_dim, mu_std=(out, out_sim)))
            # Planner noise floors: the same planner twice with independent draws.
            n_rep = len(replans)
            # Extra plans whose true return is logged below: the floor reruns
            # (regret floor = the planner's own seed-to-seed return spread) and
            # the BC warm start (what CEM starts from).
            extra_plans = {"bc": init_mu} if init_mu is not None else {}
            if int(args.sim_floor_every) > 0 and n_rep % int(args.sim_floor_every) == 0:
                o2 = sim_cem(cmp_gen(1))
                cmp["floor_sim"] = plan_distance(out_sim["best_plan"], o2["best_plan"], K_exec,
                                                 act_mask_1d, real_dim, mu_std=(out_sim, o2))
                extra_plans["sim2"] = o2["best_plan"]
            if int(args.wm_floor_every) > 0 and n_rep % int(args.wm_floor_every) == 0:
                o2 = wm_cem(cmp_gen(1))
                cmp["floor_wm"] = plan_distance(out["best_plan"], o2["best_plan"], K_exec,
                                                act_mask_1d, real_dim, mu_std=(out, o2))
                extra_plans["wm2"] = o2["best_plan"]
            if args.exec == "sim":
                plan = out_sim["best_plan"]

        # One extra rollout of the plan that will be executed: splits the cost
        # into its parts for the log, and supplies the video's prediction row.
        # Under --policy bc this is the only scoring done, and it is a fresh
        # stochastic rollout, so it is an unbiased estimate of what the model
        # expects — unlike the CEM cost, which is the max over sampled plans and
        # therefore optimistic.
        wm_pred: Optional[np.ndarray] = None
        pred_return = float("nan")
        goal_dist = float("nan")
        goal_term = float("nan")
        value_term = float("nan")
        log_unc = bool(args.log_unc)
        unc: Optional[dict] = None
        r_hat_seq: Optional[torch.Tensor] = None
        t_diag = time.time()
        if compare or rew_head is not None or z_goal is not None or log_unc:
            plan16 = torch.zeros((1, H_plan, 16), device=device)
            plan16[0, :, :act_dim] = plan
            plan16 = plan16 * act_mask_1d.view(1, 1, 16)
            want_h = rew_head is not None
            rolled = imagine_batch(
                dyn, z_start=z_ctx.unsqueeze(0), action_plan=plan16, horizon=H_plan,
                sched=sched, k_max=k_max, act_mask_1d=act_mask_1d,
                lang_emb=lang_emb, use_amp=use_amp, return_h=want_h,
                past_actions=a_ctx.unsqueeze(0), max_ctx=int(args.max_ctx), tau_ctx=float(args.tau_ctx),
                return_unc=log_unc, unc_seeds=int(args.unc_seeds),
            )
            if log_unc:
                z_pred, h_pred, uf_seq, us_seq = rolled
            else:
                z_pred, h_pred = rolled if want_h else (rolled, None)
            if log_unc:
                # Per imagined step t (the latent produced by plan[t]), all on
                # this one rollout:
                #   u_f    flow instability of the denoising run
                #   u_s    paper's inter-seed variance: unc_seeds samples of the
                #          next latent from the same context and action
                #   u_r    tokenizer round-trip residual
                #   motion RMS(z_{t+1} - z_t), the normalizer
                # All raw; *_norm = u / motion is left to the analysis.
                unc = {
                    "n_seeds": int(args.unc_seeds),
                    "u_f": _tolist(uf_seq[0]),
                    "u_s": _tolist(None if us_seq is None else us_seq[0]),
                    "u_r": _tolist(roundtrip_residual(
                        encoder, decoder, z_pred[0, 1:], info=info, use_amp=use_amp)),
                    "motion": _tolist(_rms(z_pred[0, 1:] - z_pred[0, :-1])),
                }
            if h_pred is not None:
                r_hat_seq = predict_rewards(rew_head, h_pred, use_amp=use_amp)[0]  # (H,)
            if h_pred is not None:
                # Reward-only sum, kept separate from the terminal value so it
                # stays directly comparable to the executed return (and to the
                # reward_terminal arm) regardless of --cost.
                pred_return = -float(reward_cost(
                    rew_head, h_pred, use_amp=use_amp, discount=float(args.discount)).item())
                if value_head is not None:
                    # gamma^H * V(h_{H-1}): the learned stand-in for the
                    # latent-distance goal term. Sign matches goal_term (both
                    # are added to the cost), so it prints as -gamma^H V.
                    with torch.autocast(device_type=device.type,
                                        enabled=(use_amp and device.type == "cuda")):
                        v_H = float(value_head.predict(h_pred)[:, -1].float().item())
                    value_term = -(float(args.discount) ** h_pred.shape[1]) * v_H
            if z_goal is not None:
                scale = goal_scale(z_ctx, z_goal)
                goal_dist = float(latent_cost(z_pred, z_goal, kind="terminal").item()) / scale
                goal_term = float(args.goal_weight) * goal_dist
            if compare:
                # Decode the real context + prediction in one pass and keep the
                # prediction: the decoder's time attention is causal, so
                # decoding the prediction alone changes it (as run_dynamics_eval
                # and hallucination_labels.py decode context + rollout).
                wm_pred = decode(torch.cat([z_ctx, z_pred[0, 1:1 + K_exec]], dim=0)
                                 )[z_ctx.shape[0]:]           # (K_exec, 3, H, W) uint8
        dt_diag = time.time() - t_diag
        if step == 0:
            render_probe("after diag + wm_pred decode")

        if cmp is not None:
            # Both plans scored by both models, from the replan's start state,
            # over the executed K steps and the full horizon H.
            gam = float(args.discount)
            dsum = lambda r, n: float(sum(gam ** t * float(v) for t, v in enumerate(list(r)[:n])))
            plans = {"wm": out["best_plan"], "sim": out_sim["best_plan"]}
            env_plans = {k: to_env_actions(v[None], act_mask_1d, real_dim)[0] for k, v in plans.items()}
            env_extra = {k: to_env_actions(v[None], act_mask_1d, real_dim)[0]
                         for k, v in extra_plans.items()}
            for k, a in {**env_plans, **env_extra}.items():
                cmp[f"Rsim_K_{k}"] = float(sim.returns(a[None, :K_exec], discount=gam)[0])
                cmp[f"Rsim_H_{k}"] = float(sim.returns(a[None], discount=gam)[0])
            if value_head is not None:
                # True-model objective of every logged plan after K and after H
                # steps: J = R + gamma^t V(true state), V as in sim_value_score.
                # J_K is what an MPC loop that replans after K steps gets.
                names = list(plans) + list(extra_plans)
                stack = torch.stack([{**plans, **extra_plans}[k] for k in names])
                for tag, T in (("K", K_exec), ("H", H_plan)):
                    _R, V, J = sim_value_score(stack, T)
                    for k, v, j in zip(names, V, J):
                        cmp[f"Vsim_{tag}_{k}"] = float(v)
                        cmp[f"Jsim_{tag}_{k}"] = float(j)
            ex = str(args.exec)
            other = "sim" if ex == "wm" else "wm"
            if r_hat_seq is not None:                       # executed plan: the diag rollout
                cmp[f"Rwm_K_{ex}"] = dsum(r_hat_seq.tolist(), K_exec)
                cmp[f"Rwm_H_{ex}"] = dsum(r_hat_seq.tolist(), H_plan)
            if value_head is not None and h_pred is not None:
                with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                    v_seq = value_head.predict(h_pred)[0].float()
                cmp[f"Vwm_K_{ex}"] = float(v_seq[K_exec - 1])
                cmp[f"Vwm_H_{ex}"] = float(v_seq[H_plan - 1])
            # The other plan: one fresh WM rollout (rewards + its K predicted
            # frames) and the frames the simulator actually produces for it.
            p16 = torch.zeros((1, H_plan, 16), device=device)
            p16[0, :, :act_dim] = plans[other]
            p16 = p16 * act_mask_1d.view(1, 1, 16)
            z_o, h_o, uf_o, _ = imagine_batch(
                dyn, z_start=z_ctx.unsqueeze(0), action_plan=p16, horizon=H_plan,
                sched=sched, k_max=k_max, act_mask_1d=act_mask_1d,
                lang_emb=lang_emb, use_amp=use_amp, return_h=True,
                past_actions=a_ctx.unsqueeze(0), max_ctx=int(args.max_ctx), tau_ctx=float(args.tau_ctx),
                return_unc=True, unc_seeds=1,
            )
            # u_f of the other plan's WM rollout (free from the sampler); the
            # executed plan's is unc["u_f"] under --log_unc.
            cmp[f"uf_{other}"] = _tolist(uf_o[0])
            r_o = predict_rewards(rew_head, h_o, use_amp=use_amp)[0].tolist()
            cmp[f"Rwm_K_{other}"] = dsum(r_o, K_exec)
            cmp[f"Rwm_H_{other}"] = dsum(r_o, H_plan)
            if value_head is not None:
                with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                    v_seq = value_head.predict(h_o)[0].float()
                cmp[f"Vwm_K_{other}"] = float(v_seq[K_exec - 1])
                cmp[f"Vwm_H_{other}"] = float(v_seq[H_plan - 1])
            for k in ("wm", "sim"):
                for tag, T in (("K", K_exec), ("H", H_plan)):
                    if f"Vwm_{tag}_{k}" in cmp and f"Rwm_{tag}_{k}" in cmp:
                        cmp[f"Jwm_{tag}_{k}"] = cmp[f"Rwm_{tag}_{k}"] + gam ** T * cmp[f"Vwm_{tag}_{k}"]
            wm_o = decode(torch.cat([z_ctx, z_o[0, 1:1 + K_exec]], dim=0))[z_ctx.shape[0]:]
            frames_o, rew_o = sim.rollout_frames(env_plans[other][:K_exec], render_fn)
            cmp[f"hausdorff_{other}"] = [
                mask_hausdorff(np.transpose(f, (1, 2, 0)), np.transpose(w, (1, 2, 0)), str(args.task))
                for f, w in zip(frames_o, wm_o)]
            if int(args.pop_truth) > 0 and "pop_samples" in out:
                # Ground truth for part of the CEM population: N samples evenly
                # spaced in u_f rank (all iterations pooled), each replayed in the
                # simulator (true return, frames) and in one fresh WM rollout
                # (predicted frames -> Hausdorff; u_f, u_s over unc_seeds, u_r,
                # motion, predicted rewards per step), like the non-executed plan
                # above. Tests whether the predictors rank plans within a state and
                # whether every candidate hallucinates.
                t_pt = time.time()
                pp = out["pop"]
                where = [(i, j) for i, d in enumerate(pp["iters"]) for j in range(len(d["uf_H"]))]
                u_all = np.array([pp["iters"][i]["uf_H"][j] for i, j in where])
                order = np.argsort(u_all, kind="stable")
                n_pt = min(int(args.pop_truth), len(order))
                pick = [where[order[q]] for q in np.unique(np.linspace(0, len(order) - 1, n_pt).round().astype(int))]
                cand = torch.stack([out["pop_samples"][i][j] for i, j in pick])      # (N, H, act_dim)
                c16 = torch.zeros((len(pick), H_plan, 16), device=device)
                c16[:, :, :act_dim] = cand
                c16 = c16 * act_mask_1d.view(1, 1, 16)
                z_c, h_c, uf_c, us_c = imagine_batch(
                    dyn, z_start=z_ctx.unsqueeze(0).expand(len(pick), *z_ctx.shape).contiguous(),
                    action_plan=c16, horizon=H_plan, sched=sched, k_max=k_max,
                    act_mask_1d=act_mask_1d, lang_emb=lang_emb, use_amp=use_amp, return_h=True,
                    past_actions=a_ctx.unsqueeze(0).expand(len(pick), *a_ctx.shape).contiguous(),
                    max_ctx=int(args.max_ctx), tau_ctx=float(args.tau_ctx),
                    return_unc=True, unc_seeds=max(2, int(args.unc_seeds)),
                )
                r_c = predict_rewards(rew_head, h_c, use_amp=use_amp)              # (N, H)
                mot_c = (z_c[:, 1:] - z_c[:, :-1]).float().pow(2).flatten(2).mean(2).sqrt()  # (N, H)
                env_c = to_env_actions(cand, act_mask_1d, real_dim)
                Rsim_H_c = sim.returns(env_c, discount=gam)
                pt = []
                for n, (i, j) in enumerate(pick):
                    wm_n = decode(torch.cat([z_ctx, z_c[n, 1:1 + K_exec]], dim=0))[z_ctx.shape[0]:]
                    fr_n, rw_n = sim.rollout_frames(env_c[n, :K_exec], render_fn)
                    pt.append({
                        "iter": int(i), "idx": int(j),
                        "cost": float(pp["iters"][i]["cost"][j]), "uf_H_pop": float(pp["iters"][i]["uf_H"][j]),
                        "elite": int(pp["iters"][i]["elite"][j]),
                        "chosen": int((i, j) == (pp["chosen_iter"], pp["chosen_idx"])),
                        "uf": _tolist(uf_c[n]), "us": _tolist(us_c[n]), "motion": _tolist(mot_c[n]),
                        "ur": _tolist(roundtrip_residual(encoder, decoder, z_c[n, 1:],
                                                         info=info, use_amp=use_amp)),
                        "r_pred": _tolist(r_c[n]),
                        "Rwm_K": dsum(r_c[n].tolist(), K_exec), "Rwm_H": dsum(r_c[n].tolist(), H_plan),
                        "Rsim_K": dsum(rw_n, K_exec), "Rsim_H": float(Rsim_H_c[n]), "r_sim": list(map(float, rw_n)),
                        "hausdorff": [mask_hausdorff(np.transpose(f, (1, 2, 0)), np.transpose(w, (1, 2, 0)),
                                                     str(args.task)) for f, w in zip(fr_n, wm_n)],
                    })
                cmp["pop_truth"] = pt
                cmp["pop_truth_seconds"] = time.time() - t_pt
            cmp["regret_K"] = cmp["Rsim_K_sim"] - cmp["Rsim_K_wm"]
            cmp["regret_H"] = cmp["Rsim_H_sim"] - cmp["Rsim_H_wm"]
            if "Jsim_K_sim" in cmp:
                cmp["regret_J_K"] = cmp["Jsim_K_sim"] - cmp["Jsim_K_wm"]
                cmp["regret_J_H"] = cmp["Jsim_H_sim"] - cmp["Jsim_H_wm"]

        r_chunk = 0.0
        n_hist_before = len(z_hist)
        floor_frame = frames_u8[-1]            # the paper's baseline: copy this frame forward
        halluc = ({"mse_wm": [], "mse_floor": [], "psnr_wm": [], "psnr_floor": [], "hausdorff": []}
                  if (args.halluc and wm_pred is not None) else None)
        # env.step includes the EGL render (pixels wrapper); timed separately
        # from encode/video to localize slow-render episodes.
        dt_env = 0.0
        t_exec = time.time()
        for k in range(K_exec):
            a16 = torch.zeros(16, device=device)
            a16[:act_dim] = plan[k]
            a16 = a16 * act_mask_1d
            a_np = a16.detach().cpu().numpy().astype(np.float32)

            if probe_on and step < 3:
                # Split one env.step into physics vs render, with the exact action.
                phys = getattr(getattr(pix.env, "env", None), "physics", None)
                print(f"[probe] step {step} action={a_np[:real_dim].tolist()} "
                      f"dtype={a_np.dtype} finite={bool(np.isfinite(a_np).all())}", flush=True)
            t_env = time.time()
            obs_dict, reward, terminated, truncated, step_info = env.env.step(a_np[:real_dim])
            dt_step = time.time() - t_env
            dt_env += dt_step
            if 0 < float(args.render_stall_s) < dt_step:
                # A healthy step (2 physics substeps + one 224px render) is
                # 2-15 ms. Some Snellius GPUs have a stalling graphics engine:
                # every render then waits ~18 s, returns a stale frame, and the
                # process eventually aborts in mjr_readPixels. The episode data is
                # invalid from that point, so stop now instead of burning the
                # job; eval_planner.sbatch records the node/GPU on exit code 75.
                print(f"[render-stall] env.step took {dt_step:.1f}s at ep {ep_index} step {step} "
                      f"on {os.uname().nodename}; GPU graphics engine stalled, aborting "
                      f"(exit 75)", flush=True)
                os._exit(75)   # skip EGL/GL teardown, which can hang on a stalled GPU
            if probe_on and step < 3:
                print(f"[probe] step {step} env.step total {1e3 * (time.time() - t_env):9.1f} ms",
                      flush=True)
                if phys is not None:
                    print(f"[probe] step {step} qpos={np.round(phys.data.qpos, 4).tolist()} "
                          f"qvel={np.round(phys.data.qvel, 4).tolist()} "
                          f"ctrl={np.round(phys.data.ctrl, 4).tolist()}", flush=True)
                render_probe(f"after env step {step}")
            frame = env._extract_rgb(obs_dict)
            done = bool(terminated or truncated)
            step += 1

            rewards.append(float(reward))
            r_chunk += float(reward)
            if isinstance(step_info, dict):
                success = max(success, float(step_info.get("success", 0.0)))
            frames_u8.append(frame)
            z_hist.append(encode(frame))
            a_hist.append(a16)
            if record:
                vid.append(compose_frame(frame, None if wm_pred is None else wm_pred[k]))
            if halluc is not None:
                for key, (a, b) in (("wm", (wm_pred[k], frame)), ("floor", (floor_frame, frame))):
                    mse, psnr = _psnr_u8(a, b)
                    halluc["mse_" + key].append(mse)
                    halluc["psnr_" + key].append(psnr)
                # Foreground-mask Hausdorff (px) on the raw frames; plan_halluc.py
                # thresholds its chunk mean per task.
                halluc["hausdorff"].append(mask_hausdorff(
                    np.transpose(frame, (1, 2, 0)), np.transpose(wm_pred[k], (1, 2, 0)),
                    str(args.task)))
            if done or step >= args.max_steps:
                break
        dt_rest = time.time() - t_exec - dt_env                  # encode + video
        if halluc is not None:
            t_h = time.time()
            halluc["shuf_mse_true"], halluc["shuf_mse_rand"] = shuffle_mse_executed(
                dyn, z_hist, a_hist, list(range(n_hist_before, len(z_hist))),
                act_dim=act_dim, act_mask_1d=act_mask_1d, k_max=k_max, lang_emb=lang_emb,
                max_ctx=int(args.max_ctx), use_amp=use_amp, gen=halluc_gen,
                seed=int(env_seed))
            halluc["seconds"] = time.time() - t_h

        if cmp is not None and halluc is not None:
            # The executed plan's label comes from the real frames.
            cmp["hausdorff_" + str(args.exec)] = list(halluc["hausdorff"])

        if unc is not None:
            # Ground truth for the steps actually executed: latent error against
            # the encoded real frame, and reward-head error against the env.
            n_exec = len(z_hist) - n_hist_before
            z_real = torch.stack(z_hist[n_hist_before:], dim=0)       # (n_exec, S, D)
            unc["e_z"] = _tolist(_rms(z_pred[0, 1:1 + n_exec] - z_real))
            unc["motion_real"] = _tolist(_rms(
                z_real - torch.stack(z_hist[n_hist_before - 1:-1], dim=0)))
            if r_hat_seq is not None:
                unc["r_pred"] = _tolist(r_hat_seq[:n_exec])
                unc["e_r"] = [p_ - r_ for p_, r_ in zip(unc["r_pred"], rewards[-n_exec:])]

        replans.append({
            "step": step,
            "plan_horizon": int(H_plan),
            "exec_K": int(K_exec),
            "plan_cost": float(out["best_cost"]),
            "pred_return": pred_return,
            "goal_dist_norm": goal_dist,
            "goal_term": goal_term,
            "value_term": value_term,
            "reward_executed": r_chunk,
            "bc_seconds": dt_bc,
            "cem_seconds": dt_cem,
            "diag_seconds": dt_diag,
            "env_seconds": dt_env,
            "encode_seconds": dt_rest,
            **({"unc": unc} if unc is not None else {}),
            **({"halluc": halluc} if halluc is not None else {}),
            **({"compare": cmp} if cmp is not None else {}),
            **({"cem_pop": out["pop"]} if "pop" in out else {}),
            **({"unc_lam": out["unc_lam"], "best_obj": out["best_obj"]} if "unc_lam" in out else {}),
        })
        # cost = -pred_return + goal_term, so the two terms show which one the
        # planner is actually optimizing. Under --policy bc there is no CEM cost.
        cost_str = (f"cost=  ({args.policy})  " if (bc_only or random_only)
                    else f"cost={out['best_cost']:+.3f}")
        term_str = (f"value {value_term:+.2f}" if value_head is not None
                    else f"goal {goal_term:+.2f} (d={goal_dist:.3f})")
        print(f"  [ep {ep_index} step {step:4d}] {cost_str} "
              f"= -pred_ret {pred_return:+.2f} + {term_str}"
              f" | r_exec={r_chunk:+.3f} return={sum(rewards):+.3f} "
              f"(bc {dt_bc:.1f}s, cem {dt_cem:.1f}s, diag {dt_diag:.1f}s, "
              f"env {dt_env:.1f}s, enc {dt_rest:.1f}s)", flush=True)
        if cmp is not None:
            hd = {k: _nanmean(cmp.get("hausdorff_" + k, [])) for k in ("wm", "sim")}
            print(f"      cmp: d_exec {cmp['d_exec']:.3f} (floor sim "
                  f"{cmp.get('floor_sim', {}).get('d_exec', float('nan')):.3f}) w2 {cmp.get('w2', float('nan')):.3f} "
                  f"| Rsim_K wm {cmp['Rsim_K_wm']:.2f} sim {cmp['Rsim_K_sim']:.2f} (regret {cmp['regret_K']:+.2f}) "
                  f"| Rwm_K wm {cmp.get('Rwm_K_wm', float('nan')):.2f} sim {cmp.get('Rwm_K_sim', float('nan')):.2f} "
                  f"| HD wm {hd['wm']:.1f} sim {hd['sim']:.1f} | sim {cmp['sim_seconds']:.1f}s", flush=True)
            if "pop_truth" in cmp:
                hd_pt = [_nanmean(c["hausdorff"]) for c in cmp["pop_truth"]]
                print(f"      pop_truth: {len(hd_pt)} plans HD {np.min(hd_pt):.1f}-{np.max(hd_pt):.1f} "
                      f"| Rsim_K {min(c['Rsim_K'] for c in cmp['pop_truth']):.2f}-"
                      f"{max(c['Rsim_K'] for c in cmp['pop_truth']):.2f} "
                      f"| {cmp['pop_truth_seconds']:.1f}s", flush=True)
        if "pop" in out:
            pp = out["pop"]
            ku = {"u_f": "uf_H", "u_r": "ur_H", "u_f_norm": "ufn_H", "u_r_norm": "urn_H",
                  "u_f_res": "uf_H", "u_r_res": "ur_H"}[pp["unc_kind"]]
            u_all = [u for d in pp["iters"] for u in d[ku]]
            u_ch = pp["iters"][pp["chosen_iter"]][ku][pp["chosen_idx"]]
            print(f"      cem_pop: lam {pp['lam']:.3g} | {pp['unc_kind']}(H) chosen {u_ch:.4f} "
                  f"pop mean {np.mean(u_all):.4f} (chosen pct {100 * np.mean(np.array(u_all) < u_ch):.0f})",
                  flush=True)
        if unc is not None:
            print(f"      unc: u_r {_nanmean(unc['u_r']):.4f} u_f {_nanmean(unc['u_f']):.4f} "
                  f"u_s {_nanmean(unc['u_s']):.4f} motion {_nanmean(unc['motion']):.4f} "
                  f"| e_z {_nanmean(unc['e_z']):.4f} "
                  f"|e_r| {_nanmean([abs(v) for v in unc.get('e_r', [])]):.3f}", flush=True)

    row = {
        "task": str(args.task),
        "policy": str(args.policy),
        "episode": int(ep_index),
        "steps": int(step),
        "return": float(sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "success": float(success),
        "cost_kind": str(args.cost),
        "bc_head": (f"{args.bc_head_type}:" + str(args.bc_readout)) if str(args.policy_ckpt) else "released",
        "goal_weight": float(args.goal_weight),
        "horizon": int(args.horizon),
        "replan_every": int(args.replan_every),
        "cem_samples": int(args.cem_samples),
        "cem_iters": int(args.cem_iters),
        "cem_n_rollouts": int(args.cem_n_rollouts),
        "warm_start": str(args.warm_start),
        "ctx_frames": int(args.ctx_frames),
        "max_ctx": int(args.max_ctx),
        "tau_ctx": float(args.tau_ctx),
        "eval_d": float(args.eval_d),
        "unc_weight": float(args.unc_weight),
        "unc_kind": str(args.unc_kind),
        "unc_mode": str(args.unc_mode),
        "unc_thr": float(args.unc_thr),
        "unc_thr_pct": float(args.unc_thr_pct),
        "unc_rate": float(args.unc_rate),
        "env_seed": int(env_seed),
        "n_replans": len(replans),
        "pred_return_mean": _nanmean([r["pred_return"] for r in replans]),
        # Horizon-corrected predicted return for the WHOLE episode. Each replan
        # predicts over plan_horizon steps but only exec_K of them are executed
        # before the next replan supersedes the rest, so the chunks are scaled
        # by exec_K/plan_horizon before summing. Directly comparable to
        # "return"; under --policy bc the weights are 1 (H_plan == K_exec).
        "pred_return_episode": float(np.nansum([
            r["pred_return"] * (r["exec_K"] / max(1, r["plan_horizon"]))
            for r in replans])) if replans else float("nan"),
        "goal_dist_mean": _nanmean([r["goal_dist_norm"] for r in replans]),
        "value_term_mean": _nanmean([r["value_term"] for r in replans]),
        "plan_seconds_mean": float(np.mean([r["cem_seconds"] for r in replans])) if replans else float("nan"),
        "episode_seconds": float(time.time() - t_ep),
        "env_seconds": float(sum(r["env_seconds"] for r in replans)),
    }
    if args.log_unc:
        def _pool(key, f=lambda v: v):
            return _nanmean([f(v) for r in replans for v in r.get("unc", {}).get(key, [])])
        row.update({
            "unc_seeds": int(args.unc_seeds),
            "u_r_mean": _pool("u_r"),
            "u_f_mean": _pool("u_f"),
            "u_s_mean": _pool("u_s"),
            "motion_mean": _pool("motion"),
            "e_z_mean": _pool("e_z"),
            "e_r_abs_mean": _pool("e_r", abs),
        })
    cmps = [r["compare"] for r in replans if "compare" in r]
    if cmps:
        _m = lambda f: _nanmean([float(f(c)) for c in cmps])
        row.update({
            "exec": str(args.exec),
            "cmp_d_exec_mean": _m(lambda c: c["d_exec"]),
            "cmp_w2_mean": _m(lambda c: c.get("w2", float("nan"))),
            "cmp_floor_sim_d_exec_mean": _nanmean([float(c["floor_sim"]["d_exec"]) for c in cmps if "floor_sim" in c]),
            "cmp_floor_wm_d_exec_mean": _nanmean([float(c["floor_wm"]["d_exec"]) for c in cmps if "floor_wm" in c]),
            "cmp_regret_K_mean": _m(lambda c: c["regret_K"]),
            "cmp_regret_H_mean": _m(lambda c: c["regret_H"]),
            "cmp_sim_seconds_mean": _m(lambda c: c["sim_seconds"]),
        })
        print(f"  [compare] d_exec {row['cmp_d_exec_mean']:.3f} (floors sim "
              f"{row['cmp_floor_sim_d_exec_mean']:.3f}, wm {row['cmp_floor_wm_d_exec_mean']:.3f}) | "
              f"regret K {row['cmp_regret_K_mean']:+.3f} H {row['cmp_regret_H_mean']:+.3f}", flush=True)
    chunks = halluc_chunks(replans)
    if chunks:
        row.update({
            "halluc_replans": len(chunks),
            "frac_scene_divergent": float(np.mean([c["scene_divergent"] for c in chunks])),
            "frac_action_ignored": float(np.mean([c["action_ignored"] for c in chunks])),
            "delta_psnr_median": float(np.nanmedian([c["delta_psnr"] for c in chunks])),
            "shuffle_ratio_median": float(np.nanmedian([c["shuffle_ratio"] for c in chunks])),
        })
        print(f"  [halluc] divergent replans {sum(c['scene_divergent'] for c in chunks)}/"
              f"{len(chunks)}, action-ignored replans {sum(c['action_ignored'] for c in chunks)}/"
              f"{len(chunks)}, dPSNR median {row['delta_psnr_median']:+.2f} dB, "
              f"shuffle ratio median {row['shuffle_ratio_median']:.2f}", flush=True)

    if out_dir:
        base = f"ep{ep_index:03d}_seed{env_seed}"
        save_strip(os.path.join(out_dir, base + ".png"), frames_u8,
                   max_frames=int(args.n_viz_frames))
        if chunks and vid:
            vid = add_halluc_panel(vid, replans)
        for path in save_video(os.path.join(out_dir, base), vid,
                               fps=int(args.video_fps), kind=str(args.video)):
            print(f"  [video] {path}")
        row["video_frames"] = len(vid)
        detail = dict(row)
        detail["rewards"] = rewards
        detail["replans"] = replans
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, base + ".json")
        with open(json_path + ".tmp", "w") as f:
            json.dump(detail, f, indent=2)
        os.replace(json_path + ".tmp", json_path)
        if args.log_unc:
            # Diagnostic only: a plotting failure must not kill the eval.
            try:
                from plot_unc_episode import plot_episode
                unc_png = os.path.join(out_dir, base + "_unc.png")
                plot_episode(detail, unc_png)
                print(f"  [unc plot] {unc_png}")
            except Exception as e:
                print(f"  [unc plot] skipped: {type(e).__name__}: {e}")

    return row


# -------------------- main -------------------- #

def evaluate(args):
    from env_wrapper import EnvCollector  # lazy: only this script needs a simulator

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    py_rng = np.random.default_rng(args.seed)
    torch_rng = torch.Generator(device=device).manual_seed(args.seed)

    encoder, decoder, dyn, rew_head, policy_head, sched, info = build_models(args, device)
    use_amp = bool(args.amp) and device.type == "cuda"

    if needs_reward_head(args.cost) and rew_head is None:
        raise SystemExit(
            f"--cost={args.cost!r} requires a reward head in the dynamics ckpt. "
            f"Loaded ckpt {args.dynamics_ckpt!r} has none."
        )
    if str(args.policy_ckpt):
        from heads_rl import CategoricalPolicyAdapter, load_policy_head_from_ckpt
        policy_head = CategoricalPolicyAdapter(
            load_policy_head_from_ckpt(args.policy_ckpt, device),
            readout=str(args.bc_readout),
        ).to(device).eval().requires_grad_(False)
        args.bc_head_type = policy_head.head.policy_type
        print(f"[mpc] loaded {args.bc_head_type} BC head from {args.policy_ckpt} "
              f"(readout={args.bc_readout})")

    value_head = None
    if needs_value_head(args.cost):
        if not str(args.value_ckpt):
            raise SystemExit(
                f"--cost={args.cost!r} requires --value_ckpt pointing at a value head "
                f"from train_heads_offline.py."
            )
        from heads_rl import load_value_head_from_ckpt
        value_head = load_value_head_from_ckpt(args.value_ckpt, device)
        print(f"[mpc] loaded value head from {args.value_ckpt}")

    if (args.warm_start == "bc" or args.policy == "bc") and policy_head is None:
        raise SystemExit(
            f"--warm_start=bc / --policy=bc require a policy head in the dynamics "
            f"ckpt. Loaded ckpt {args.dynamics_ckpt!r} has none; use --warm_start none."
        )

    act_dim, act_mask_1d = load_act_meta(args.tasks_json, args.task, device=device)
    if act_dim == 0:
        raise SystemExit(f"task {args.task} has act_dim=0; nothing to plan")
    lang_emb = load_lang_emb(args.tasks_json, args.task, int(args.lang_dim), device)
    z_goal = load_goal_latent(args, encoder, info, device, use_amp, py_rng)

    # Every run gets its own timestamped directory, so repeated runs of the same
    # task never overwrite each other and a run's config travels with its results.
    suffix = "_bc" if str(args.policy) == "bc" else ""
    run_name = args.run_name or f"{time.strftime('%Y%m%d-%H%M%S')}_{args.task}{suffix}"
    out_dir = os.path.join(args.viz_dir, run_name) if args.viz_dir else ""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)

    max_steps_arg = int(args.max_steps)
    rows = []
    t0 = time.time()
    for ep_index in range(int(args.n_episodes)):
        # A fresh env per episode with its own seed: dm_control draws the initial
        # state from that seed, so episode i starts from a different but
        # reproducible state, and two runs with the same --env_seed are
        # comparable episode by episode.
        env_seed = int(args.env_seed if args.env_seed >= 0 else args.seed) + ep_index
        env = EnvCollector(args.task, img_size=int(info["H"]), action_dim=16, seed=env_seed)
        args.max_steps = max_steps_arg if max_steps_arg > 0 else int(env._max_episode_steps or 500)
        if ep_index == 0:
            head = (f"policy={args.policy} (no search), K={args.replan_every}" if args.policy != "cem"
                    else f"policy=cem cost={args.cost} H={args.horizon} K={args.replan_every}")
            head += (f" bc_head={args.bc_head_type}({args.bc_readout})" if str(args.policy_ckpt)
                     else " bc_head=released(deterministic)")
            print(f"[mpc] task={args.task} act_dim={act_dim} real_act_dim={env._real_action_dim} "
                  f"max_steps={args.max_steps} {head}")
            print(f"[mpc] run dir: {out_dir or '(no artifacts)'}")
        try:
            rows.append(run_episode(
                args=args, env=env, encoder=encoder, decoder=decoder, dyn=dyn, rew_head=rew_head,
                policy_head=policy_head, sched=sched, info=info, device=device,
                use_amp=use_amp, act_dim=act_dim, act_mask_1d=act_mask_1d,
                lang_emb=lang_emb, z_goal=z_goal, rng=torch_rng, ep_index=ep_index,
                env_seed=env_seed, out_dir=out_dir, value_head=value_head,
            ))
        finally:
            env.close()
        r = rows[-1]
        print(f"[ep {ep_index}] seed={env_seed} return={r['return']:+.3f} "
              f"success={r['success']:.0f} steps={r['steps']} ({r['episode_seconds']:.0f}s, "
              f"{r['plan_seconds_mean']:.1f}s/replan)", flush=True)

    if rows:
        rets = np.array([r["return"] for r in rows], dtype=np.float64)
        succ = np.array([r["success"] for r in rows], dtype=np.float64)
        print(f"\n========== {len(rows)} episodes on {args.task} ==========")
        print(f"  return   mean={rets.mean():+.3f}  median={np.median(rets):+.3f}  "
              f"min={rets.min():+.3f}  max={rets.max():+.3f}")
        print(f"  success  rate={succ.mean():.1%}")
        print(f"  total wallclock: {time.time() - t0:.1f}s")

    if rows:
        out_csv = args.output or os.path.join(out_dir or ".", "summary.csv")
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[mpc] wrote {out_csv}")


def main():
    p = argparse.ArgumentParser()

    # task / data (data dirs are only used to pick a goal frame)
    p.add_argument("--task", type=str, default="cup-catch")
    p.add_argument("--data_dir", type=str, default="./data/expert")
    p.add_argument("--frames_dir", type=str, default="./data/expert-shards")
    p.add_argument("--tasks_json", type=str, default="../tasks.json")
    p.add_argument("--shard_size", type=int, default=4096)
    p.add_argument("--lang_dim", type=int, default=512)
    p.add_argument("--goal_ep", type=int, default=-1,
                   help="demo episode whose last frame is the goal (-1 = random)")

    # checkpoints
    p.add_argument("--tokenizer_ckpt", type=str, default="./checkpoints/combined/tokenizer.pt")
    p.add_argument("--dynamics_ckpt", type=str, default="./checkpoints/combined/dynamics.pt")
    p.add_argument("--policy_ckpt", type=str, default="",
                   help="stochastic (categorical or Gaussian) BC head from train_heads_offline.py, used in "
                        "place of the deterministic one in dynamics.pt. Drives both "
                        "--policy bc and the --warm_start bc CEM init.")
    p.add_argument("--bc_readout", type=str, default="sample",
                   choices=["sample", "mean", "argmax"],
                   help="(--policy_ckpt) how to collapse the per-dim action "
                        "distribution into an action (Gaussian: mean == argmax). 'sample' is the default for a "
                        "reason: on cup-catch the sampled readout scored ~0.47 "
                        "closed-loop against ~0.10 -- the random floor -- for both "
                        "mean and argmax. Use the deterministic readouts for "
                        "diagnostics, not as the headline number.")
    p.add_argument("--value_ckpt", type=str, default="",
                   help="(--cost=reward_value) value head from train_heads_offline.py. "
                        "Replaces reward_terminal's latent-distance goal term with a "
                        "learned terminal value, and needs no goal latent. --discount "
                        "must match the gamma the head was fitted with.")

    # rollout / denoising schedule
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--schedule", type=str, default="shortcut", choices=["finest", "shortcut"])
    p.add_argument("--eval_d", type=float, default=0.125)
    p.add_argument("--amp", action="store_true", default=True)

    # cost
    p.add_argument("--cost", type=str, default="reward_terminal", choices=list(ALL_COST_KINDS),
                   help="see plan_cem.py --cost")
    p.add_argument("--goal_weight", type=float, default=0.25,
                   help="(--cost=reward_terminal) weight of the normalized terminal "
                        "latent-distance term; 0 makes it pure predicted return. Set 0 "
                        "for cyclic tasks, where a demo's final frame is not a goal.")
    p.add_argument("--discount", type=float, default=1.0)

    # MPC (defaults follow the paper's collection setting)
    p.add_argument("--policy", type=str, default="cem", choices=["cem", "bc", "random"],
                   help="'cem' searches over action sequences with the world model. "
                        "'bc' skips the search and executes the policy head's own "
                        "actions — the baseline CEM is warm-started from. Use "
                        "--replan_every 1 for true closed-loop BC (each action "
                        "conditioned on the real frame); larger K executes actions "
                        "generated inside the model's imagination. Run dirs get a "
                        "_bc suffix. 'random' executes uniform actions (the floor "
                        "of the normalized score); use it with --warm_start none")
    p.add_argument("--horizon", type=int, default=32, help="plan horizon H")
    p.add_argument("--replan_every", type=int, default=16, help="actions executed per plan (K)")
    p.add_argument("--ctx_frames", type=int, default=4,
                   help="real observations used as rollout context; >1 gives the model "
                        "the velocity information a single frame cannot carry")
    p.add_argument("--max_ctx", type=int, default=24,
                   help="sliding context window inside the rollout, in frames; keeps the "
                        "sequence within the training length (train_dynamics --seq_len). "
                        "0 disables windowing")
    p.add_argument("--compare_sim", action="store_true",
                   help="at every replan also plan under the TRUE model (CEM in the simulator, "
                        "true reward; sim_planner.py) with the same noise draws, and log both "
                        "plans scored by both models, action distances and the hallucination "
                        "label of the non-executed plan. Needs --policy cem --cost reward")
    p.add_argument("--exec", type=str, default="wm", choices=["wm", "sim"],
                   help="--compare_sim: which planner's plan is executed")
    p.add_argument("--sim_floor_every", type=int, default=1,
                   help="--compare_sim: rerun the sim CEM with independent noise every N replans "
                        "(planner noise floor of the action distance; cheap). 0 = off")
    p.add_argument("--wm_floor_every", type=int, default=0,
                   help="--compare_sim: same for the WM CEM (costs a second WM CEM). 0 = off")
    p.add_argument("--tau_ctx", type=float, default=0.1,
                   help="context corruption per imagined step in every planner rollout (BC, "
                        "CEM, diagnostic). 0.1 = the annotated rollouts; 0 (the old hardcoded "
                        "value) hallucinates 8-25 pts more (ctx_ablation.py)")

    # CEM
    p.add_argument("--cem_samples", type=int, default=32, help="population size")
    p.add_argument("--cem_elites", type=int, default=8)
    p.add_argument("--cem_iters", type=int, default=3)
    p.add_argument("--cem_n_rollouts", type=int, default=2, help="rollouts per candidate")
    p.add_argument("--cem_init_std", type=float, default=1.0)
    p.add_argument("--warm_start_init_std", type=float, default=0.3)
    p.add_argument("--cem_min_std", type=float, default=0.05)
    p.add_argument("--rollout_batch", type=int, default=64)
    p.add_argument("--noise_beta", type=float, default=0.0,
                   help="action-noise power spectrum 1/f^beta (2 = Brownian, for "
                        "navigation / smooth-velocity tasks)")
    p.add_argument("--warm_start", type=str, default="bc", choices=["none", "bc"])

    # episodes / outputs
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=0,
                   help="0 = the env's own episode length")
    p.add_argument("--env_seed", type=int, default=-1,
                   help="seed of the first episode's environment; episode i uses "
                        "env_seed+i, which sets its initial state. -1 = use --seed. "
                        "Runs sharing an --env_seed start from identical states and are "
                        "directly comparable")
    p.add_argument("--output", type=str, default="",
                   help="summary CSV path; default writes summary.csv inside the run dir")
    p.add_argument("--viz_dir", type=str, default="./logs/mpc_env_viz",
                   help="parent directory; each run gets a timestamped subdirectory")
    p.add_argument("--run_name", type=str, default="",
                   help="name of the run subdirectory (default: <timestamp>_<task>)")
    p.add_argument("--render_stall_s", type=float, default=5.0,
                   help="abort with exit code 75 if one env.step takes longer than this "
                        "(stalled GPU graphics engine); healthy steps take 2-15 ms. 0 = off")
    p.add_argument("--n_viz_frames", type=int, default=16,
                   help="frames in the summary PNG strip (the video keeps every step)")

    # video
    p.add_argument("--video", type=str, default="mp4", choices=["none", "gif", "mp4", "both"],
                   help="episode recording format (gif needs no VS Code extension but is "
                        "much larger)")
    p.add_argument("--video_fps", type=int, default=15)
    p.add_argument("--video_compare", action="store_true", default=True,
                   help="put the world model's prediction for each executed step beside "
                        "the real frame (reuses the per-replan diagnostic rollout)")
    p.add_argument("--no-video_compare", dest="video_compare", action="store_false")
    p.add_argument("--halluc", action="store_true", default=True,
                   help="per replan: the paper's two hallucination labels (scene divergent, "
                        "action ignored with uniform random actions), logged to the episode "
                        "JSON/CSV and charted under the video. Needs --video_compare")
    p.add_argument("--no-halluc", dest="halluc", action="store_false")
    p.add_argument("--seed", type=int, default=0)

    # hallucination predictors
    p.add_argument("--log_unc", action="store_true",
                   help="log u_r / u_f / u_s / motion per imagined step of the executed "
                        "plan, plus realized latent (e_z) and reward (e_r) error on the "
                        "executed steps, into each replan of the episode JSON")
    p.add_argument("--unc_seeds", type=int, default=4,
                   help="denoising seeds per imagined step for u_s under --log_unc "
                        "(the paper's inter-seed variance; needs >= 2)")

    # uncertainty-aware CEM (plan_cem.cem_plan)
    p.add_argument("--unc_weight", type=float, default=0.0,
                   help="WM CEM selects elites / the plan by cost + lam * u_f (u_f = flow "
                        "instability averaged over the H steps), lam = c * std(cost)/std(u_f) "
                        "of the iteration-0 population. c = this value; 0 = unchanged planner")
    p.add_argument("--unc_kind", type=str, default="u_f", choices=["u_f", "u_r", "u_f_norm", "u_r_norm", "u_f_res", "u_r_res"],
                   help="predictor penalized by --unc_weight. u_r = tokenizer round-trip "
                        "residual of every imagined step of every CEM sample (~+26%% planner "
                        "time); *_norm = per-step u / rollout motion, then averaged; with u_r or "
                        "u_r_norm, --log_cem_pop logs u_f, u_r and both normalized versions; *_res = residual of "
                        "log u on log motion, fitted on the iteration-0 population (soft only)")
    p.add_argument("--unc_mode", type=str, default="soft", choices=["soft", "hard"],
                   help="soft: obj = cost + lam*u (lam = c*std(cost)/std(u)). hard: flag each "
                        "imagined step with u_t > tau; obj = cost + c*std(cost) * 1[flagged "
                        "fraction of the H steps > --unc_rate] (plan_cem.cem_plan)")
    p.add_argument("--unc_thr", type=float, default=0.0,
                   help="--unc_mode hard: absolute per-step threshold tau on u_t (> 0)")
    p.add_argument("--unc_thr_pct", type=float, default=0.0,
                   help="--unc_mode hard, when --unc_thr is 0: tau = this quantile of all per-step "
                        "u of the iteration-0 population (relative to the current state)")
    p.add_argument("--unc_rate", type=float, default=0.5,
                   help="--unc_mode hard: penalize samples whose flagged-step fraction exceeds this")
    p.add_argument("--ur_chunk", type=int, default=256, help="latents per decode/encode call for u_r")
    p.add_argument("--log_cem_pop", action="store_true",
                   help="log every CEM sample's cost, u_f and rollout motion (first K steps and "
                        "all H), elite flags and the chosen sample into each replan (cem_pop)")
    p.add_argument("--pop_truth", type=int, default=0,
                   help="--compare_sim + --log_cem_pop: per replan, replay N CEM samples (evenly "
                        "spaced in u_f rank over all iterations) in the simulator and in a fresh WM "
                        "rollout; logs their true return, Hausdorff per executed step and u_f into "
                        "compare.pop_truth. 0 = off")

    args = p.parse_args()
    if args.pop_truth > 0 and not (args.compare_sim and args.log_cem_pop):
        p.error("--pop_truth needs --compare_sim and --log_cem_pop")
    if args.compare_sim:
        # Same objective on both sides: the reward head vs the true reward. Goal
        # / value terms have no simulator counterpart.
        # reward_value: the simulator side adds the same value head on h
        # teacher-forced from the true frames (sim_value_score).
        if args.policy != "cem" or args.cost not in ("reward", "reward_value"):
            p.error("--compare_sim needs --policy cem --cost reward|reward_value")
        if not (args.halluc and args.video_compare and args.video != "none"):
            p.error("--compare_sim needs the compare video path (--halluc, --video_compare) for the "
                    "per-step hallucination labels")
    evaluate(args)


if __name__ == "__main__":
    main()
