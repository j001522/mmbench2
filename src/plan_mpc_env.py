# plan_mpc_env.py
"""
Closed-loop MPC evaluation: plan in imagination, act in the real environment.

Each control cycle:
  - encode the last --ctx_frames real observations into packed latents;
  - warm-start CEM's mean with a BC rollout of the integrated policy head;
  - run CEM over action sequences, scoring every candidate by rolling it out
    through the world model (predicted return from the reward head, plus a
    terminal term: a latent distance to a goal frame (--cost reward_terminal)
    or a learned value head (--cost reward_value --value_ckpt));
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
from typing import List, Optional

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
    reward_cost,
    _tile_row,
)

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
    the observation after env step i (frame 0 = initial obs)."""
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

    # Video: environment on the left, the model's prediction for the same step
    # on the right. The first video frame has no prediction yet, so it shows the
    # tokenizer round-trip of the initial observation instead.
    vid: List[np.ndarray] = []
    if record:
        wm0 = decode(z_hist[0].unsqueeze(0))[0] if compare else None
        vid.append(compose_frame(frame, wm0))

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
            )

        t_cem = time.time()
        if bc_only:
            # Baseline: execute the policy head's own actions, no search.
            out = {"best_plan": init_mu, "best_cost": float("nan"), "history": []}
        elif random_only:
            # Floor of the normalized score: uniform actions in [-1, 1].
            rand_plan = torch.rand((H_plan, act_dim), device=device, generator=random_gen) * 2.0 - 1.0
            out = {"best_plan": rand_plan, "best_cost": float("nan"), "history": []}
        else:
            out = wm_cem(rng)
        dt_cem = time.time() - t_cem
        plan = out["best_plan"]                               # (H_plan, act_dim)

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
        t_diag = time.time()
        if compare or rew_head is not None or z_goal is not None:
            plan16 = torch.zeros((1, H_plan, 16), device=device)
            plan16[0, :, :act_dim] = plan
            plan16 = plan16 * act_mask_1d.view(1, 1, 16)
            want_h = rew_head is not None
            rolled = imagine_batch(
                dyn, z_start=z_ctx.unsqueeze(0), action_plan=plan16, horizon=H_plan,
                sched=sched, k_max=k_max, act_mask_1d=act_mask_1d,
                lang_emb=lang_emb, use_amp=use_amp, return_h=want_h,
                past_actions=a_ctx.unsqueeze(0), max_ctx=int(args.max_ctx), tau_ctx=float(args.tau_ctx),
            )
            z_pred, h_pred = rolled if want_h else (rolled, None)
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
        r_chunk = 0.0
        # env.step includes the EGL render (pixels wrapper); timed separately
        # from encode/video to localize slow-render episodes.
        dt_env = 0.0
        t_exec = time.time()
        for k in range(K_exec):
            a16 = torch.zeros(16, device=device)
            a16[:act_dim] = plan[k]
            a16 = a16 * act_mask_1d
            a_np = a16.detach().cpu().numpy().astype(np.float32)

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
            if done or step >= args.max_steps:
                break
        dt_rest = time.time() - t_exec - dt_env                  # encode + video
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

    if out_dir:
        base = f"ep{ep_index:03d}_seed{env_seed}"
        save_strip(os.path.join(out_dir, base + ".png"), frames_u8,
                   max_frames=int(args.n_viz_frames))
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
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
