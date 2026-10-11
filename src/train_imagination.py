"""
Dreamer 4 phase 3: imagination training of the policy (PMPO) and value heads
inside the FROZEN world model.

Each step:
  1. draw real context windows from the dataset (ctx_len frames + led-to actions),
  2. unroll the world model closed-loop for `horizon` steps, sampling actions
     from the trainable policy (`imagine_policy_rollout`),
  3. annotate the imagined states with the frozen reward head and the value head,
  4. TD(lambda) returns -> value loss (Eq. 10), advantage signs -> PMPO with a
     reverse KL to the frozen BC prior (Eq. 11).

Only the policy and value heads train; tokenizer, dynamics and reward head are
frozen and asserted so. Starting point (deviation from the paper, which starts
the value head fresh): both heads are initialized from a `train_heads_offline.py`
checkpoint -- its Gaussian BC head becomes the prior AND the policy init, its
value head (fit on real TD(lambda) targets) warm-starts the value. Use
`--value_init fresh` for the paper's setup.

Checkpoints use the `train_heads_offline.py` format, so
`plan_mpc_env.py --policy bc --policy_ckpt <out>` evaluates the trained policy
closed-loop without changes.

Run from inside src/, e.g.:

    python train_imagination.py --task cartpole-swingup \
        --heads_ckpt ../../logs/heads/cartpoleswingup_27849014.pt \
        --data_dirs ./data/mixed-large --frame_dirs ./data/mixed-large-shards \
        --out ../../logs/imagination/cartpole.pt
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import time
from typing import Dict, Optional

import torch
import wandb
from torch.utils.data import DataLoader

from heads_rl import ValueHeadMTP, load_policy_head_from_ckpt, load_value_head_from_ckpt
from interactive import (
    load_dynamics_from_ckpt,
    load_tokenizer_from_ckpt,
    make_tau_schedule,
    sample_one_timestep_packed,
)
from model import pack_bottleneck_to_spatial, temporal_patchify
from plan_cem import predict_rewards
from rl_algorithms import (
    GAMMA, LAMBDA, PMPO_ALPHA, PMPO_BETA,
    compute_advantages, compute_lambda_returns, pmpo_policy_loss, value_loss,
)
from train_dynamics import parse_batch
from wm_dataset import WMDataset, collate_batch


# ---------------------------------------------------------------------------
# Closed-loop imagined rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def imagine_policy_rollout(
    dyn, policy, rew_head, value_head, *,
    z_ctx: torch.Tensor,          # (B, C, n_spatial, d_spatial) clean packed latents
    a_ctx: torch.Tensor,          # (B, C, 16) led-to actions (a_ctx[:, 0] = 0)
    m_ctx: torch.Tensor,          # (B, C, 16) context action mask (parse_batch convention)
    act_mask_1d: torch.Tensor,    # (B, 16) live action dims of each sample's task
    lang_emb: Optional[torch.Tensor],  # (B, lang_dim)
    horizon: int,
    sched: dict,
    k_max: int,
    tau_ctx: float,
    max_ctx: int,
    use_amp: bool,
) -> Dict[str, torch.Tensor]:
    """Batched version of `plan_cem.bc_rollout_integrated` that samples from a
    stochastic policy head and keeps the agent states.

    Alignment (same conventions as bc_rollout_integrated / imagine_batch):
      * h_t is the agent state at slot t; h_0 is the last context slot, from a
        clean prefill pass. The policy at h_t gives the action applied at z_t.
      * Applying a_t produces z_{t+1} and h_{t+1}; the reward head's MTP l=0
        at h_{t+1} predicts the reward of that transition, so r_t = R(h_{t+1}).
        (dreamer_v4 reads r and a off the same h, which is shifted by one here.)
      * V is evaluated on h_0..h_H; v_H is the TD(lambda) bootstrap.
    Episodes in MMBench2 end only by time limit, so continues are all 1.

    Returns h (B, H+1, ...), actions (B, H, A_head) as executed (clamped to
    [-1, 1]; this is what the world model saw, so it is the PMPO label),
    rewards (B, H), values (B, H+1).
    """
    device = z_ctx.device
    B, C = z_ctx.shape[:2]
    emax = int(round(math.log2(int(k_max))))
    A = int(policy.act_dim_max)

    full_actions = torch.zeros((B, C + horizon, 16), device=device, dtype=torch.float32)
    full_actions[:, :C] = a_ctx
    full_mask = act_mask_1d.view(B, 1, 16).expand(B, C + horizon, 16).clone()
    full_mask[:, :C] = m_ctx

    # Prefill: one clean pass over the context for h_0 (as bc_rollout_integrated).
    start = max(0, C - max_ctx) if max_ctx > 0 else 0
    n_pre = C - start
    with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
        _, h_full = dyn(
            full_actions[:, start:C],
            torch.full((B, n_pre), emax, device=device, dtype=torch.long),
            torch.full((B, n_pre), int(k_max), device=device, dtype=torch.long),
            z_ctx[:, start:],
            act_mask=full_mask[:, start:C],
            agent_tokens=None,
            lang_emb=lang_emb,
        )
    if h_full is None:
        raise RuntimeError("dynamics returned h_t=None; this checkpoint has n_agent=0.")

    past = z_ctx
    h_cur = h_full[:, -1:]                                   # (B, 1, ...)
    hs, acts = [h_cur], []
    for t in range(horizon):
        a = policy.sample(h_cur)[:, 0].float().clamp(-1.0, 1.0)   # (B, A)
        acts.append(a)
        n_ctx = past.shape[1]
        full_actions[:, n_ctx, :A] = a * act_mask_1d[:, :A]

        start = max(0, n_ctx - max_ctx) if max_ctx > 0 else 0
        z_next, h_cur, _ = sample_one_timestep_packed(
            dyn,
            past_packed=past[:, start:],
            k_max=k_max,
            sched=sched,
            actions=full_actions[:, start:n_ctx + 1],
            act_mask=full_mask[:, start:n_ctx + 1],
            use_amp=use_amp,
            return_h=True,
            tau_ctx=tau_ctx,
            lang_emb=lang_emb,
            use_kv_cache=True,
        )
        # The sampler runs under inference_mode; clone so later autograd
        # (policy/value losses on h) can save these tensors.
        z_next, h_cur = z_next.clone(), h_cur.clone()
        past = torch.cat([past, z_next.unsqueeze(1)], dim=1)
        hs.append(h_cur)

    h = torch.cat(hs, dim=1).float()                         # (B, H+1, ...)
    rewards = predict_rewards(rew_head, h[:, 1:], use_amp=use_amp)   # (B, H)
    values = value_head.predict(h)                           # (B, H+1)
    return {"h": h, "actions": torch.stack(acts, dim=1), "rewards": rewards.float(),
            "values": values.float()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="cup-catch")
    p.add_argument("--tasks_json", type=str, default="../tasks.json")
    p.add_argument("--data_dirs", type=str, nargs="+", default=["./data/mixed-large"])
    p.add_argument("--frame_dirs", type=str, nargs="+", default=["./data/mixed-large-shards"])
    p.add_argument("--tokenizer_ckpt", type=str, default="./checkpoints/combined/tokenizer.pt")
    p.add_argument("--dynamics_ckpt", type=str, default="./checkpoints/combined/dynamics.pt")
    p.add_argument("--heads_ckpt", type=str, required=True,
                   help="train_heads_offline.py checkpoint: its policy head is the "
                        "frozen prior and the policy init; its value head the value init")
    p.add_argument("--value_init", type=str, default="ckpt", choices=["ckpt", "fresh"],
                   help="fresh = paper (value head initialized at phase-3 start)")
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--lang_dim", type=int, default=512)
    # rollout
    p.add_argument("--ctx_len", type=int, default=4,
                   help="real context frames per rollout (plan_mpc_env --ctx_frames default)")
    p.add_argument("--horizon", type=int, default=15, help="imagined steps H (dreamer_v4: 15)")
    p.add_argument("--max_ctx", type=int, default=24,
                   help="sliding window (frames); the dynamics was trained at seq_len 24")
    p.add_argument("--tau_ctx", type=float, default=0.1, help="context corruption (paper 0.1)")
    p.add_argument("--schedule", type=str, default="shortcut", choices=["finest", "shortcut"])
    p.add_argument("--eval_d", type=float, default=0.125)
    # RL
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--lambda_", type=float, default=LAMBDA)
    p.add_argument("--alpha", type=float, default=PMPO_ALPHA)
    p.add_argument("--beta", type=float, default=PMPO_BETA)
    # optimization
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-5, help="dreamer_v4 phase 3: 3e-5")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--out", type=str, default="./logs/imagination/imag.pt")
    p.add_argument("--wandb_project", type=str, default="mmbench2-imagination")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    # --- frozen world model -------------------------------------------------
    tok, tok_info = load_tokenizer_from_ckpt(args.tokenizer_ckpt, device)
    dyn, rew_head, released_policy, dyn_info = load_dynamics_from_ckpt(
        args.dynamics_ckpt, device=device,
        d_bottleneck=int(tok_info["d_bottleneck"]),
        n_latents=int(tok_info["n_latents"]),
        packing_factor=int(args.packing_factor),
    )
    encoder = tok.encoder
    frozen = {"encoder": encoder, "decoder": tok.decoder, "dynamics": dyn,
              "reward_head": rew_head, "policy_head(released)": released_policy}
    for m in frozen.values():
        if m is not None:
            m.eval()
            m.requires_grad_(False)
    if rew_head is None:
        raise SystemExit("dynamics checkpoint has no reward head; imagination needs one.")

    patch = int(tok_info["patch"])
    n_spatial = int(dyn_info["n_spatial"])
    k_max = int(dyn_info["k_max"])
    d_model = int(dyn_info["d_model"])
    sched = make_tau_schedule(k_max=k_max, schedule=args.schedule,
                              d=(args.eval_d if args.schedule == "shortcut" else None))
    print(f"[wm] d_model={d_model} k_max={k_max} n_spatial={n_spatial} K={sched['K']}")

    # --- heads --------------------------------------------------------------
    heads_ck = torch.load(args.heads_ckpt, map_location="cpu", weights_only=False)
    heads_args = heads_ck.get("args", {}) or {}
    if heads_args.get("task") not in (None, args.task):
        raise SystemExit(f"--heads_ckpt was trained on {heads_args.get('task')!r}, not {args.task!r}")
    prior = load_policy_head_from_ckpt(args.heads_ckpt, device)          # frozen, eval
    if getattr(prior, "policy_type", "") != "gaussian":
        print(f"[warn] prior is {type(prior).__name__}; PMPO was set up for the Gaussian head")
    policy = copy.deepcopy(prior).requires_grad_(True)
    if args.value_init == "ckpt":
        value = load_value_head_from_ckpt(args.heads_ckpt, device).requires_grad_(True)
    else:
        value = ValueHeadMTP(
            d_model=d_model, num_bins=int(heads_ck.get("num_bins", 255)),
            mlp_ratio=float(heads_args.get("head_mlp_ratio", 2.0)),
            pool_agent=str(heads_args.get("head_pool", "attn")),
        ).to(device)
    # Heads stay in eval(): the offline heads use dropout, which would add noise
    # to the rollout's sampled actions, the value bootstrap and PMPO's log-probs.
    policy.eval()
    value.eval()
    params = list(policy.parameters()) + list(value.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Freeze guard, as in train_heads_offline.py: a stray requires_grad on the
    # world model would silently corrupt it while the losses still look healthy.
    frozen["policy_prior"] = prior
    trainable_ids = {id(p_) for p_ in params}
    for name, mod in frozen.items():
        if mod is None:
            continue
        if any(p_.requires_grad for p_ in mod.parameters()):
            raise RuntimeError(f"{name} has trainable params; it must be frozen.")
        if any(id(p_) in trainable_ids for p_ in mod.parameters()):
            raise RuntimeError(f"{name} parameters leaked into the optimizer.")
    print(f"[heads] policy={type(policy).__name__} "
          f"{sum(p_.numel() for p_ in policy.parameters())/1e6:.2f}M, "
          f"value={sum(p_.numel() for p_ in value.parameters())/1e6:.2f}M "
          f"(init {args.value_init}) from {args.heads_ckpt}")

    # --- data ---------------------------------------------------------------
    ds = WMDataset(
        data_dir=args.data_dirs, frames_dir=args.frame_dirs,
        seq_len=args.ctx_len, img_size=224, action_dim=16, lang_dim=args.lang_dim,
        tasks_json=args.tasks_json, tasks=[args.task], verbose=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_batch,
                        drop_last=True, pin_memory=True)

    def save(path, step):
        # train_heads_offline.py format, so heads_rl loaders and plan_mpc_env
        # read it unchanged. Head-shape args come from the heads checkpoint.
        torch.save({
            "args": {**heads_args, **vars(args)},
            "d_model": d_model, "step": step, "epoch": 0,
            "policy_head": policy.state_dict(),
            "policy_type": heads_ck.get("policy_type", "gaussian"),
            "policy_num_bins": heads_ck.get("policy_num_bins", 41),
            "bc_L": heads_ck.get("bc_L", 1),
            "act_dim_max": heads_ck.get("act_dim_max", policy.act_dim_max),
            "value_head": value.state_dict(),
            "num_bins": heads_ck.get("num_bins", 255),
        }, path)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name,
               entity=args.wandb_entity, mode=args.wandb_mode, config=vars(args))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_stem = os.path.splitext(args.out)[0]

    step, t0 = 0, time.time()
    data_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        frames, a_ctx, m_ctx, _, lang_emb = parse_batch(batch, use_rewards=False, device=device)
        act_mask_1d = batch["act_mask"][:, 0].to(device)                 # (B, 16)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                z_btLd, _ = encoder(temporal_patchify(frames, patch))
            z_ctx = pack_bottleneck_to_spatial(z_btLd.float(), n_spatial=n_spatial,
                                               k=args.packing_factor)

        ro = imagine_policy_rollout(
            dyn, policy, rew_head, value,
            z_ctx=z_ctx, a_ctx=a_ctx, m_ctx=m_ctx, act_mask_1d=act_mask_1d,
            lang_emb=lang_emb, horizon=args.horizon, sched=sched, k_max=k_max,
            tau_ctx=args.tau_ctx, max_ctx=args.max_ctx, use_amp=use_amp,
        )
        h, rewards, values = ro["h"], ro["rewards"], ro["values"]
        H = args.horizon
        continues = torch.ones_like(rewards)
        returns = compute_lambda_returns(rewards, values, continues,
                                         gamma=args.gamma, lambda_=args.lambda_)
        adv = compute_advantages(returns, values)

        A = policy.act_dim_max
        l_v = value_loss(value, h[:, :H], returns)
        l_pi = pmpo_policy_loss(policy, prior, h[:, :H], ro["actions"], adv,
                                act_mask=act_mask_1d[:, :A].view(-1, 1, A).expand(-1, H, A),
                                alpha=args.alpha, beta=args.beta)
        loss = l_v + l_pi
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        step += 1

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                m = act_mask_1d[:, :A].view(-1, 1, A).expand(-1, H, A)
                kl = policy.kl_to(prior, h[:, :H], act_mask=m).mean()
                s = {
                    "loss_value": float(l_v), "loss_policy": float(l_pi), "kl_prior": float(kl),
                    "adv_pos_frac": float((adv >= 0).float().mean()),
                    "adv_abs": float(adv.abs().mean()),
                    "reward_mean": float(rewards.mean()),
                    "return_mean": float(returns[:, 0].mean()),
                    "value_mean": float(values.mean()),
                    "policy_std": float(policy.action_std(h[:, :H])),
                    "grad_norm": float(gnorm),
                }
            it_s = step / (time.time() - t0)
            print(f"[{step}] " + " ".join(f"{k}={v:.4g}" for k, v in s.items())
                  + f" ({it_s:.2f} it/s)", flush=True)
            wandb.log({f"train/{k}": v for k, v in s.items()} | {"it_per_s": it_s}, step=step)

        if step % args.save_every == 0:
            save(args.out, step)
            save(f"{out_stem}_s{step:06d}.pt", step)
            print(f"[ckpt] step {step} -> {args.out}")

    save(args.out, step)
    print(f"[done] {step} steps -> {args.out}")
    wandb.finish()


if __name__ == "__main__":
    main()
