"""
Fit a value head on real dataset trajectories, against a frozen world model.

This is a deliberate deviation from Dreamer 4, which initializes the value head
at the start of imagination training and only ever fits it to imagined
lambda-returns. Fitting it on data first gives (a) a value head the CEM planner
can use immediately as a terminal value -- replacing the latent-distance
stand-in `plan_cem.py` flags as a substitute "for the value of the final
imagined state (there is no value head)" -- and (b) a warm start for phase 3,
which matters because with gamma=0.997 (effective horizon ~333 steps) and a
short imagination horizon, the bootstrap carries nearly all the signal.

Nothing existing is modified: the tokenizer, dynamics, reward head and BC head
are loaded frozen, and only the new head's parameters get gradients. That is
sound because this repo already isolates BC gradients with `h_t.detach()`
(train_dynamics.py:408), so head training never needed the dynamics graph.

Run from inside src/, e.g.:

    python train_heads_offline.py --head value --task cup-catch \
        --data_dirs ./data/mixed-large ./data/expert \
        --frame_dirs ./data/mixed-large-shards ./data/expert-shards \
        --out ./logs/value_head/cup-catch.pt
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from heads_rl import CategoricalPolicyHeadMTP, ValueHeadMTP
from interactive import load_dynamics_from_ckpt, load_tokenizer_from_ckpt
from model import pack_bottleneck_to_spatial, temporal_patchify
from rl_algorithms import GAMMA, LAMBDA, compute_lambda_returns, value_loss
from train_dynamics import _emax_from_kmax, _sample_tau_for_step, parse_batch
from wm_dataset import WMDataset, collate_batch


# ---------------------------------------------------------------------------
# h_t extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_h(
    dyn,
    *,
    z1: torch.Tensor,            # (B,T,n_spatial,d_spatial) clean packed latents
    actions: torch.Tensor,       # (B,T,16) led-to convention
    act_mask: torch.Tensor,      # (B,T,16)
    lang_emb: Optional[torch.Tensor],
    k_max: int,
    mode: str = "train",
    use_amp: bool = True,
) -> torch.Tensor:
    """Agent-token states h_t for a teacher-forced real window. (B,T,n_agent,D)

    mode="train" (default) reproduces the noise distribution the released
    reward/BC heads were trained under: step_idx = emax (finest shortcut step,
    the "empirical" rows of `compute_losses`) and a per-(row, time) signal level
    drawn by `_sample_tau_for_step`. We match that rather than using clean
    latents because the reward head -- trained on exactly this distribution --
    is known to transfer to plan time, where `sample_one_timestep_packed` reads
    h off the final denoising iteration at signal index sched["tau_idx"][-1],
    which is neither fully clean nor the training draw. Inheriting the reward
    head's mismatch is the low-risk choice; inventing a third distribution is not.

    mode="clean" puts every slot at full signal (signal_idx = k_max), matching
    the context treatment in `bc_rollout_integrated`'s prefill and in
    `imagine_batch` (tau_ctx=0.0). Exposed for ablation.
    """
    device = z1.device
    B, T = z1.shape[:2]
    emax = _emax_from_kmax(k_max)
    step_idx = torch.full((B, T), emax, device=device, dtype=torch.long)

    if mode == "train":
        sigma, sigma_idx = _sample_tau_for_step(device, B, T, k_max, step_idx)
        z_in = (1.0 - sigma)[..., None, None] * torch.randn_like(z1) + sigma[..., None, None] * z1
    elif mode == "clean":
        sigma_idx = torch.full((B, T), int(k_max), device=device, dtype=torch.long)
        z_in = z1
    else:
        raise ValueError(f"unknown h_mode {mode!r}; expected 'train' or 'clean'")

    with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda"),
                        dtype=torch.bfloat16):
        _, h_t = dyn(actions, step_idx, sigma_idx, z_in, act_mask=act_mask,
                     agent_tokens=None, lang_emb=lang_emb)
    if h_t is None:
        raise RuntimeError("dynamics returned h_t=None; this checkpoint has n_agent=0.")
    return h_t.float()


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def bc_targets(actions: torch.Tensor, act_mask: torch.Tensor, L: int):
    """Replicate the released BC supervision exactly (train_dynamics.py:419-453).

    At slot t the causal context has seen obs[0..t] and actions[0..t] in led-to
    convention, so the natural target is the NEXT action, actions[t+1]; head l
    predicts actions[t+1+l]. Returns (targets, mask) both (B, T, L, A), where
    the mask is position validity (t+1+l <= T-1) times per-dim validity.

    This alignment is the single easiest thing to get silently wrong here, so
    it is copied rather than re-derived.
    """
    B, T, A = actions.shape
    device = actions.device

    actions_next = torch.zeros_like(actions)
    actions_next[:, :-1] = actions[:, 1:]
    mask_next = torch.zeros_like(act_mask)
    mask_next[:, :-1] = act_mask[:, 1:]

    valid_bt = torch.ones((B, T), device=device, dtype=torch.bool)
    valid_bt[:, -1] = False

    pad = max(0, L - 1)
    tgt = F.pad(actions_next, (0, 0, 0, pad)).unfold(1, T, 1).permute(0, 3, 1, 2).contiguous()
    dim_mask = F.pad(mask_next, (0, 0, 0, pad)).unfold(1, T, 1).permute(0, 3, 1, 2).contiguous()
    pos_mask = F.pad(valid_bt, (0, pad), value=False).unfold(1, T, 1).permute(0, 2, 1).contiguous()

    full_mask = pos_mask.unsqueeze(-1).to(dim_mask.dtype) * dim_mask   # (B,T,L,A)
    return tgt, full_mask


def value_targets(value_head, h: torch.Tensor, rewards: torch.Tensor,
                  gamma: float, lambda_: float):
    """TD(lambda) targets from REAL rewards, bootstrapped by the head itself.

    Index alignment, pinned to this repo's conventions:
      * `wm_dataset.__getitem__` stores rew[t] = reward for obs[t] -> obs[t+1].
      * `parse_batch` keeps rewards unshifted and takes frames = obs[:-1], so
        slot t of h corresponds to obs[t].
      * Hence V(h_t) = sum_k gamma^k * rew[t+k], using rewards as they come out
        of the dataset. (The reward head shifts to rew[t-1] internally because
        it predicts the reward *arriving at* obs[t]; that shift is the reward
        head's, not ours.)

    With T slots we have T values but only T-1 usable in-window transitions, so
    the horizon is H = T-1 and v_{T-1} is the bootstrap.

    Monte-Carlo returns-to-go are NOT an option here: cup-catch episodes end by
    500-step time limit, not termination, and gamma=0.997 has a longer effective
    horizon than the episode, so truncation bias would contaminate every slot.
    Bootstrapping is what makes the target well-posed.
    """
    T = h.shape[1]
    # Bootstrap targets come from the deterministic head: with --head_dropout
    # the train-mode head would inject dropout noise into its own targets.
    was_training = value_head.training
    value_head.eval()
    values = value_head.predict(h)                       # (B,T) no_grad
    value_head.train(was_training)
    continues = torch.ones_like(rewards[:, :T - 1])      # never terminates; time limit only
    returns = compute_lambda_returns(
        rewards[:, :T - 1].float(), values.float(), continues,
        gamma=gamma, lambda_=lambda_,
    )                                                    # (B, T-1)
    return returns, values


# ---------------------------------------------------------------------------
# Monte-Carlo value check
# ---------------------------------------------------------------------------

def mc_returns_per_frame(ep: torch.Tensor, rew: torch.Tensor, gamma: float,
                         tail_steps: int = 50):
    """Ground-truth discounted return-to-go for every frame of a partition.

    The value loss only measures agreement with TD(lambda) targets that the head
    bootstraps from itself, so a head can be self-consistent and still far off.
    This gives it a reference built from real rewards only.

    Same alignment as `value_targets`: rew[j] (dataset storage) is the reward for
    obs[j-1] -> obs[j], so from frame i with episode last frame b:
        G[i] = sum_{j=i+1..b} gamma^(j-i-1) * rew[j]

    Episodes end by time limit (~500 steps), while gamma=0.997 has a longer
    horizon than that, so the truncated G undershoots the infinite-horizon value
    the head is trained toward (the missing tail weighs gamma^(b-i), ~0.22 at the
    start of a 500-step episode). `G_tail` adds that tail assuming the reward
    stays at its mean over the episode's last `tail_steps` transitions:
        G_tail[i] = G[i] + gamma^(b-i) * rbar_tail / (1 - gamma)
    This is an approximation (fine for tasks that settle, e.g. reacher/cup-catch
    holding the goal), which is why the truncated error is logged too.
    """
    ep = ep.cpu().numpy()
    r = torch.nan_to_num(rew.float(), nan=0.0).cpu().double().numpy()
    N = r.shape[0]
    G = torch.zeros(N, dtype=torch.float64).numpy()
    G_tail = torch.zeros(N, dtype=torch.float64).numpy()
    bounds = [0] + [i for i in range(1, N) if ep[i] != ep[i - 1]] + [N]
    for a, e in zip(bounds[:-1], bounds[1:]):
        b = e - 1                                   # last frame of this episode
        acc = 0.0
        for i in range(b - 1, a - 1, -1):
            acc = r[i + 1] + gamma * acc
            G[i] = acc
        n_tail = max(1, min(tail_steps, b - a))
        rbar = float(r[b - n_tail + 1:b + 1].mean()) if b > a else 0.0
        for i in range(a, b + 1):
            G_tail[i] = G[i] + gamma ** (b - i) * rbar / (1.0 - gamma)
    return torch.from_numpy(G).float(), torch.from_numpy(G_tail).float()


class MCWindows(torch.utils.data.Dataset):
    """Evenly spaced windows over a (non-iid) WMDataset, each carrying the
    per-slot ground-truth returns `mc_ret` / `mc_ret_tail` (T,).

    Evenly spaced rather than the first N: the val loader's first N windows at
    stride 1 all come from a single episode.
    """

    def __init__(self, ds: WMDataset, n_windows: int, gamma: float, tail_steps: int = 50):
        n = len(ds)
        k = max(1, min(int(n_windows), n))
        self.ds = ds
        self.T = ds.T
        self.idx = torch.linspace(0, n - 1, k).round().long().unique().tolist()
        self.G, self.G_tail = [], []
        for ti in range(len(ds.tasks)):
            g, gt = mc_returns_per_frame(ds.ep[ti], ds.rew[ti], gamma, tail_steps)
            self.G.append(g)
            self.G_tail.append(gt)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        idx = self.idx[i]
        item = self.ds[idx]
        ti, start = self.ds._lookup(idx)
        # slot t of the window is obs[start + t] (parse_batch drops obs[T])
        item["mc_ret"] = self.G[ti][start:start + self.T]
        item["mc_ret_tail"] = self.G_tail[ti][start:start + self.T]
        return item


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_loader(args, split_data, split_frames, shuffle):
    ds = WMDataset(
        data_dir=split_data,
        frames_dir=split_frames,
        seq_len=args.seq_len,
        img_size=224,
        action_dim=16,
        lang_dim=args.lang_dim,
        tasks_json=args.tasks_json,
        tasks=[args.task],
        verbose=True,
    )
    return ds, DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
        collate_fn=collate_batch, drop_last=True, pin_memory=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", type=str, default="both", choices=["value", "policy", "both"],
                   help="'both' is near-free: the encoder+dynamics forward that produces "
                        "h_t is the whole cost, and both heads read the same h_t.")
    p.add_argument("--task", type=str, default="cup-catch")
    p.add_argument("--tasks_json", type=str, default="../tasks.json")
    p.add_argument("--data_dirs", type=str, nargs="+", default=["./data/mixed-large"])
    p.add_argument("--frame_dirs", type=str, nargs="+", default=["./data/mixed-large-shards"])
    p.add_argument("--val_data_dirs", type=str, nargs="+", default=["./data/val"])
    p.add_argument("--val_frame_dirs", type=str, nargs="+", default=["./data/val-shards"])
    p.add_argument("--tokenizer_ckpt", type=str, default="./checkpoints/combined/tokenizer.pt")
    p.add_argument("--dynamics_ckpt", type=str, default="./checkpoints/combined/dynamics.pt")
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--lang_dim", type=int, default=512)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=0, help="0 = no cap")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--lambda_", type=float, default=LAMBDA)
    p.add_argument("--warmup_slots", type=int, default=2,
                   help="drop the first N slots from the loss: their h has almost "
                        "no temporal context (slot 0 sees one frame, so no velocity)")
    p.add_argument("--h_mode", type=str, default="train", choices=["train", "clean"])
    p.add_argument("--value_num_bins", type=int, default=255)
    p.add_argument("--policy_num_bins", type=int, default=41)
    p.add_argument("--bc_L", type=int, default=8, help="match the released head's MTP length")
    p.add_argument("--bc_act_dim", type=int, default=16,
                   help="action dims the BC head outputs; set to the task's act_dim "
                        "(cup-catch 2, mw-* 4) to drop the zero-padded ones")
    # Head size / regularization. Defaults reproduce the original (large,
    # unregularized) heads so old checkpoints and commands behave the same.
    p.add_argument("--head_mlp_ratio", type=float, default=2.0,
                   help="projector MLP width is d_model*ratio*2/3 (SwiGLU)")
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--head_pool", type=str, default="attn", choices=["attn", "mean", "first"],
                   help="agent-token pooling; attn adds a 2*d_model^2 projection")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--mc_windows", type=int, default=320,
                   help="value head: evenly spaced val windows scored against real "
                        "discounted returns-to-go at every validation (0 = off)")
    p.add_argument("--mc_tail_steps", type=int, default=50,
                   help="transitions at episode end whose mean reward extrapolates "
                        "the return past the time limit (see mc_returns_per_frame)")
    p.add_argument("--out", type=str, default="./logs/value_head/value.pt")
    # Logging matches the repo convention: wandb, no tensorboard.
    p.add_argument("--wandb_project", type=str, default="mmbench2-heads")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"],
                   help="offline writes to ./wandb and syncs later with `wandb sync`")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    # --- frozen world model -------------------------------------------------
    tok, tok_info = load_tokenizer_from_ckpt(args.tokenizer_ckpt, device)
    dyn, rew_head, policy_head, dyn_info = load_dynamics_from_ckpt(
        args.dynamics_ckpt, device=device,
        d_bottleneck=int(tok_info["d_bottleneck"]),
        n_latents=int(tok_info["n_latents"]),
        packing_factor=int(args.packing_factor),
    )
    encoder = tok.encoder
    # The decoder is never called here (we only need latents, not pixels), but it
    # is freezed and counted anyway so the guard below accounts for the whole
    # ~356M released model rather than silently omitting a piece of it.
    decoder = tok.decoder
    for m in (encoder, decoder, dyn, rew_head, policy_head):
        if m is not None:
            m.eval()
            m.requires_grad_(False)

    patch = int(tok_info["patch"])
    n_spatial = int(dyn_info["n_spatial"])
    k_max = int(dyn_info["k_max"])
    d_model = int(dyn_info["d_model"])
    print(f"[wm] d_model={d_model} k_max={k_max} n_spatial={n_spatial} patch={patch}")

    # --- new heads ----------------------------------------------------------
    want_value = args.head in ("value", "both")
    want_policy = args.head in ("policy", "both")

    head = pol = None
    params = []
    if want_value:
        head = ValueHeadMTP(
            d_model=d_model, num_bins=args.value_num_bins,
            mlp_ratio=args.head_mlp_ratio, dropout=args.head_dropout,
            pool_agent=args.head_pool,
        ).to(device)
        params += list(head.parameters())
        print(f"[head] ValueHeadMTP params={sum(p_.numel() for p_ in head.parameters())/1e6:.2f}M")
    if want_policy:
        pol = CategoricalPolicyHeadMTP(
            d_model=d_model, L=args.bc_L, act_dim_max=args.bc_act_dim,
            num_bins=args.policy_num_bins,
            mlp_ratio=args.head_mlp_ratio, dropout=args.head_dropout,
            pool_agent=args.head_pool,
        ).to(device)
        params += list(pol.parameters())
        print(f"[head] CategoricalPolicyHeadMTP params={sum(p_.numel() for p_ in pol.parameters())/1e6:.2f}M")
        if policy_head is None:
            print("[bc] released deterministic head absent -- agreement check disabled.")
    if not params:
        raise SystemExit(f"--head={args.head!r} selected no trainable head.")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # --- freeze guard -------------------------------------------------------
    # Only the new heads may train. This is asserted rather than assumed: a
    # stray requires_grad on the dynamics would silently corrupt the pretrained
    # world model, and the run would still look healthy in the loss curves.
    frozen = {"encoder": encoder, "decoder": decoder, "dynamics": dyn,
              "reward_head": rew_head, "policy_head(released)": policy_head}
    for name, mod in frozen.items():
        if mod is None:
            continue
        leaked = [n for n, p_ in mod.named_parameters() if p_.requires_grad]
        if leaked:
            raise RuntimeError(
                f"{name} has {len(leaked)} trainable params (e.g. {leaked[:3]}); "
                f"it must be frozen.")
        if mod.training:
            raise RuntimeError(f"{name} is in train() mode; it must be in eval().")
    trainable_ids = {id(p_) for p_ in params}
    for name, mod in frozen.items():
        if mod is not None and any(id(p_) in trainable_ids for p_ in mod.parameters()):
            raise RuntimeError(f"{name} parameters leaked into the optimizer.")
    n_frozen = sum(sum(p_.numel() for p_ in m.parameters())
                   for m in frozen.values() if m is not None)
    print("[freeze] " + "  ".join(
        f"{k}={sum(p_.numel() for p_ in m.parameters())/1e6:.1f}M"
        for k, m in frozen.items() if m is not None))
    print(f"[freeze] ok: {n_frozen/1e6:.1f}M frozen (the full released world model), "
          f"{sum(p_.numel() for p_ in params)/1e6:.2f}M trainable "
          f"({100*sum(p_.numel() for p_ in params)/(n_frozen+sum(p_.numel() for p_ in params)):.1f}%)")

    def save(path, best_bc=False, best_value=False):
        blob = {"args": vars(args), "d_model": d_model, "step": step, "epoch": epoch}
        if head is not None:
            blob["value_head"] = head.state_dict()
            blob["num_bins"] = args.value_num_bins
        if pol is not None:
            blob["policy_head"] = pol.state_dict()
            blob["policy_num_bins"] = args.policy_num_bins
            blob["bc_L"] = args.bc_L
            blob["act_dim_max"] = args.bc_act_dim
        blob["best_bc"] = bool(best_bc)
        blob["best_value"] = bool(best_value)
        torch.save(blob, path)

    # --- data ---------------------------------------------------------------
    _, loader = build_loader(args, args.data_dirs, args.frame_dirs, shuffle=True)
    val_ds = val_loader = mc_loader = None
    if args.val_data_dirs and os.path.isdir(str(args.val_data_dirs[0])):
        try:
            val_ds, val_loader = build_loader(args, args.val_data_dirs, args.val_frame_dirs, shuffle=False)
        except Exception as e:  # a partition without this task is not fatal
            print(f"[val] disabled: {e}")
    if head is not None and val_ds is not None and args.mc_windows > 0:
        mc_ds = MCWindows(val_ds, args.mc_windows, args.gamma, args.mc_tail_steps)
        mc_loader = DataLoader(
            mc_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_batch, drop_last=False, pin_memory=True,
        )
        print(f"[mc] {len(mc_ds)} val windows, evenly spaced over {len(val_ds)} "
              f"(gamma={args.gamma}, tail_steps={args.mc_tail_steps})")

    def get_h(batch):
        frames, actions, act_mask, rewards, lang_emb = parse_batch(
            batch, use_rewards=True, device=device)
        if rewards is None:
            raise RuntimeError("batch has no 'rew'; value targets need rewards.")
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                z_btLd, _ = encoder(temporal_patchify(frames, patch))
            z1 = pack_bottleneck_to_spatial(
                z_btLd.float(), n_spatial=n_spatial, k=args.packing_factor)
        h = extract_h(dyn, z1=z1, actions=actions, act_mask=act_mask, lang_emb=lang_emb,
                      k_max=k_max, mode=args.h_mode, use_amp=use_amp)
        return h, actions, act_mask, rewards

    def run_batch(batch, train: bool):
        h, actions, act_mask, rewards = get_h(batch)

        stats, loss = {}, torch.zeros((), device=device)

        if head is not None:
            returns, values = value_targets(head, h, rewards, args.gamma, args.lambda_)
            H = returns.shape[1]
            mask = torch.ones_like(returns)
            if args.warmup_slots > 0:
                mask[:, :min(args.warmup_slots, H)] = 0.0
            l_v = value_loss(head, h[:, :H], returns, mask=mask)
            loss = loss + l_v
            m = mask.bool()
            stats.update(
                ce=float(l_v.detach()),
                V=float(values[:, :H][m].mean()),
                R=float(returns[m].mean()),
                mae=float((values[:, :H] - returns).abs()[m].mean()),
            )

        if pol is not None:
            tgt, full_mask = bc_targets(actions, act_mask, args.bc_L)
            l_bc = pol.bc_loss(h, tgt, full_mask)
            loss = loss + l_bc
            stats["nll"] = float(l_bc.detach())
            with torch.no_grad():
                stats["astd"] = float(pol.action_std(h))
                # Pipeline check: the categorical head's mean action should
                # track the released deterministic head on the same h_t and the
                # same l=0 target. A large gap means h_t extraction or the MTP
                # shift is wrong, not that the head is undertrained -- it is the
                # only correctness signal available without a closed-loop eval.
                if policy_head is not None:
                    A = pol.act_dim_max
                    ref = policy_head(h.detach())[:, :, 0, :A]        # (B,T,A) tanh means
                    mine = pol.mean(h)                                # (B,T,A)
                    dm = full_mask[:, :, 0, :A]                       # (B,T,A)
                    stats["agree"] = float(
                        (((mine - ref) ** 2) * dm).sum() / dm.sum().clamp_min(1.0)
                    )

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()

        stats["loss"] = float(loss.detach())
        return stats

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config={
            **vars(args),
            "model/frozen_params": n_frozen,
            "model/trainable_params": sum(p_.numel() for p_ in params),
            "model/d_model": d_model,
            "model/k_max": k_max,
            "model/n_spatial": n_spatial,
            "data/train_windows": len(loader.dataset),
            "data/val_windows": len(val_loader.dataset) if val_loader is not None else 0,
        },
    )

    trainables = [m for m in (head, pol) if m is not None]

    def _fmt(s):
        order = ("loss", "ce", "V", "R", "mae", "nll", "agree", "astd")
        return " ".join(f"{k}={s[k]:.4g}" for k in order if k in s)

    @torch.no_grad()
    def validate():
        for m in trainables:
            m.eval()
        acc, n = {}, 0
        for i, b in enumerate(val_loader):
            if i >= args.val_batches:
                break
            s = run_batch(b, train=False)
            for k, v in s.items():
                acc[k] = acc.get(k, 0.0) + v
            n += 1
        for m in trainables:
            m.train()
        return {k: v / max(n, 1) for k, v in acc.items()}

    @torch.no_grad()
    def mc_validate():
        """V(h_t) vs real discounted returns-to-go, over the value-trained slots
        [warmup_slots, T-1). mc_* use the tail-corrected return; mc_mae_trunc the
        plain truncated one. mc_corr is what matters most for ranking in CEM."""
        head.eval()
        V, G, Gt = [], [], []
        for b in mc_loader:
            h = get_h(b)[0]
            sl = slice(args.warmup_slots, h.shape[1] - 1)
            V.append(head.predict(h)[:, sl].flatten().float().cpu())
            G.append(b["mc_ret"][:, sl].flatten())
            Gt.append(b["mc_ret_tail"][:, sl].flatten())
        head.train()
        V, G, Gt = torch.cat(V), torch.cat(G), torch.cat(Gt)
        corr = torch.corrcoef(torch.stack([V, Gt]))[0, 1]
        return {
            "mc_mae": float((V - Gt).abs().mean()),
            "mc_bias": float((V - Gt).mean()),
            "mc_corr": float(torch.nan_to_num(corr, nan=0.0)),
            "mc_mae_trunc": float((V - G).abs().mean()),
            "mc_V": float(V.mean()),
            "mc_G": float(Gt.mean()),
        }

    # --- loop ---------------------------------------------------------------
    # The two heads converge on very different timescales: BC is a dense
    # supervised map that tends to bottom out within a few epochs and then
    # overfit, while the value head needs enough passes for the window-edge
    # bootstrap to propagate (~333/seq_len epochs at gamma=0.997). Rather than
    # scheduling them separately, keep a best-by-val-NLL BC snapshot so the
    # policy head is not hostage to the value head's longer run.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_stem = os.path.splitext(args.out)[0]
    best_path = out_stem + "_bc_best.pt"
    best_nll = float("inf")
    # Value snapshot selected by the MC check, which measures accuracy against
    # real returns (the value ce/mae only measure self-consistency).
    best_value_path = out_stem + "_value_best.pt"
    best_mc = float("inf")
    for m in trainables:
        m.train()

    step, epoch, t0 = 0, 0, time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            s = run_batch(batch, train=True)
            step += 1
            if step % args.log_every == 0:
                print(f"[{epoch}:{step}] {_fmt(s)} ({step/(time.time()-t0):.2f} it/s)")
                wandb.log({f"train/{k}": v for k, v in s.items()}
                          | {"epoch": epoch, "it_per_s": step / (time.time() - t0)},
                          step=step)
            if val_loader is not None and step % args.val_every == 0:
                v = validate()
                print(f"[{epoch}:{step}] VAL {_fmt(v)}")
                wandb.log({f"val/{k}": x for k, x in v.items()}, step=step)
                if pol is not None and "nll" in v and v["nll"] < best_nll:
                    best_nll = v["nll"]
                    save(best_path, best_bc=True)
                    wandb.log({"val/best_nll": best_nll}, step=step)
                    print(f"[ckpt] new best BC val nll={best_nll:.4f} -> {best_path}")
                if mc_loader is not None:
                    mc = mc_validate()
                    print(f"[{epoch}:{step}] MC  " + " ".join(f"{k}={x:.4g}" for k, x in mc.items()))
                    wandb.log({f"val/{k}": x for k, x in mc.items()}, step=step)
                    if mc["mc_mae"] < best_mc:
                        best_mc = mc["mc_mae"]
                        save(best_value_path, best_value=True)
                        wandb.log({"val/best_mc_mae": best_mc}, step=step)
                        print(f"[ckpt] new best value mc_mae={best_mc:.4g} -> {best_value_path}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        save(args.out)
        # Keep every epoch as well: the latest-only file above made it impossible
        # to go back to an earlier value head after seeing closed-loop results.
        ep_path = f"{out_stem}_ep{epoch:02d}.pt"
        save(ep_path)
        print(f"[ckpt] epoch {epoch} -> {args.out} (+ {ep_path})")

    save(args.out)
    print(f"[done] {step} steps -> {args.out}"
          + (f" (best BC: {best_path}, nll={best_nll:.4f})" if best_nll < float("inf") else "")
          + (f" (best value: {best_value_path}, mc_mae={best_mc:.4g})" if best_mc < float("inf") else ""))
    wandb.finish()


if __name__ == "__main__":
    main()
