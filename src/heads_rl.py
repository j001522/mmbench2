"""
Heads added for Dreamer-4 phase 3 (imagination RL).

Currently: `ValueHeadMTP`, the symexp-twohot state-value head the paper
initializes at the start of imagination training ("We initialize a value head
and a frozen copy of the policy head that serves as a behavioral prior").
The dynamics, tokenizer, reward head and BC head are all untouched.

The value head subclasses `RewardHeadMTP` rather than copying it, so the
agent-token attention pooling stays bit-identical to the reward head's -- both
read the same h_t, and any fix to that pooling applies to both. The only
differences are L=1 (a state value has no multi-token-prediction ladder) and a
scalar `predict()` decode.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from model import MLP, PolicyHeadMTP, RewardHeadMTP, symexp


class ValueHeadMTP(RewardHeadMTP):
    """V(h_t) as a symexp-twohot distribution over `num_bins` bins.

    Bin range: the released reward head uses log_low=-10, log_high=10 in symlog
    space. Values are far larger than single-step rewards, but cup-catch caps at
    ~500 return and symlog(500) ~= 6.2, so the same range covers it with room to
    spare. Keep both heads on the same grid unless a task needs otherwise.

    forward(h_t) -> (logits, centers_log) with logits (B, T, 1, K), matching the
    reward head's contract so `rl_algorithms.value_loss` can consume either.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_bins: int = 255,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        log_low: float = -10.0,
        log_high: float = 10.0,
        pool_agent: str = "attn",
    ):
        super().__init__(
            d_model=d_model,
            L=1,
            num_bins=num_bins,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            log_low=log_low,
            log_high=log_high,
            pool_agent=pool_agent,
        )
        # RewardHeadMTP biases its output hard toward the symlog(0) bin (-5
        # everywhere, +5 at zero), which matches a sparse-reward marginal. A
        # value head's marginal is not sparse -- under gamma=0.997 on cup-catch
        # V sits in the tens-to-hundreds -- so that spike is a confidently wrong
        # prior the head has to unlearn (CE 10.0 at init vs ln(K)=5.5 uniform).
        # Reset to uniform.
        with torch.no_grad():
            self.out.bias.zero_()

    @torch.no_grad()
    def predict(self, h_t: torch.Tensor) -> torch.Tensor:
        """Expected value as a scalar. (B, T, n_agent, D) or (B, T, D) -> (B, T).

        Decodes the distribution mean, not its mode -- the same convention
        `plan_cem.reward_cost` uses for predicted reward.
        """
        logits, centers_log = self(h_t)                         # (B,T,1,K), (K,)
        probs = logits[..., 0, :].float().softmax(dim=-1)       # (B,T,K)
        expected_symlog = (probs * centers_log.float().view(1, 1, -1)).sum(-1)
        return symexp(expected_symlog)                          # (B,T)


class _PooledPolicyBase(nn.Module):
    """Shared trunk of the stochastic policy heads: agent pooling + projector +
    one zero-initialized output layer of `out_per_dim` numbers per (l, action dim).

    Parameter names (pool_query, pool_kv, projector, out) are the ones the
    categorical head always had, so its existing checkpoints load unchanged.

    Head l at slot t describes action_{t+1+l}, matching the BC target shift in
    train_dynamics.py.

    `act_mask` is not optional in practice. Actions are zero-padded to 16 dims
    and cup-catch uses 2, so an unmasked sum over dims lets 14 dead dimensions
    outvote the live ones 7:1 in both the log-prob and the KL.
    """

    def __init__(
        self,
        *,
        d_model: int,
        L: int,
        act_dim_max: int,
        out_per_dim: int,
        mlp_ratio: float,
        dropout: float,
        pool_agent: str,
    ):
        super().__init__()
        self.L = int(L)
        self.act_dim_max = int(act_dim_max)
        self.d_model = int(d_model)
        self.pool_agent = pool_agent
        if pool_agent not in ("attn", "mean", "first"):
            raise ValueError(f"pool_agent must be attn|mean|first, got {pool_agent}")

        if pool_agent == "attn":
            self.pool_query = nn.Parameter(torch.randn(self.d_model) * 0.02)
            self.pool_kv = nn.Linear(self.d_model, 2 * self.d_model, bias=False)

        self.projector = MLP(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.out = nn.Linear(d_model, self.L * self.act_dim_max * int(out_per_dim))
        # Zero init -> maximum-entropy start (uniform bins / N(0, 1)), so BC
        # does not begin from an arbitrary peak. Gradients still flow from
        # step 1 because the bias is what moves first.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _raw(self, h_t: torch.Tensor) -> torch.Tensor:
        """-> (B, T, L, A, out_per_dim)."""
        h = self._pool(h_t)
        B, T, _ = h.shape
        return self.out(self.projector(h)).view(B, T, self.L, self.act_dim_max, -1)

    # -- pooling: mirrors RewardHeadMTP/PolicyHeadMTP exactly so all three
    #    heads read h_t identically. Duplicated rather than refactored to keep
    #    this file additive -- model.py is untouched.
    def _pool(self, h_t: torch.Tensor) -> torch.Tensor:
        if h_t.dim() != 4:
            return h_t
        if self.pool_agent == "first":
            return h_t[:, :, 0, :]
        if self.pool_agent == "mean":
            return h_t.mean(dim=2)
        B, T, N, D = h_t.shape
        k, v = self.pool_kv(h_t).chunk(2, dim=-1)
        q = self.pool_query.to(dtype=k.dtype)
        attn = ((k * q).sum(dim=-1) / math.sqrt(D)).softmax(dim=-1)
        return (attn.unsqueeze(-1) * v).sum(dim=2)

    # -- helpers ------------------------------------------------------------
    def _clip_dims(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Drop trailing padded action dims beyond act_dim_max.

        Actions arrive zero-padded to 16; a head built with act_dim_max < 16
        only has logits for the first act_dim_max. Dropping a dim that is live
        in the mask would silently train on a subset of the action, so fail.
        """
        A = self.act_dim_max
        if x.shape[-1] <= A:
            return x, mask
        if mask is not None and bool((mask[..., A:] != 0).any()):
            raise ValueError(
                f"act_dim_max={A} but the action mask has live dims beyond it; "
                f"set --bc_act_dim to at least the task's action dim.")
        return x[..., :A], (mask[..., :A] if mask is not None else None)

    @staticmethod
    def _broadcast_mask(act_mask: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        """(A,) or (B,T,A) -> (B,T,A) float, or all-ones if None."""
        if act_mask is None:
            return torch.ones_like(like)
        m = act_mask.to(dtype=like.dtype, device=like.device)
        if m.dim() == 1:
            m = m.view(1, 1, -1).expand_as(like)
        return m

    def act(self, h_t: torch.Tensor, readout: str = "sample", l: int = 0) -> torch.Tensor:
        if readout == "sample":
            return self.sample(h_t, l)
        if readout == "mean":
            return self.mean(h_t, l)
        if readout in ("argmax", "mode"):
            return self.argmax(h_t, l)
        raise ValueError(f"unknown readout {readout!r}; expected sample|mean|argmax")


class CategoricalPolicyHeadMTP(_PooledPolicyBase):
    """Per-dimension discretized policy over agent tokens h_t.

    The released `PolicyHeadMTP` is a deterministic tanh-MSE head: it has no
    density, so it cannot supply `log_prob` or act as a KL target, and PMPO
    needs both. This is the paper's parameterization -- "categorical or
    vectorized binary distribution, depending on the action space" -- keeping
    the same attention pooling, projector and MTP layout so it is a drop-in
    alongside the existing heads. See `GaussianPolicyHeadMTP` for the standard
    continuous-control alternative.

    Output: logits (B, T, L, A, K).
    """

    policy_type = "categorical"

    def __init__(
        self,
        *,
        d_model: int,
        L: int = 8,
        act_dim_max: int = 16,
        num_bins: int = 41,
        action_bound: float = 1.0,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        pool_agent: str = "attn",
    ):
        super().__init__(
            d_model=d_model, L=L, act_dim_max=act_dim_max, out_per_dim=num_bins,
            mlp_ratio=mlp_ratio, dropout=dropout, pool_agent=pool_agent,
        )
        self.num_bins = int(num_bins)
        self.action_bound = float(action_bound)
        self.register_buffer(
            "bin_centers",
            torch.linspace(-self.action_bound, self.action_bound, self.num_bins),
        )

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """-> (B, T, L, A, K) logits."""
        return self._raw(h_t)

    def _nearest_bin(self, actions: torch.Tensor) -> torch.Tensor:
        a = actions.clamp(-self.action_bound, self.action_bound)
        pos = (a + self.action_bound) / (2 * self.action_bound) * (self.num_bins - 1)
        return pos.round().long().clamp(0, self.num_bins - 1)

    # -- distribution interface consumed by PMPO ----------------------------
    def log_prob(self, h_t: torch.Tensor, actions: torch.Tensor,
                 act_mask: Optional[torch.Tensor] = None, l: int = 0) -> torch.Tensor:
        """sum over VALID dims of log pi(a_d | h). (B, T)."""
        logp = self(h_t)[:, :, l].log_softmax(dim=-1)        # (B,T,A,K)
        actions, act_mask = self._clip_dims(
            actions, None if act_mask is None else act_mask.expand_as(actions))
        A = actions.shape[-1]
        bins = self._nearest_bin(actions)                    # (B,T,A)
        lp = logp[:, :, :A].gather(-1, bins.unsqueeze(-1)).squeeze(-1)   # (B,T,A)
        m = self._broadcast_mask(act_mask, lp)
        return (lp * m).sum(-1)

    def kl_to(self, prior: "CategoricalPolicyHeadMTP", h_t: torch.Tensor,
              act_mask: Optional[torch.Tensor] = None, l: int = 0) -> torch.Tensor:
        """Reverse KL[pi || prior], summed over valid dims. (B, T)."""
        logp = self(h_t)[:, :, l].log_softmax(dim=-1)
        with torch.no_grad():
            logq = prior(h_t)[:, :, l].log_softmax(dim=-1)
        kl = (logp.exp() * (logp - logq)).sum(-1)            # (B,T,A)
        return (kl * self._broadcast_mask(act_mask, kl)).sum(-1)

    # -- readouts -----------------------------------------------------------
    def sample(self, h_t: torch.Tensor, l: int = 0) -> torch.Tensor:
        probs = self(h_t)[:, :, l].softmax(dim=-1)           # (B,T,A,K)
        idx = torch.multinomial(probs.reshape(-1, self.num_bins), 1)
        return self.bin_centers[idx.reshape(probs.shape[:-1])]

    def mean(self, h_t: torch.Tensor, l: int = 0) -> torch.Tensor:
        probs = self(h_t)[:, :, l].softmax(dim=-1)
        return (probs * self.bin_centers).sum(-1)

    def argmax(self, h_t: torch.Tensor, l: int = 0) -> torch.Tensor:
        return self.bin_centers[self(h_t)[:, :, l].argmax(dim=-1)]

    def act(self, h_t: torch.Tensor, readout: str = "sample", l: int = 0) -> torch.Tensor:
        """Head-agnostic readout. Default "sample": the readout PMPO optimizes,
        and the one dreamer_v4 measured as the large closed-loop winner on
        cup-catch (~0.47 sampled vs ~0.10 for mean/argmax, the latter at the
        random floor). Use mean/argmax only for deterministic eval."""
        return super().act(h_t, readout=readout, l=l)

    def action_std(self, h_t: torch.Tensor, l: int = 0) -> torch.Tensor:
        """Analytic per-dim std over bin centers -> 0 as the policy collapses.
        The diagnostic to watch during PMPO."""
        probs = self(h_t)[:, :, l].softmax(dim=-1)
        mu = (probs * self.bin_centers).sum(-1, keepdim=True)
        var = (probs * (self.bin_centers - mu) ** 2).sum(-1)
        return var.clamp_min(0).sqrt().mean()

    # -- BC loss ------------------------------------------------------------
    def bc_loss(self, h_t: torch.Tensor, targets_btla: torch.Tensor,
                full_mask_btla: torch.Tensor) -> torch.Tensor:
        """Masked categorical NLL over all L MTP slots.

        `targets_btla` / `full_mask_btla` are built exactly as the released BC
        loss builds them (train_dynamics.py: shift to actions[t+1], pad+unfold
        to (B,T,L,A), multiply position validity by per-dim validity), so this
        head is trained on the same supervision as the deterministic one.
        """
        logp = self(h_t).log_softmax(dim=-1)                 # (B,T,L,A,K)
        targets_btla, full_mask_btla = self._clip_dims(targets_btla, full_mask_btla)
        A = targets_btla.shape[-1]
        bins = self._nearest_bin(targets_btla)               # (B,T,L,A)
        lp = logp[..., :A, :].gather(-1, bins.unsqueeze(-1)).squeeze(-1)  # (B,T,L,A)
        m = full_mask_btla.to(lp.dtype)
        return -(lp * m).sum() / m.sum().clamp_min(1.0)


class GaussianPolicyHeadMTP(_PooledPolicyBase):
    """Diagonal-Gaussian policy over agent tokens h_t -- the standard
    continuous-control parameterization (MPO family, which PMPO belongs to).

    Mirrors dreamer_v4/heads.py:PolicyHead: per-dim (mu, log_std) with log_std
    clamped to [LOG_STD_MIN, LOG_STD_MAX] (floor raised to DreamerV3's, see
    below), zero-init output (N(0, 1) at step 0),
    no tanh squashing. Samples can therefore leave [-1, 1]; the caller clamps
    before execution (plan_cem / plan_mpc_env already do). BC targets lie in
    [-1, 1], so the fitted mean stays there in practice.

    Output: (mu, log_std), each (B, T, L, A).
    """

    policy_type = "gaussian"
    # sigma >= 0.1 as in DreamerV3 (minstd=0.1). dreamer_v4's -5.0 (sigma ~0.007)
    # let BC collapse sigma on some states: cup-catch run 27662439 had val nll0
    # rise 1 -> 10-20 while the mean-action error (mse0) stayed flat, i.e. an
    # overconfident prior, which would also blow up PMPO's reverse KL there.
    LOG_STD_MIN = math.log(0.1)
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        *,
        d_model: int,
        L: int = 8,
        act_dim_max: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        pool_agent: str = "attn",
    ):
        super().__init__(
            d_model=d_model, L=L, act_dim_max=act_dim_max, out_per_dim=2,
            mlp_ratio=mlp_ratio, dropout=dropout, pool_agent=pool_agent,
        )

    def forward(self, h_t: torch.Tensor):
        """-> (mu, log_std), each (B, T, L, A). Computed in fp32 so the density
        is not evaluated in bf16 under autocast."""
        raw = self._raw(h_t).float()
        mu, log_std = raw[..., 0], raw[..., 1]
        return mu, log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    @staticmethod
    def _log_density(mu, log_std, actions):
        """Per-dim log N(a; mu, sigma^2)."""
        return -0.5 * (((actions - mu) * torch.exp(-log_std)) ** 2
                       + 2.0 * log_std + math.log(2.0 * math.pi))

    # -- distribution interface consumed by PMPO ----------------------------
    def log_prob(self, h_t: torch.Tensor, actions: torch.Tensor,
                 act_mask: Optional[torch.Tensor] = None, l: int = 0) -> torch.Tensor:
        """sum over VALID dims of log pi(a_d | h). (B, T)."""
        mu, log_std = self(h_t)
        mu, log_std = mu[:, :, l], log_std[:, :, l]          # (B,T,A)
        actions, act_mask = self._clip_dims(
            actions, None if act_mask is None else act_mask.expand_as(actions))
        A = actions.shape[-1]
        lp = self._log_density(mu[..., :A], log_std[..., :A], actions.float())
        return (lp * self._broadcast_mask(act_mask, lp)).sum(-1)

    def kl_to(self, prior: "GaussianPolicyHeadMTP", h_t: torch.Tensor,
              act_mask: Optional[torch.Tensor] = None, l: int = 0) -> torch.Tensor:
        """Reverse KL[pi || prior] in closed form, summed over valid dims. (B, T)."""
        mu, log_std = self(h_t)
        mu, log_std = mu[:, :, l], log_std[:, :, l]
        with torch.no_grad():
            mu_q, log_std_q = prior(h_t)
            mu_q, log_std_q = mu_q[:, :, l], log_std_q[:, :, l]
        kl = (log_std_q - log_std
              + (torch.exp(2.0 * log_std) + (mu - mu_q) ** 2) / (2.0 * torch.exp(2.0 * log_std_q))
              - 0.5)                                         # (B,T,A)
        return (kl * self._broadcast_mask(act_mask, kl)).sum(-1)

    # -- readouts -----------------------------------------------------------
    def sample(self, h_t: torch.Tensor, l: int = 0) -> torch.Tensor:
        """Plain (non-reparameterized) draw: PMPO uses the action as a fixed
        label for its score-function gradient, never backprops through it."""
        mu, log_std = self(h_t)
        return torch.normal(mu[:, :, l], log_std[:, :, l].exp())

    def mean(self, h_t: torch.Tensor, l: int = 0) -> torch.Tensor:
        return self(h_t)[0][:, :, l]

    # For a Gaussian the mean is the mode.
    argmax = mean

    def action_std(self, h_t: torch.Tensor, l: int = 0) -> torch.Tensor:
        """Mean per-dim sigma -> 0 as the policy collapses. Same diagnostic as
        the categorical head's analytic bin std."""
        return self(h_t)[1][:, :, l].exp().mean()

    # -- BC loss ------------------------------------------------------------
    def bc_loss(self, h_t: torch.Tensor, targets_btla: torch.Tensor,
                full_mask_btla: torch.Tensor) -> torch.Tensor:
        """Masked Gaussian NLL over all L MTP slots (same targets/mask as the
        categorical head). Note: a continuous NLL, so it can go negative and is
        not comparable in value to the categorical head's NLL."""
        mu, log_std = self(h_t)                              # (B,T,L,A)
        targets_btla, full_mask_btla = self._clip_dims(targets_btla, full_mask_btla)
        A = targets_btla.shape[-1]
        lp = self._log_density(mu[..., :A], log_std[..., :A], targets_btla.float())
        m = full_mask_btla.to(lp.dtype)
        return -(lp * m).sum() / m.sum().clamp_min(1.0)


def build_policy_head(policy_type: str, **kw) -> nn.Module:
    """`num_bins` / `action_bound` are only consumed by the categorical head."""
    if policy_type == "categorical":
        return CategoricalPolicyHeadMTP(**kw)
    if policy_type == "gaussian":
        kw.pop("num_bins", None)
        kw.pop("action_bound", None)
        return GaussianPolicyHeadMTP(**kw)
    raise ValueError(f"unknown policy_type {policy_type!r}; expected categorical|gaussian")


def load_value_head_from_ckpt(path: str, device) -> ValueHeadMTP:
    """Rebuild a ValueHeadMTP from a `train_heads_offline.py` checkpoint.

    Mirrors the loader convention in interactive.py: shape the module from the
    saved args, then load weights strictly so a config drift fails loudly.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    head = ValueHeadMTP(
        d_model=int(ck.get("d_model", a.get("d_model", 1024))),
        num_bins=int(ck.get("num_bins", a.get("value_num_bins", 255))),
        mlp_ratio=float(a.get("head_mlp_ratio", 2.0)),
        pool_agent=str(a.get("head_pool", "attn")),
    )
    head.load_state_dict(ck["value_head"], strict=True)
    head.to(device).eval().requires_grad_(False)
    return head


def load_policy_head_from_ckpt(path: str, device) -> nn.Module:
    """Rebuild a stochastic policy head from a `train_heads_offline.py` ckpt.
    Checkpoints written before `policy_type` existed are categorical."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    head = build_policy_head(
        str(ck.get("policy_type", a.get("policy_type", "categorical"))),
        d_model=int(ck.get("d_model", a.get("d_model", 1024))),
        L=int(ck.get("bc_L", a.get("bc_L", 8))),
        act_dim_max=int(ck.get("act_dim_max", a.get("bc_act_dim", 16))),
        num_bins=int(ck.get("policy_num_bins", a.get("policy_num_bins", 41))),
        mlp_ratio=float(a.get("head_mlp_ratio", 2.0)),
        pool_agent=str(a.get("head_pool", "attn")),
    )
    head.load_state_dict(ck["policy_head"], strict=True)
    head.to(device).eval().requires_grad_(False)
    return head


class CategoricalPolicyAdapter(nn.Module):
    """Call a stochastic policy head the way the released `PolicyHeadMTP` is called.

    Works for both `CategoricalPolicyHeadMTP` and `GaussianPolicyHeadMTP` (they
    share `act()`); the name predates the Gaussian head.

    `plan_cem.bc_rollout_integrated` does `policy_head(h_cur)[:, 0, 0, :act_dim]`,
    i.e. it expects (B, T, L, A) continuous actions in [-1, 1]. The stochastic
    heads return distribution parameters instead, so collapse them with a
    readout and re-insert the singleton L axis the caller indexes. Only l=0 is
    ever read there, so emitting L=1 is sufficient.

    Readout default is "sample", not the mean or the mode: `act()` documents that
    on cup-catch the sampled readout scores ~0.47 closed-loop while mean/argmax
    sit at ~0.10, the random floor. A discretized policy fitted by BC puts mass
    on several bins, and its per-dim mean can land in a valley between them --
    an action the policy considers unlikely. Reproducibility comes from the
    global `torch.manual_seed(args.seed)` in the planner, since
    `CategoricalPolicyHeadMTP.sample` draws on the default generator.
    """

    def __init__(self, head: _PooledPolicyBase, readout: str = "sample"):
        super().__init__()
        if readout not in ("sample", "mean", "argmax", "mode"):
            raise ValueError(f"unknown readout {readout!r}; expected sample|mean|argmax")
        self.head = head
        self.readout = str(readout)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """-> (B, T, 1, A) in [-1, 1]; the caller clamps and masks."""
        a = self.head.act(h_t, readout=self.readout, l=0)   # (B, T, A)
        return a.unsqueeze(2)
