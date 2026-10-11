"""
Imagination-RL algebra for Dreamer-4 phase 3: lambda-returns, advantages,
value loss, and the PMPO policy loss.

Ported from the `dreamer_v4` PyTorch reproduction (MIT,
https://github.com/vijayabhaskar-ev/dreamer_v4, `imagination/algorithms.py`),
adapted to this repository's symexp-twohot heads. Equation numbers refer to
Dreamer 4 (Hafner, Yan & Lillicrap, 2025; arXiv:2509.24527).

Nothing here touches the world model: these are pure tensor operations over
rollout outputs, so the same functions serve both TD(lambda) fitting on real
dataset rewards and TD(lambda) on imagined rollouts.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model import dist_cross_entropy_from_symlog, symlog

# Paper defaults (section "Imagination training"), except GAMMA: the paper's 0.997
# (~333-step horizon) is far longer than our 24-step offline windows and than
# 100-step Meta-World episodes, so the targets were almost pure self-bootstrap.
# 0.99 (~100 steps) keeps most of the target in real rewards.
GAMMA = 0.99
LAMBDA = 0.95
PMPO_ALPHA = 0.5
PMPO_BETA = 0.3


# ---------------------------------------------------------------------------
# 1. lambda-returns (Eq. 10)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_lambda_returns(
    rewards: torch.Tensor,      # (B, H)
    values: torch.Tensor,       # (B, H+1) -- values[:, -1] is the bootstrap v_H
    continues: torch.Tensor,    # (B, H)   -- c_t in [0, 1]
    gamma: float = GAMMA,
    lambda_: float = LAMBDA,
) -> torch.Tensor:
    """TD(lambda) returns.

        R_t = r_t + gamma * c_t * ((1 - lambda) * v_{t+1} + lambda * R_{t+1})
        R_H = v_H                                    (bootstrap)

    Targets only: the whole computation runs under no_grad so gradients can
    never leak back through rewards/values/continues (detach at source).

    `continues` shares `rewards`' indexing: both are same-frame predictions, so
    c_t gates exactly the future beyond r_t. On tasks that never terminate
    (e.g. cup-catch, a 500-step time limit) c_t = 1 everywhere is exact, not an
    approximation -- but the bootstrap at the window edge is then load-bearing,
    since even gamma=0.99 has an effective horizon (~100 steps) longer than any window.

    Returns (B, H).
    """
    B, H = rewards.shape
    # Full-shape asserts: a dim-1-only check would let a (1, H+1) tensor
    # broadcast silently across the batch, and a too-long `values` would make
    # the bootstrap (values[:, -1]) and the recursion (values[:, t+1]) read
    # from inconsistent columns -- silent, and wrong.
    assert values.shape == (B, H + 1), (
        f"values must be (B, H+1)=({B}, {H + 1}) with the bootstrap in the last "
        f"column; got {tuple(values.shape)}"
    )
    assert continues.shape == rewards.shape, (
        f"continues must match rewards ({B}, {H}); got {tuple(continues.shape)}"
    )

    lambda_returns = torch.zeros_like(rewards)
    acc = values[:, -1]
    for t in reversed(range(H)):
        acc = rewards[:, t] + gamma * continues[:, t] * (
            (1 - lambda_) * values[:, t + 1] + lambda_ * acc
        )
        lambda_returns[:, t] = acc
    return lambda_returns


# ---------------------------------------------------------------------------
# 2. Advantages
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_advantages(
    lambda_returns: torch.Tensor,  # (B, H)
    values: torch.Tensor,          # (B, H+1)
) -> torch.Tensor:
    """A_t = R_t - v_t. Targets only -- PMPO consumes the sign, never a gradient."""
    B, H = lambda_returns.shape
    assert values.shape == (B, H + 1), (
        f"values must be (B, H+1)=({B}, {H + 1}); got {tuple(values.shape)}"
    )
    return lambda_returns - values[:, :H]


# ---------------------------------------------------------------------------
# 3. Value loss (Eq. 10)
# ---------------------------------------------------------------------------

def value_loss(
    value_head: nn.Module,
    h: torch.Tensor,                # (B, H, n_agent, D) or (B, H, D) -- detached
    lambda_returns: torch.Tensor,   # (B, H) -- detached targets
    mask: Optional[torch.Tensor] = None,  # (B, H)
) -> torch.Tensor:
    """Cross-entropy of the symexp-twohot value head against lambda-returns.

    `value_head(h)` follows this repo's head convention and returns
    (logits, centers_log) with logits (B, H, L, K); the value head uses L=1, so
    slot 0 is the state value.
    """
    logits, centers_log = value_head(h)          # (B, H, L, K), (K,)
    return dist_cross_entropy_from_symlog(
        logits[..., 0, :],
        symlog(lambda_returns.detach()),
        centers_log,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# 4. PMPO policy loss (Eq. 11)
# ---------------------------------------------------------------------------

def pmpo_policy_loss(
    policy_head: nn.Module,
    policy_prior: nn.Module,
    h: torch.Tensor,             # (B, H, n_agent, D) or (B, H, D) -- detached
    actions: torch.Tensor,       # (B, H, A) actions sampled during the rollout
    advantages: torch.Tensor,    # (B, H) detached
    act_mask: Optional[torch.Tensor] = None,  # (A,) or (B, H, A) valid action dims
    alpha: float = PMPO_ALPHA,
    beta: float = PMPO_BETA,
) -> torch.Tensor:
    """Preference-based MPO.

        L = (1-alpha)/|D-| * sum_{D-} log pi(a|s)
          -    alpha /|D+| * sum_{D+} log pi(a|s)
          +    beta  / N   * sum_i KL[pi(.|s_i) || prior(.|s_i)]

    with D+ = {A_t >= 0}, D- = {A_t < 0}. Only the *sign* of the advantage is
    used, which is what removes the need to normalize returns or advantages.

    The KL is reverse (pi || prior), per the paper: it constrains the policy to
    the prior's support and so limits exploitation of world-model error in
    low-probability regions. Note this cannot prevent drift on *real* states --
    the KL is evaluated on imagined ones.

    `log_prob` and `kl_to` are delegated to the head, so this works unchanged
    for any head exposing that interface.

    `act_mask` must be supplied whenever actions are zero-padded to a fixed
    width: on a 2-dof task padded to 16, the 14 dead dimensions would otherwise
    outvote the real ones 7:1 in the summed log-prob.
    """
    log_prob = policy_head.log_prob(h, actions, act_mask=act_mask)   # (B, H)

    # Float masks rather than boolean indexing: keeps shapes static and avoids
    # data-dependent control flow (both sets can be empty on a given batch).
    pos_mask = (advantages >= 0).to(log_prob.dtype)
    neg_mask = 1.0 - pos_mask
    pos_term = -(alpha * (pos_mask * log_prob).sum()) / pos_mask.sum().clamp(min=1.0)
    neg_term = (1 - alpha) * (neg_mask * log_prob).sum() / neg_mask.sum().clamp(min=1.0)

    kl = policy_head.kl_to(policy_prior, h, act_mask=act_mask)       # (B, H)
    return pos_term + neg_term + beta * kl.mean()
