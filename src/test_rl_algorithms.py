"""
Unit tests for rl_algorithms.py. Run from inside src/:

    python test_rl_algorithms.py       # or: pytest test_rl_algorithms.py

These cover the parts that fail *silently*: an off-by-one in the lambda-return
recursion, a gradient leaking back through a target, and the advantage-sign
invariance that PMPO depends on.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rl_algorithms import (
    compute_advantages,
    compute_lambda_returns,
    pmpo_policy_loss,
    value_loss,
)


def _close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


# --- lambda-returns ---------------------------------------------------------

def test_lambda_zero_is_one_step_td():
    """lambda=0 collapses to R_t = r_t + gamma * c_t * v_{t+1}."""
    B, H, g = 3, 5, 0.9
    r = torch.randn(B, H)
    v = torch.randn(B, H + 1)
    c = torch.rand(B, H)
    got = compute_lambda_returns(r, v, c, gamma=g, lambda_=0.0)
    want = r + g * c * v[:, 1:]
    assert _close(got, want), "lambda=0 must reduce to one-step TD"


def test_lambda_one_is_discounted_monte_carlo():
    """lambda=1, c=1 gives R_t = sum_k gamma^k r_{t+k} + gamma^(H-t) v_H."""
    B, H, g = 2, 6, 0.95
    r = torch.randn(B, H)
    v = torch.randn(B, H + 1)
    c = torch.ones(B, H)
    got = compute_lambda_returns(r, v, c, gamma=g, lambda_=1.0)
    want = torch.zeros(B, H)
    for t in range(H):
        acc = torch.zeros(B)
        for k in range(t, H):
            acc = acc + (g ** (k - t)) * r[:, k]
        want[:, t] = acc + (g ** (H - t)) * v[:, H]
    assert _close(got, want), "lambda=1 must reduce to discounted Monte Carlo"


def test_continue_zero_truncates():
    """c_t = 0 must cut all credit beyond r_t."""
    B, H, g = 2, 4, 0.99
    r = torch.ones(B, H)
    v = torch.full((B, H + 1), 100.0)   # large, so leakage would be obvious
    c = torch.ones(B, H)
    c[:, 2] = 0.0
    got = compute_lambda_returns(r, v, c, gamma=g, lambda_=0.9)
    assert _close(got[:, 2], torch.ones(B)), "c_t=0 must truncate the return at r_t"


def test_bootstrap_column_is_used():
    """With H=1 the return is exactly r_0 + gamma*c_0*v_1 -- catches a
    values[:, :-1] / values[:, 1:] slip that other shapes would hide."""
    r = torch.tensor([[2.0]])
    v = torch.tensor([[999.0, 5.0]])    # v[0] must not be read
    c = torch.tensor([[1.0]])
    got = compute_lambda_returns(r, v, c, gamma=0.5, lambda_=0.5)
    assert _close(got, torch.tensor([[2.0 + 0.5 * 5.0]]))


def test_targets_carry_no_gradient():
    r = torch.randn(2, 3, requires_grad=True)
    v = torch.randn(2, 4, requires_grad=True)
    c = torch.rand(2, 3, requires_grad=True)
    R = compute_lambda_returns(r, v, c)
    A = compute_advantages(R, v)
    assert not R.requires_grad, "lambda-returns must be detached at source"
    assert not A.requires_grad, "advantages must be detached at source"


def test_shape_guards_fire():
    r, c = torch.randn(2, 3), torch.ones(2, 3)
    for bad in (torch.randn(2, 3), torch.randn(2, 5), torch.randn(1, 4)):
        try:
            compute_lambda_returns(r, bad, c)
        except AssertionError:
            continue
        raise AssertionError(f"bad values shape {tuple(bad.shape)} should have been rejected")


# --- value loss -------------------------------------------------------------

class _TwoHotHead(nn.Module):
    """Minimal stand-in with this repo's (logits, centers_log) head contract."""

    def __init__(self, d=8, K=41, L=1):
        super().__init__()
        self.L, self.K = L, K
        self.out = nn.Linear(d, L * K)
        self.register_buffer("centers_log", torch.linspace(-10.0, 10.0, K))

    def forward(self, h):
        B, T, _ = h.shape
        return self.out(h).view(B, T, self.L, self.K), self.centers_log


def test_value_loss_prefers_the_true_target():
    torch.manual_seed(0)
    head = _TwoHotHead()
    h = torch.randn(4, 6, 8)
    R = torch.rand(4, 6) * 50.0
    good = value_loss(head, h, R)
    bad = value_loss(head, h, R + 400.0)   # far away in symlog space
    assert good.isfinite() and bad.isfinite()
    assert good < bad, "CE must be lower against the correct return"
    good.backward()
    assert head.out.weight.grad is not None and head.out.weight.grad.abs().sum() > 0


# --- PMPO -------------------------------------------------------------------

class _StubPolicy(nn.Module):
    """log_prob linear in a learnable scale; KL is a fixed positive constant."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def log_prob(self, h, actions, act_mask=None):
        a = actions if act_mask is None else actions * act_mask
        return -(a ** 2).sum(-1) * self.scale

    def kl_to(self, prior, h, act_mask=None):
        return torch.full(h.shape[:2], 0.25, device=h.device)


def test_pmpo_uses_only_the_advantage_sign():
    torch.manual_seed(0)
    pi, prior = _StubPolicy(), _StubPolicy()
    h = torch.randn(3, 5, 8)
    a = torch.randn(3, 5, 4)
    adv = torch.randn(3, 5)
    base = pmpo_policy_loss(pi, prior, h, a, adv)
    for k in (0.01, 7.0, 1000.0):
        scaled = pmpo_policy_loss(pi, prior, h, a, adv * k)
        assert _close(base, scaled), f"PMPO must ignore advantage magnitude (k={k})"


def test_pmpo_act_mask_excludes_padded_dims():
    torch.manual_seed(0)
    pi, prior = _StubPolicy(), _StubPolicy()
    h = torch.randn(3, 5, 8)
    adv = torch.randn(3, 5)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

    a = torch.randn(3, 5, 4)
    a_junk = a.clone()
    a_junk[..., 2:] = 99.0           # garbage in the padded dims only

    assert _close(
        pmpo_policy_loss(pi, prior, h, a, adv, act_mask=mask),
        pmpo_policy_loss(pi, prior, h, a_junk, adv, act_mask=mask),
    ), "masked dims must not affect the loss"

    assert not _close(
        pmpo_policy_loss(pi, prior, h, a, adv),
        pmpo_policy_loss(pi, prior, h, a_junk, adv),
    ), "without a mask the padded dims DO leak in -- the test itself is wrong if this passes"


def test_pmpo_handles_all_positive_and_all_negative_batches():
    pi, prior = _StubPolicy(), _StubPolicy()
    h = torch.randn(2, 4, 8)
    a = torch.randn(2, 4, 3)
    for adv in (torch.ones(2, 4), -torch.ones(2, 4)):
        loss = pmpo_policy_loss(pi, prior, h, a, adv)
        assert loss.isfinite(), "empty positive/negative set must not divide by zero"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(tests)} passed")
