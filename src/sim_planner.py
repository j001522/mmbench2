# sim_planner.py
"""
Planning under the TRUE model: CEM whose rollouts run in the simulator itself,
scored with the true env reward. Used by plan_mpc_env.py --compare_sim to put
the world-model planner's decision next to the one an oracle planner would
take from the same real state.

The rollouts run in the REAL env, between a full save and an exact restore of
its state (MuJoCo integration state + the wrappers' Python counters), and
bypass the Pixels wrapper, so nothing is rendered. A separate synced copy was
not used because per-episode randomness lives outside the MuJoCo state
(reacher's target in the model, Meta-World's goal in Python attributes).
Every planning call checks that the state after the final restore is
bit-identical to the saved one.

Supported: DMControl (dmcontrol.DMControlWrapper) and Meta-World
(metaworld.MetaWorldWrapper) tasks.

Self-test (CPU, no rendering):
    python sim_planner.py --tasks reacher-easy cup-catch mw-reach mw-plate-slide
"""
import argparse

import mujoco
import numpy as np
import torch

from envs.wrappers.timeout import Timeout
from plan_cem import colored_noise

FULL = mujoco.mjtState.mjSTATE_INTEGRATION


class SimHandle:
    """Save / restore / step (no render) for the env built by envs.make_env."""

    def __init__(self, env):
        from envs.dmcontrol import DMControlWrapper
        from envs.metaworld import MetaWorldWrapper
        self.timeout, e = None, env
        while True:
            if isinstance(e, Timeout) and self.timeout is None:
                self.timeout = e
            if isinstance(e, (DMControlWrapper, MetaWorldWrapper)):
                break
            if not hasattr(e, "env"):
                raise TypeError(f"no DMControl/Meta-World layer under {type(env).__name__}")
            e = e.env
        self.base = e
        if isinstance(e, DMControlWrapper):
            from dm_control.rl import control
            self.kind = "dmc"
            self.ctrl = e.env                       # action_scale wrapper: takes [-1, 1] actions
            dm = e.env
            while not isinstance(dm, control.Environment):
                dm = dm._env
            self.dm = dm
            self.m, self.d = dm.physics.model.ptr, dm.physics.data.ptr
            self.dtype = e.action_spec_dtype
        else:
            self.kind = "mw"
            self.core = e.env                       # SawyerXYZEnv
            self.m, self.d = self.core.model, self.core.data
        self._size = mujoco.mj_stateSize(self.m, FULL)

    # ---------------------------------------------------------------- state
    def mj_state(self):
        s = np.empty(self._size, np.float64)
        mujoco.mj_getState(self.m, self.d, s, FULL)
        return s

    def save(self):
        st = {"mj": self.mj_state(), "t": None if self.timeout is None else getattr(self.timeout, "_t", None)}
        if self.kind == "dmc":
            st.update(step_count=self.dm._step_count, reset_next=self.dm._reset_next_step,
                      cum=self.base._cumulative_reward)
        else:
            c = self.core
            st.update(path=c.curr_path_length, prev_obs=np.copy(c._prev_obs),
                      last_obs=None if c._last_stable_obs is None else np.copy(c._last_stable_obs),
                      sim_exc=c._did_see_sim_exception)
        return st

    def restore(self, st):
        mujoco.mj_setState(self.m, self.d, st["mj"], FULL)
        mujoco.mj_forward(self.m, self.d)
        if self.timeout is not None and st["t"] is not None:
            self.timeout._t = st["t"]
        if self.kind == "dmc":
            self.dm._step_count = st["step_count"]
            self.dm._reset_next_step = st["reset_next"]
            self.base._cumulative_reward = st["cum"]
        else:
            c = self.core
            c.curr_path_length = st["path"]
            c._prev_obs = np.copy(st["prev_obs"])
            c._last_stable_obs = None if st["last_obs"] is None else np.copy(st["last_obs"])
            c._did_see_sim_exception = st["sim_exc"]

    def same(self, st):
        return np.array_equal(self.mj_state(), st["mj"])

    # ---------------------------------------------------------------- step
    def step(self, action):
        """One env step as the wrappers do it (2 sub-steps, rewards summed),
        without rendering. action: real-dim float array in [-1, 1].
        Returns (reward, done)."""
        r = 0.0
        if self.kind == "dmc":
            a = np.asarray(action, dtype=self.dtype)
            for _ in range(2):
                ts = self.ctrl.step(a)
                r += float(ts.reward or 0.0)
                if ts.last():
                    return r, True
            return r, False
        for _ in range(2):
            _o, rr, term, trunc, _i = self.core.step(np.asarray(action, dtype=np.float32).copy())
            r += float(rr)
            if term or trunc:
                return r, True
        return r, False

    def returns(self, plans, discount=1.0):
        """True discounted returns of (N, H, real_dim) plans from the current
        state; the state is restored afterwards (checked)."""
        base = self.save()
        out = np.zeros(len(plans))
        for i, p in enumerate(plans):
            self.restore(base)
            g = 1.0
            for a in p:
                r, done = self.step(a)
                out[i] += g * r
                g *= discount
                if done:
                    break
        self.restore(base)
        if not self.same(base):
            raise RuntimeError("sim_planner: env state not restored exactly")
        return out

    def returns_last_frames(self, plans, render_fn, n_last, discount=1.0):
        """sim.returns, also rendering the frames after each of the last
        n_last steps of every plan (for a learned terminal value on the true
        state). Returns (returns (N,), frames: N lists of (3, H, W) uint8,
        n_steps (N,) steps actually taken; < H if the episode ended)."""
        base = self.save()
        out = np.zeros(len(plans))
        frames, n_steps = [], np.zeros(len(plans), dtype=int)
        for i, p in enumerate(plans):
            self.restore(base)
            g, fr, T = 1.0, [], len(p)
            for t, a in enumerate(p):
                r, done = self.step(a)
                out[i] += g * r
                g *= discount
                n_steps[i] = t + 1
                if t >= T - n_last:
                    fr.append(render_fn())
                if done:
                    break
            frames.append(fr)
        self.restore(base)
        if not self.same(base):
            raise RuntimeError("sim_planner: env state not restored exactly")
        return out, frames, n_steps

    def rollout_frames(self, plan, render_fn):
        """Execute `plan` (K, real_dim), render after each step with
        render_fn(), restore. Returns (frames list, rewards)."""
        base = self.save()
        frames, rews = [], []
        for a in plan:
            r, done = self.step(a)
            frames.append(render_fn())
            rews.append(r)
            if done:
                break
        self.restore(base)
        if not self.same(base):
            raise RuntimeError("sim_planner: env state not restored exactly")
        return frames, rews


def to_env_actions(plans, act_mask_1d, real_dim):
    """(N, H, act_dim) model-space plans -> (N, H, real_dim) numpy env actions,
    exactly as plan_mpc_env executes them (pad to 16, mask, slice)."""
    N, H, A = plans.shape
    a16 = torch.zeros((N, H, 16), device=plans.device, dtype=torch.float32)
    a16[..., :A] = plans
    a16 = a16 * act_mask_1d.view(1, 1, 16)
    return a16[..., :real_dim].cpu().numpy().astype(np.float32)


def sim_cem_plan(sim, *, horizon, act_dim, act_mask_1d, real_dim, n_samples, n_elites, n_iters,
                 init_std, min_std, noise_beta, device, generator=None, init_mu=None, discount=1.0,
                 score_fn=None):
    """plan_cem.cem_plan with the simulator as the model and the true reward
    as the objective: same sampling (colored_noise from `generator`), elite
    update and returned plan (the best sample seen). Deterministic sim, so
    one rollout per candidate. score_fn(samples (N, H, act_dim) model space)
    -> (N,) objective replaces the true return (e.g. return + terminal value)."""
    if init_mu is not None:
        mu = init_mu.detach().to(device=device, dtype=torch.float32).clamp(-1.0, 1.0).clone()
    else:
        mu = torch.zeros((horizon, act_dim), device=device, dtype=torch.float32)
    std = torch.full((horizon, act_dim), float(init_std), device=device, dtype=torch.float32)
    best_cost, best_plan, history = float("inf"), mu.clone(), []
    for it in range(n_iters):
        noise = colored_noise(n_samples, horizon, act_dim, float(noise_beta), device=device, generator=generator)
        samples = (mu + std * noise).clamp(-1.0, 1.0)
        R = (score_fn(samples) if score_fn is not None
             else sim.returns(to_env_actions(samples, act_mask_1d, real_dim), discount=discount))
        costs = -torch.as_tensor(R, device=device, dtype=torch.float32)
        elite_idx = torch.topk(costs, k=n_elites, largest=False).indices
        elites = samples[elite_idx]
        if float(costs[elite_idx[0]]) < best_cost:
            best_cost = float(costs[elite_idx[0]])
            best_plan = samples[elite_idx[0]].clone()
        history.append({"iter": it, "mean_cost": float(costs.mean()), "best_cost": float(costs[elite_idx[0]])})
        mu, std = elites.mean(dim=0), elites.std(dim=0).clamp_min(float(min_std))
    return {"best_plan": best_plan, "best_cost": best_cost, "mu": mu, "std": std, "history": history}


# ------------------------------------------------------------------ self-test

def _selftest(task, n_steps=40, seed=0):
    """Restore must be exact and transparent. One env: from a saved start,
    (1) run a reference trajectory; (2) rewind and rerun it with a planning
    burst (random rollouts between save/restore) before every step: rewards
    and MuJoCo states must be bit-identical to (1); (3) returns() of a plan
    must equal the rewards of executing it. Two env instances cannot be
    compared directly: Meta-World draws its goal from a global RNG at reset."""
    from env_wrapper import _EnvCfg
    from envs import make_env
    rng = np.random.default_rng(seed)
    cfg = _EnvCfg(task, seed=seed)
    cfg.obs = "state"
    env = make_env(cfg)
    env.reset()
    adim = cfg.action_dim
    h = SimHandle(env)
    acts = rng.uniform(-1, 1, (n_steps, adim)).astype(np.float32)
    start = h.save()
    ref_r, ref_s = [], []
    for a in acts:
        ref_r.append(float(env.step(a)[1]))
        ref_s.append(h.mj_state())
    h.restore(start)
    ok_traj, max_dr = True, 0.0
    for t, a in enumerate(acts):
        h.returns(rng.uniform(-1, 1, (4, 16, adim)).astype(np.float32))
        r = float(env.step(a)[1])
        max_dr = max(max_dr, abs(r - ref_r[t]))
        ok_traj &= np.array_equal(h.mj_state(), ref_s[t])
    plan = rng.uniform(-1, 1, (12, adim)).astype(np.float32)
    pred = h.returns(plan[None])[0]
    real = sum(float(env.step(a)[1]) for a in plan)
    ok = ok_traj and max_dr == 0.0 and abs(pred - real) < 1e-9
    print(f"{task:20s} kind={h.kind:3s} states identical with planning bursts: {ok_traj} "
          f"(max |dr| {max_dr:.2e}, ref return {sum(ref_r):.3f}) | returns() {pred:.6f} vs executed {real:.6f} "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["reacher-easy", "cup-catch", "mw-reach"])
    a = p.parse_args()
    res = [_selftest(t) for t in a.tasks]
    raise SystemExit(0 if all(res) else 1)
