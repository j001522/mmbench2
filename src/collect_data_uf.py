# collect_data_uf.py
"""
Curiosity-driven data collection with RAW flow instability u_f as the CEM
objective -- a parallel variant of `collect_data.py --policy curiosity_u_r_norm`.

Everything except the score is the released code path, reused as-is: env loop,
receding-horizon CEM (curiosity._cem_multi_step / _rollout_scores, incl. the
0.5*mean + 0.5*max aggregation), context handling, saving, and stats. Two
things are swapped, only inside this process:

  1. The sampler used by curiosity.py / uncertainty.py is replaced by
     interactive.sample_one_timestep_packed(per_sample_instability=True). It is
     the same Euler/shortcut sampler (same RNG order, same output) but also
     returns per-sample u_f = mean RMS change of x1_hat over the tail half of the
     denoising steps. The value is stashed for the scorer.
  2. The scorer returns that stashed u_f (mean over the N seeds per candidate).
     No decode / re-encode in the CEM loop. --n_samples_mpc is independent of
     the scorer (u_r_norm also runs at N=1: its "seed mean" is then the single
     sample); keep it equal to the u_r run's (released: 2) for a clean comparison.

u_f is used RAW (no motion normalization): with the per-step normalization of
URNormScorer, u_f_norm ~ 1/motion, so maximizing it would favor static plans.
A per-task constant normalization (the paper's text) gives the same CEM ranking
as raw u_f.

Per executed env step it logs (in <task>_uf_extra.json, aligned with frames
1..N-1): u_f (scoring signal), u_r, motion, u_r_norm (one decode/encode per env
step, not per candidate), next to collect_data's prediction_error. The
`uncertainty_timeseries` in <task>_collect_stats.json is u_f here.

Usage (from src/):
  python collect_data_uf.py --tasks mw-push --n_episodes 50 --n_samples_mpc 1 \\
    --tokenizer_ckpt ./checkpoints/coverage_aware/tokenizer.pt \\
    --dynamics_ckpt  ./checkpoints/coverage_aware/dynamics.pt \\
    --out_data_dir ./data/collected/curiosity_u_f
"""
import json
import sys
from pathlib import Path

import torch

import collect_data as CD
import curiosity
import uncertainty
from interactive import sample_one_timestep_packed as _sample_with_inst

POLICY = "curiosity_u_f"

# ---- 1. sampler that also returns per-sample u_f --------------------------------
_STASH = {}


def _sampler_stash_uf(dyn, **kw):
    z_next, inst = _sample_with_inst(dyn, per_sample_instability=True, **kw)
    _STASH["u_f"] = inst.float()                                   # (B,) B = K*N
    return z_next


curiosity.sample_one_timestep_packed = _sampler_stash_uf          # CEM rollouts
uncertainty.sample_one_timestep_packed = _sampler_stash_uf        # per-step logging call


# ---- 2. scorer ------------------------------------------------------------------
class UFScorer:
    """CEM score = raw u_f of the sample(s) just drawn for these K candidates."""

    def __init__(self, ur_scorer: uncertainty.URNormScorer):
        self.ur = ur_scorer
        self.log = []          # one dict per executed env step

    def _u_f(self, predictions_KN):
        K, N = predictions_KN.shape[:2]
        u_f = _STASH.pop("u_f")
        assert u_f.numel() == K * N, (u_f.shape, K, N)
        return u_f.reshape(K, N).mean(1)                              # (K,)

    @torch.no_grad()
    def __call__(self, predictions_KN, z_prev_K):
        return self._u_f(predictions_KN)

    @torch.no_grad()
    def score_components(self, predictions_KN, z_prev_K):
        # Only called by collect_data for the per-env-step log (K=1). collect_data
        # records the "u_r_norm" key as the step's uncertainty; here that is u_f.
        u_f = self._u_f(predictions_KN)
        comps = self.ur.score_components(predictions_KN, z_prev_K)
        self.log.append({"u_f": float(u_f[0]), "u_r": float(comps["u_r"][0]),
                         "motion": float(comps["motion"][0]),
                         "u_r_norm": float(comps["u_r_norm"][0])})
        return {"u_r_norm": u_f}


_orig_build_scorer = CD._build_scorer


def _build_scorer(policy, **kw):
    assert policy == POLICY, policy
    return UFScorer(_orig_build_scorer("curiosity_u_r_norm", **kw))


CD._build_scorer = _build_scorer
CD.WM_POLICIES.add(POLICY)
CD.ALL_POLICIES.add(POLICY)

# ---- 3. episode / task wrappers (released MPC branch, extra per-step log) ---------
_orig_episode = CD.collect_episode_with_uncertainty
_orig_run_task = CD.run_task
_EXTRA = []


def _episode(**kw):
    scorer = kw["scorer"]
    i0 = len(scorer.log)
    kw["policy"] = "curiosity_u_r_norm"        # selects collect_data's CEM-MPC branch; the scorer is u_f
    ep = _orig_episode(**kw)
    steps = scorer.log[i0:]
    assert len(steps) == ep["frames"].shape[0] - 1, (len(steps), ep["frames"].shape)
    _EXTRA.append({"episode_id": ep["episode_id"],
                   "replan_time_s": [r["time_s"] for r in ep["replan_log"]],
                   **{k: [s[k] for s in steps] for k in ("u_f", "u_r", "motion", "u_r_norm")}})
    return ep


def _run_task(*, task, args, **kw):
    _EXTRA.clear()
    _orig_run_task(task=task, args=args, **kw)
    out = Path(args.out_data_dir) / f"{task}_uf_extra.json"
    with open(out, "w") as f:
        json.dump({"task": task, "policy": POLICY, "n_samples_mpc": args.n_samples_mpc,
                   "eval_d": args.eval_d, "episodes": _EXTRA}, f)
    print(f"[collect_uf] wrote {out}")


CD.collect_episode_with_uncertainty = _episode
CD.run_task = _run_task


if __name__ == "__main__":
    # Reuse collect_data's CLI verbatim, with the policy pinned to curiosity_u_f.
    if "--policy" in sys.argv:
        sys.exit("collect_data_uf.py: --policy is fixed to curiosity_u_f")
    sys.argv += ["--policy", POLICY]
    src = Path(CD.__file__).read_text()
    main_block = src[src.index('if __name__ == "__main__":'):]
    exec(compile(main_block.replace('if __name__ == "__main__":', "if True:", 1),
                 CD.__file__, "exec"), vars(CD))
