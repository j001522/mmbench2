# Contributor Guide

Orientation for contributors and AI coding assistants working on this repository — the code release for the paper *"Hallucination in World Models is Predictable and Preventable"* (the **MMBench2** world model, dataset tooling, and hallucination detection/mitigation method). For installation and usage, see `README.md`.

## What this repository contains
A Dreamer-4-style generative **world model** for large-scale multi-task continuous control (~350M parameters: ≈100M tokenizer + ≈250M dynamics), the **MMBench2** dataset tooling (210 tasks across 10 domains, 224×224 RGB observations, 1–16 action dimensions), and the paper's method: hallucination predictors, coverage-aware training, and curiosity-driven targeted data collection.

## Code layout
All code lives under `src/`:
- `model.py` — shared model components (tokenizer encoder/decoder, dynamics transformer, reward/BC heads); imported by both training scripts.
- `train_tokenizer.py` / `train_dynamics.py` — the two training stages (a causal video tokenizer, then a block-causal flow-matching dynamics model with reward and behavior-cloning heads).
- `uncertainty.py` — the hallucination predictors `u_r` / `u_f` / `u_s` (motion-normalized).
- `collect_data.py` / `curiosity.py` — curiosity-driven targeted data collection.
- `plan_cem.py` — evaluation via CEM planning; `--cost reward_value` adds a learned terminal value `gamma^H V(h_H)` to the predicted return.
- `plan_mpc_env.py` — closed-loop MPC in the real environment (CEM, `--policy bc`, or `--policy random`); writes per-episode JSON, a summary CSV, and a real | world-model side-by-side video. Investigation extras: `--compare_sim` (true-model CEM via `sim_planner.py`, regret), `--halluc` (`plan_halluc.py` chunk labels), `--log_unc` (u_r/u_f/u_s per replan).
- `train_heads_offline.py` — fits new heads on the agent tokens `h_t` of the FROZEN world model (asserted at startup): a value head (symexp two-hot, TD(λ) targets from real rewards, γ=0.99 (paper: 0.997), λ=0.95, as dreamer_v4 phase 3) and a stochastic BC head (`--policy_type gaussian|categorical`, `--bc_L 1` = next action only). Val reports BC NLL/MSE and the value head against real discounted returns.
- `heads_rl.py` — the value / Gaussian / categorical policy heads and their checkpoint loaders; `rl_algorithms.py` — λ-returns and the value loss (`test_rl_algorithms.py`).
- Investigation (hallucination predictors / masks / metrics / plots): `hallucination_labels.py`, `mask_metrics.py`, `paper_predictors.py`, `roc_*.py`, `predictor_cost*.py`, `plan_regret.py`, `plan_figures.py`, `plot_unc_episode.py`, `dino_*`/`clef_*`/`vlm_judge_metrics.py`, `eval_annotations.py`, `export_annotation_set.py`, `metric_io.py`.
- `interactive.py` (+ `interactive.html`) — the browser interface for open-ended interaction with the model.
- `wm_dataset.py`, `sharded_frame_dataset.py`, `preprocess_dataset.py`, `task_set.py` — dataset loading and preprocessing.
- `envs/` — Gymnasium-compatible simulators for every task.
- `download_dataset.py` / `download_checkpoints.py` — fetch the dataset and pretrained checkpoints from the Hugging Face Hub.

## Dataset / observation convention (load-bearing)
For a trajectory of length `T+1` there are `T+1` observations but only `T` actions and `T` rewards. The first tuple is `(obs_0, nan, nan)` (no action or reward precedes the initial observation); thereafter `(obs_t, action_{t-1}, reward_{t-1})`, where `action_{t-1}` and `reward_{t-1}` describe the transition `obs_{t-1} → obs_t`. Keep this convention in mind for any change to data loading or preprocessing.

## Conventions
- The training scripts use flat imports and must be run from inside `src/`.
- Training is expensive — prefer minimal, well-justified changes, and verify with smoke tests when a change could affect convergence or correctness.
- Add comments only where they clarify non-obvious logic; use American English in docs and comments.

See `CONTRIBUTING.md` for how to propose changes, and `README.md` for installation, training, evaluation, and the interactive interface.
