#!/usr/bin/env bash
#SBATCH --job-name=mmb2-bench
#SBATCH --account=tdsei8531
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/bench-%j.out
#
# Short timing run: train for $STEPS steps and report seconds/step, so a full
# run can be extrapolated before committing GPU budget. Trains nothing useful —
# checkpoints and evals are disabled, wandb runs offline.
#
# Usage:
#   sbatch bench_train_slurm.sh tokenizer          # default
#   sbatch bench_train_slurm.sh dynamics
#   STEPS=400 GPUS=1 sbatch bench_train_slurm.sh tokenizer
#   STEPS=100 BOOT=50 sbatch bench_train_slurm.sh dynamics   # both bootstrap regimes in one run
#   sbatch -p gpu_a100 bench_train_slurm.sh tokenizer   # the other GPU partition
#
# 15 min is deliberate: 200 steps take a few minutes, and a short request
# backfills sooner. Raise --time only if STEPS is raised well past 200.
#
# Env vars:
#   STEPS      steps to run                 (default: 200)
#   GPUS       GPUs to use                  (default: $SLURM_GPUS_ON_NODE, else 1)
#   BATCH      per-GPU batch size           (default: the script's own default)
#   DATA       shard dir for the tokenizer  (default: ./data/expert-shards)
#   RAW        raw dir for the dynamics     (default: ./data/expert)
#   TOKENIZER  tokenizer ckpt for dynamics  (default: ./checkpoints/combined/tokenizer.pt)
#   BOOT       dynamics --bootstrap_start   (default: 50, i.e. inside the window)
#   PY         python interpreter           (default: the mmbench2 conda env)
#
# The first ~20 steps include cudnn autotuning and dataloader warmup, so read the
# LAST printed s/step, not the first.
set -euo pipefail
# Run from src/ (scripts use flat imports). Under sbatch, $0 is the spooled copy
# of this script, so the submit dir is the only reliable anchor.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs/slurm

STAGE="${1:-tokenizer}"
STEPS="${STEPS:-200}"
GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
PY="${PY:-/projects/0/prjs0951/conda_envs/mmbench2/bin/python}"
DATA="${DATA:-./data/expert-shards}"
RAW="${RAW:-./data/expert}"
TOKENIZER="${TOKENIZER:-./checkpoints/combined/tokenizer.pt}"

# Keep the benchmark self-contained: no cloud logging, no checkpoints on disk.
export WANDB_MODE=offline
export WANDB_SILENT=true
# LPIPS (tokenizer stage) pulls a 244 MB AlexNet through torch.hub; keep it out
# of $HOME, where every rank races to write it.
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch_cache}"
# --cpus-per-gpu leaves SLURM_CPUS_PER_TASK unset; SLURM_CPUS_ON_NODE is the total.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-8}} / GPUS ))
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "stage=$STAGE steps=$STEPS gpus=$GPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

common=(--max_steps "$STEPS" --log_every 25 --save_every 0 --seed 0)

case "$STAGE" in
  tokenizer)
      # --print_every is a tokenizer-only flag; train_dynamics.py prints on --log_every.
      args=(--data_dirs "$DATA" --val_every 0 --viz_every 0 --print_every 25 "${common[@]}")
      [[ -n "${BATCH:-}" ]] && args+=(--batch_size "$BATCH")
      set -x
      "$PY" -m torch.distributed.run --nproc_per_node="$GPUS" --standalone \
          train_tokenizer.py "${args[@]}"
      ;;
  dynamics)
      # Needs a tokenizer: the released one works for timing, the architecture is
      # what matters here, not the weights.
      # $BOOT moves the self-bootstrap objective (two extra no-grad forwards over
      # --self_fraction of the batch) into the benchmark window; upstream starts
      # it at step 10,000, which a short run would never reach. Setting it below
      # $STEPS means the log shows both regimes: cheap steps before $BOOT, the
      # real steady-state cost after.
      args=(--data_dirs "$RAW" --frame_dirs "$DATA" --tokenizer_ckpt "$TOKENIZER"
            --val_every 0 --eval_every 0 --action_attr_n_batches 0
            --bootstrap_start "${BOOT:-50}"
            --resume ./logs/dynamics_ckpts/latest.pt "${common[@]}")
      [[ -n "${BATCH:-}" ]] && args+=(--batch_size "$BATCH")
      set -x
      "$PY" -m torch.distributed.run --nproc_per_node="$GPUS" --standalone \
          train_dynamics.py "${args[@]}"
      ;;
  *)
      echo "unknown stage '$STAGE' (expected: tokenizer | dynamics)" >&2
      exit 2
      ;;
esac
