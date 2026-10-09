#!/usr/bin/env bash
#SBATCH --job-name=mmb2-preprocess
#SBATCH --account=tdsei8531
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/preprocess-%j.out
#
# genoa/rome are covered by the L4 CPU budget (separate from the GPU budget).
# Memory, not cores, is the binding constraint: each process peaks near 3 GB
# (concat buffer + its sources + the save-time clone), so 12 in parallel need
# ~40 GB and the 1792 MiB/core default is not enough. Swap the partition for
# rome if genoa's queue is busier.
#
# Preprocess the raw MMBench2 PNG partitions into sharded .pt frames — one
# preprocess_dataset.py run per partition found under $DATA_DIR, all launched
# in parallel (CPU-only work; no GPU needed).
#
# Usage:
#   sbatch preprocess_slurm.sh                        # all partitions, task(s) = $TASKS (default cup-catch)
#   sbatch preprocess_slurm.sh walker-run cheetah-run # explicit task list
#   SPLITS="expert val" sbatch preprocess_slurm.sh cup-catch
#
# Env vars:
#   DATA_DIR  raw data root                (default: ./data)
#   SPLITS    partitions to process        (default: every non-*-shards dir under DATA_DIR)
#   WORKERS   preprocess_dataset workers   (default: 1 — see note below)
#   PY        python interpreter           (default: the mmbench2 conda env)
#
# Note: parallelism is across partitions (one process each), not inside
# preprocess_dataset.py. Its internal mp.Pool deadlocked once on a GPU node, and
# with WORKERS=1 the script takes the in-process path instead. Raise WORKERS only
# when a single partition holds many tasks.
set -euo pipefail
# Run from src/ (scripts use flat imports). Under sbatch, $0 is the spooled copy
# of this script, so the submit dir is the only reliable anchor.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs/slurm

PY="${PY:-/projects/0/prjs0951/conda_envs/mmbench2/bin/python}"
DATA_DIR="${DATA_DIR:-./data}"
WORKERS="${WORKERS:-1}"
TASKS=("${@:-cup-catch}")

# Partitions: explicit via $SPLITS, else every raw partition dir (skip our own output).
if [[ -n "${SPLITS:-}" ]]; then
    read -r -a splits <<< "$SPLITS"
else
    splits=()
    for d in "$DATA_DIR"/*/; do
        name="$(basename "$d")"
        [[ "$name" == *-shards ]] && continue
        compgen -G "$d"/*.pt > /dev/null || continue     # raw partitions carry <task>.pt
        splits+=("$name")
    done
fi

if [[ ${#splits[@]} -eq 0 ]]; then
    echo "No raw partitions found under $DATA_DIR" >&2
    exit 1
fi

echo "python:     $PY"
echo "tasks:      ${TASKS[*]}"
echo "partitions: ${splits[*]}"
echo "workers:    $WORKERS per partition, ${#splits[@]} partitions in parallel"
echo

# One torch thread per process: many processes on one node, each is I/O bound.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

start=$SECONDS
pids=()
for split in "${splits[@]}"; do
    log="logs/slurm/preprocess-${SLURM_JOB_ID:-local}-${split}.log"
    (
        "$PY" preprocess_dataset.py \
            --filedir "$DATA_DIR/$split" \
            --outdir "$DATA_DIR/${split}-shards" \
            --tasks "${TASKS[@]}" \
            --num_workers "$WORKERS"
    ) > "$log" 2>&1 &
    pids+=("$!:$split")
done

status=0
for entry in "${pids[@]}"; do
    pid="${entry%%:*}"; split="${entry#*:}"
    if wait "$pid"; then
        echo "[OK]   $split"
    else
        echo "[FAIL] $split  (see logs/slurm/preprocess-${SLURM_JOB_ID:-local}-${split}.log)"
        status=1
    fi
done

echo
echo "elapsed: $((SECONDS - start))s"
du -sh "$DATA_DIR"/*-shards 2>/dev/null || true
exit $status
