#!/bin/bash
#SBATCH --job-name=tent-official-v2
#SBATCH --output=logs/official-v2-%j.out
#SBATCH --error=logs/official-v2-%j.err
#SBATCH --partition=gpu4090
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
set -euo pipefail
PROJECT="${TENT_PROJECT_DIR:-$HOME/deep_learning/src/RRv2}"
ENV_DIR="${TENT_ENV_DIR:-$HOME/deep_learning/env_tent}"
DATA_DIR="${TENT_DATA_DIR:-$HOME/deep_learning/data/tent}"
MODEL_DIR="${TENT_MODEL_DIR:-$HOME/deep_learning/models/tent}"
RUN_BATCH128="${TENT_RUN_BATCH128:-0}"
RUN_DIR="$PROJECT/results/job-${SLURM_JOB_ID}"
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="$RUN_DIR/plot-cache"
mkdir -p "$PROJECT/results"
mkdir "$RUN_DIR"
echo "Started: $(date -Iseconds)"
echo "Job: $SLURM_JOB_ID; results: $RUN_DIR"
record_exit() {
  local exit_code=$?
  echo "Job $SLURM_JOB_ID exited with status $exit_code at $(date -Iseconds)"
}
trap record_exit EXIT
# NVML failure does not by itself establish that PyTorch CUDA is unusable.
if ! nvidia-smi; then
  echo "WARNING: nvidia-smi failed; checking CUDA execution with PyTorch next." >&2
fi
preflight_args=(--require-cuda --data-dir "$DATA_DIR" --model-dir "$MODEL_DIR"
  --output "$RUN_DIR/environment.json")
batches=(200)
if [[ "$RUN_BATCH128" == "1" ]]; then
  preflight_args+=(--batch128)
  batches+=(128)
fi
python "$PROJECT/scripts/preflight.py" "${preflight_args[@]}"
python -m pip freeze > "$RUN_DIR/pip-freeze.txt"
for batch in "${batches[@]}"; do
  for method in source norm tent; do
    python "$PROJECT/scripts/run_experiment.py" --method "$method" --batch-size "$batch" \
      --data-dir "$DATA_DIR" --model-dir "$MODEL_DIR" \
      --run-dir "$RUN_DIR/runs/b${batch}/$method"
  done
done
python "$PROJECT/scripts/aggregate_results.py" --results-dir "$RUN_DIR"
python "$PROJECT/scripts/build_report.py" --results-dir "$RUN_DIR" --output-dir "$RUN_DIR/report"
echo "Finished: $(date -Iseconds)"
echo "Package command: bash scripts/package_results.sh $SLURM_JOB_ID"
