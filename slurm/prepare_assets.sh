#!/bin/bash
set -euo pipefail
PROJECT="${TENT_PROJECT_DIR:-$HOME/deep_learning/src/RRv2}"
ENV_DIR="${TENT_ENV_DIR:-$HOME/deep_learning/env_tent}"
DATA_DIR="${TENT_DATA_DIR:-$HOME/deep_learning/data/tent}"
MODEL_DIR="${TENT_MODEL_DIR:-$HOME/deep_learning/models/tent}"
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
python "$PROJECT/scripts/prepare_assets.py" --data-dir "$DATA_DIR" --model-dir "$MODEL_DIR"
python "$PROJECT/scripts/preflight.py" --data-dir "$DATA_DIR" --model-dir "$MODEL_DIR" \
  --output "$PROJECT/results/preparation.json"
