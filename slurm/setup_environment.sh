#!/bin/bash
set -euo pipefail
PROJECT="${TENT_PROJECT_DIR:-$HOME/deep_learning/src/RRv2}"
ENV_DIR="${TENT_ENV_DIR:-$HOME/deep_learning/env_tent}"
if [[ -x "$ENV_DIR/bin/python" ]]; then
  source "$ENV_DIR/bin/activate"
  python -m pip install -r "$PROJECT/requirements-extra.txt"
else
  python3 -m venv "$ENV_DIR"
  source "$ENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r "$PROJECT/requirements-cluster.txt"
  python -m pip install --force-reinstall --no-deps \
    'git+https://github.com/fra31/auto-attack.git@a39220048b3c9f2cca9a4d3a54604793c68eca7e'
fi
export PYTHONDONTWRITEBYTECODE=1
python -c 'import torch, robustbench, yacs, iopath; from autoattack import AutoAttack; print("torch:", torch.__version__, "CUDA build:", torch.version.cuda); print("Official dependencies imported")'
echo "Environment ready: $ENV_DIR"
