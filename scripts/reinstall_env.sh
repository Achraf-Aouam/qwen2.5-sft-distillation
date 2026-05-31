#!/usr/bin/env bash
set -euo pipefail

python -m pip install unsloth
python -m pip uninstall unsloth -y
python -m pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
python -m pip install wandb
python -m pip install pyarrow

python -m pip check || true

echo
echo "Environment ready."
echo "Run: python train.py --config train_config.toml"
