#!/usr/bin/env bash
# Quantise a BF16 checkpoint to the parent's NVFP4 mixed-precision plan.
set -euo pipefail
cd "$(dirname "$0")/.."
QUANT_VENV=${QUANT_VENV:-/mnt/nvme8tb/hf-ablate/.venv-quant}
SRC=${1:?usage: quantize.sh SRC_DIR OUT_DIR}
OUT=${2:?}
./scripts/free_gpu.sh
PYTHONPATH="$PWD" "$QUANT_VENV/bin/python" -m groot_ablit.quantize "$SRC" "$OUT"
