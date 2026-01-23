#!/usr/bin/env bash
set -euo pipefail

# ---------
# Config
# ---------
MODEL_TAG="small"
CONTEXT_LEN=128

D_MODEL=768
D_FF=3072
NUM_LAYERS=12
NUM_HEADS=12

MODE="train_step"
WARM_UP=5
NSTEPS=1
DEVICE=cuda
DTYPE=fp32

OUT_DIR="results/nsys"
OUT="${OUT_DIR}/${MODEL_TAG}_ctx${CONTEXT_LEN}_${MODE}"

mkdir -p "${OUT_DIR}"

echo "Profiling model=${MODEL_TAG}, ctx=${CONTEXT_LEN}"
echo "Output: ${OUT}.nsys-rep"

# ---------
# Run nsys
# ---------
nsys profile \
  --trace=cuda,nvtx \
  --pytorch=autograd-nvtx \
  --force-overwrite=true \
  -o "${OUT}" \
  uv run python cs336_systems/nsys_profile.py \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --warm-up "${WARM_UP}" \
    --nsteps "${NSTEPS}" \
    --model-tag "${MODEL_TAG}" \
    --mode "${MODE}" \
    --context-len "${CONTEXT_LEN}" \
    --d-model "${D_MODEL}" \
    --d-ff "${D_FF}" \
    --num-layers "${NUM_LAYERS}" \
    --num-heads "${NUM_HEADS}"