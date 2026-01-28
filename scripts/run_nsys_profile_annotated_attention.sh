#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT}/cs336_systems/nsys_profile.py"


WARM_UP=5
NSTEPS=10
DEVICE=cuda
DTYPE=fp32

TAG="large"
D_MODEL=1280
D_FF=5120
NUM_LAYERS=36
NUM_HEADS=20
CTX_LEN=256
MODE=forward_only

BASE="${ROOT}/results/nsys/${TAG}_ctx${CTX_LEN}_${MODE}_annotated"
REP="${BASE}.nsys-rep"

echo "Run: ${TAG} ctx=${CTX_LEN} mode=${MODE}"

mkdir -p "${ROOT}/results/nsys"

nsys profile \
  --trace=cuda,nvtx \
  --pytorch=autograd-nvtx \
  --force-overwrite=true \
  -o "${BASE}" \
  uv run python "${PY_SCRIPT}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --warm-up "${WARM_UP}" \
    --nsteps "${NSTEPS}" \
    --model-tag "${TAG}" \
    --mode "${MODE}" \
    --context-len "${CTX_LEN}" \
    --d-model "${D_MODEL}" \
    --d-ff "${D_FF}" \
    --num-layers "${NUM_LAYERS}" \
    --num-heads "${NUM_HEADS}" \
    --annotated
