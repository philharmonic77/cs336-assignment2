#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT}/cs336_systems/nsys_profile.py"

OUT_DIR="${ROOT}/results/nsys"

mkdir -p "${OUT_DIR}"

WARM_UP=0
NSTEPS=3
DEVICE=cuda
DTYPE=fp32

CTX_LENS=(128 256 512)
BATCH_SIZE=1 
MODES=(forward_only train_step)

MODELS=(
  "large  1280  5120   36  20"
)

run_one () {
  local tag="$1" ctx="$2" mode="$3" d="$4" ff="$5" L="$6" h="$7"

  echo "Run: ${tag} ctx=${ctx} mode=${mode}"

  uv run python "${PY_SCRIPT}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --warm-up "${WARM_UP}" \
    --nsteps "${NSTEPS}" \
    --model-tag "${tag}" \
    --mode "${mode}" \
    --context-len "${ctx}" \
    --d-model "${d}" \
    --d-ff "${ff}" \
    --num-layers "${L}" \
    --num-heads "${h}" \
    --batch-size "${BATCH_SIZE}" \
    --mem-profile \
    --use-bf16 \
}

for line in "${MODELS[@]}"; do
  read -r tag d ff L h <<< "${line}"
  for ctx in "${CTX_LENS[@]}"; do
    for mode in "${MODES[@]}"; do
      run_one "${tag}" "${ctx}" "${mode}" "${d}" "${ff}" "${L}" "${h}"
    done
  done
done

echo "Done."
