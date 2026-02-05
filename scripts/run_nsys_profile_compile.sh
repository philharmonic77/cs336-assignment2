#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT}/cs336_systems/nsys_profile.py"
OUT="${ROOT}/results/nsys/model_benchmark_mix_precision_compile.jsonl"
mkdir -p "${ROOT}/results/nsys"
: > "$OUT"

WARM_UP=5
NSTEPS=10
DEVICE=cuda
DTYPE=fp32

CTX_LENS=(128 256 512 1024)
MODES=(forward_only train_step)
COMPILE_FLAGS=(0 1)

MODELS=(
  "small  768   3072   12  12"
  "medium 1024  4096   24  16"
  "large  1280  5120   36  20"
  "xl     1600  6400   48  25"
  "2.7B   2560  10240  32  32"
)

run_one () {
  local tag="$1" ctx="$2" mode="$3" d="$4" ff="$5" L="$6" h="$7" use_compile="$8"
  echo "Run: ${tag} ctx=${ctx} mode=${mode}"
  uv run python "${SCRIPT}" \
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
    --output "${OUT}" \
    --use-bf16 \
    $([[ "${use_compile}" -eq 1 ]] && printf "%s" "--use-torch-compile")
}

for line in "${MODELS[@]}"; do
  read -r tag d ff L h <<< "${line}"
  for ctx in "${CTX_LENS[@]}"; do
    for mode in "${MODES[@]}"; do
      for use_compile in "${COMPILE_FLAGS[@]}"; do
        run_one "${tag}" "${ctx}" "${mode}" "${d}" "${ff}" "${L}" "${h}" "${use_compile}"
      done
    done
  done
done

echo "Done."
