#!/usr/bin/env bash
set -euo pipefail
# --------------------------------------------------
# 1) Resolve repo root
# --------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --------------------------------------------------
# 2) Output paths
# --------------------------------------------------
RESULTS_JSONL="${ROOT}/results/first_benchmark.jsonl"
rm -f "${RESULTS_JSONL}"

# --------------------------------------------------
# 3) Define params
# --------------------------------------------------

WARM_UP=5
NSTEPS=10
CONTEXT_LEN=128
VOCAB_SIZE=10000
BATCH_SIZE=4
DEVICE=auto
DTYPE=fp32

run_one () {
  local model_tag="$1"
  local mode="$2"
  local d_model="$3"
  local d_ff="$4"
  local num_layers="$5"
  local num_heads="$6"

  echo "Running: ${model_tag} mode=${mode} L=${num_layers} d=${d_model} h=${num_heads} ff=${d_ff}"

  python "${ROOT}/benchmark.py" \
    --model-tag "${model_tag}" \
    --mode "${mode}" \
    --warm-up "${WARM_UP}" \
    --nsteps "${NSTEPS}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --context-len "${CONTEXT_LEN}" \
    --vocab-size "${VOCAB_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --d-model "${d_model}" \
    --d-ff "${d_ff}" \
    --num-layers "${num_layers}" \
    --num-heads "${num_heads}" \
    --output "${RESULTS_JSONL}"
}

# --------------------------------------------------
# 4) Run
# --------------------------------------------------

run_one "small"     "forward"          768   3072   12  12
run_one "small"     "forward_backward" 768   3072   12  12

run_one "medium"    "forward"          1024  4096   24  16
run_one "medium"    "forward_backward" 1024  4096   24  16

run_one "large"     "forward"          1280  5120   36  20
run_one "large"     "forward_backward" 1280  5120   36  20

run_one "xl"        "forward"          1600  6400   48  25
run_one "xl"        "forward_backward" 1600  6400   48  25

run_one "2.7B"      "forward"          2560  10240  32  32
run_one "2.7B"      "forward_backward" 2560  10240  32  32


echo "Run finished. Results written to:"
echo "  ${RESULTS_JSONL}"