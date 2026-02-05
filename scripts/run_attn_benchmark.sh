#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${ROOT}/cs336_systems/attn_benchmark.py"
OUT="${ROOT}/results/attn/attn_benchmark.jsonl"
mkdir -p "${ROOT}/results/attn"
: > "$OUT"

D_MODELS=(16 32 64 128)
SEQ_LENS=(256 1024 4096 8192 16384)

WARM_UP=10
NSTEPS=100

for d in "${D_MODELS[@]}"; do
  for t in "${SEQ_LENS[@]}"; do
    echo "Running d_model=$d context_len=$t"
    uv run python "$SCRIPT" \
      --d-model "$d" \
      --context-len "$t" \
      --warm-up "$WARM_UP" \
      --nsteps "$NSTEPS" \
      --output "$OUT" || true
  done
done