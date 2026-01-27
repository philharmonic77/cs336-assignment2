#!/usr/bin/env bash
set -u
set -o pipefail

# ----------------------------
# Paths
# ----------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT}/cs336_systems/nsys_profile.py"

OUT_DIR="${ROOT}/results/nsys"
LOG_JSONL="${OUT_DIR}/runs.jsonl"
mkdir -p "${OUT_DIR}"
: > "${LOG_JSONL}"

# ----------------------------
# Global params
# ----------------------------
WARM_UP=5
NSTEPS=1
DEVICE=cuda
DTYPE=fp32

CTX_LENS=(128 256 512 1024)
MODES=(forward_only train_step)

# tag d_model d_ff num_layers num_heads
MODELS=(
  "small  768   3072   12  12"
  "medium 1024  4096   24  16"
  "large  1280  5120   36  20"
  "xl     1600  6400   48  25"
  "2.7B   2560  10240  32  32"
)

# ----------------------------
# One run
# ----------------------------
run_one () {
  local tag="$1" ctx="$2" mode="$3" d="$4" ff="$5" L="$6" h="$7"
  local base="${OUT_DIR}/${tag}_ctx${ctx}_${mode}"
  local rep="${base}.nsys-rep"

  echo "Run: ${tag} ctx=${ctx} mode=${mode}"
  echo "  -> ${rep}"

  # Run and capture stdout/stderr to detect OOM and exit code.
  local out
  out="$(
    set +e
    nsys profile \
      --trace=cuda,nvtx \
      --pytorch=autograd-nvtx \
      --sample=none \
      --cpuctxsw=none \
      --force-overwrite=true \
      -o "${base}" \
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
      2>&1
    echo "<<<EXIT:$?>>>"
  )"

  local code
  code="$(printf "%s" "${out}" | sed -n 's/.*<<<EXIT:\([0-9]\+\)>>>.*/\1/p')"
  if [[ -z "${code}" ]]; then code=99; fi

  local oom=false
  if printf "%s" "${out}" | grep -qiE "out of memory|cuda out of memory|OutOfMemoryError"; then
    oom=true
  fi

  local rep_exists=false
  if [[ -f "${rep}" ]]; then rep_exists=true; fi

  local ok=false
  if [[ "${code}" -eq 0 && "${rep_exists}" == "true" ]]; then ok=true; fi

  # Log one JSONL record
  printf '{"model_tag":"%s","context_len":%s,"mode":"%s","exit_code":%s,"oom":%s,"ok":%s,"rep":"%s","rep_exists":%s}\n' \
    "${tag}" "${ctx}" "${mode}" "${code}" "${oom}" "${ok}" "${rep}" "${rep_exists}" >> "${LOG_JSONL}"

  if [[ "${ok}" != "true" ]]; then
    echo "  -> recorded failure (oom=${oom}, exit=${code}, rep_exists=${rep_exists})"
  fi
}

# ----------------------------
# Main loops
# ----------------------------
for line in "${MODELS[@]}"; do
  read -r tag d ff L h <<< "${line}"
  for ctx in "${CTX_LENS[@]}"; do
    for mode in "${MODES[@]}"; do
      run_one "${tag}" "${ctx}" "${mode}" "${d}" "${ff}" "${L}" "${h}"
    done
  done
done

echo "Done. Logs: ${LOG_JSONL}"