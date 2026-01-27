#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT}/cs336_systems/nsys_profile.py"

OUT_DIR="${ROOT}/results/nsys"
LOG_DIR="${OUT_DIR}/logs"
RUNS_JSONL="${OUT_DIR}/runs.jsonl"
TIMES_JSONL="${OUT_DIR}/times.jsonl"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
: > "${RUNS_JSONL}"
: > "${TIMES_JSONL}"

WARM_UP=5
NSTEPS=10
DEVICE=cuda
DTYPE=fp32

CTX_LENS=(128 256 512 1024)
MODES=(forward_only train_step)

MODELS=(
  "small  768   3072   12  12"
  "medium 1024  4096   24  16"
  "large  1280  5120   36  20"
  "xl     1600  6400   48  25"
  "2.7B   2560  10240  32  32"
)

run_one () {
  local tag="$1" ctx="$2" mode="$3" d="$4" ff="$5" L="$6" h="$7"
  local base="${OUT_DIR}/${tag}_ctx${ctx}_${mode}"
  local rep="${base}.nsys-rep"
  local log="${LOG_DIR}/${tag}_ctx${ctx}_${mode}.log"

  echo "Run: ${tag} ctx=${ctx} mode=${mode}"

  local out code oom ok
  out="$(
    set +e
    nsys profile \
      --trace=cuda,nvtx \
      --pytorch=autograd-nvtx \
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
        --output "${TIMES_JSONL}" \
      2>&1
    echo "<<<EXIT:$?>>>"
  )"

  printf "%s\n" "${out}" > "${log}"

  code="$(printf "%s" "${out}" | sed -n 's/.*<<<EXIT:\([0-9]\+\)>>>.*/\1/p')"
  [[ -z "${code}" ]] && code=999

  oom=false
  if printf "%s" "${out}" | grep -qiE "out of memory|cuda out of memory|OutOfMemoryError"; then
    oom=true
  fi

  ok=false
  if [[ "${code}" -eq 0 && -f "${rep}" ]]; then
    ok=true
  fi

  printf '{"model_tag":"%s","context_len":%s,"mode":"%s","exit_code":%s,"oom":%s,"ok":%s,"rep":"%s","log":"%s"}\n' \
    "${tag}" "${ctx}" "${mode}" "${code}" \
    "${oom}" "${ok}" \
    "${rep}" "${log}" >> "${RUNS_JSONL}"

  if [[ "${ok}" == "true" ]]; then
    echo "  -> OK: ${rep}"
  else
    echo "  -> FAIL: exit=${code} oom=${oom} rep_exists=$([[ -f "${rep}" ]] && echo true || echo false)"
    echo "     log: ${log}"
  fi
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
echo "  Runs log:  ${RUNS_JSONL}"
echo "  Times log: ${TIMES_JSONL}"