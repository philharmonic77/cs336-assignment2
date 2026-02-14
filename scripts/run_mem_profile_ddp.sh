#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/results/torch_profiler/ddp_bucket"
LOG_FILE="${OUT_DIR}/run_mem_profile_ddp.log"

mkdir -p "${OUT_DIR}"
: > "${LOG_FILE}"

run_one () {
  local name="$1" cmd="$2"
  local tmp_log
  tmp_log="$(mktemp)"

  echo "Run: ${name}"
  (
    set +e
    ${cmd}
    echo "<<<EXIT:$?>>>"
  ) > "${tmp_log}" 2>&1

  {
    echo "===== ${name} ====="
    cat "${tmp_log}"
    echo
  } >> "${LOG_FILE}"

  local code
  code="$(sed -n 's/.*<<<EXIT:\([0-9]\+\)>>>.*/\1/p' "${tmp_log}" | tail -n 1)"
  [[ -z "${code}" ]] && code=999
  rm -f "${tmp_log}"

  if [[ "${code}" -eq 0 ]]; then
    echo "  -> OK"
  else
    echo "  -> FAIL: exit=${code}"
    echo "     log: ${LOG_FILE}"
  fi
}

# bucketed ddp only: run each bucket size separately with CUDA memory snapshot
for bs in 1 10 100 1000; do
  run_one "ddp_bucket_bs${bs}" "uv run python ${ROOT}/cs336_systems/ddp/ddp_bucket_benchmark.py --bucket-size-mb ${bs} --profile-out ${OUT_DIR}/bucket${bs}_rank0_memory_snapshot.pickle"
done

echo "Done."
echo "Log file: ${LOG_FILE}"
