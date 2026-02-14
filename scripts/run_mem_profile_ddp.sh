#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/results/torch_profiler/ddp_bucket"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

run_one () {
  local name="$1" cmd="$2"
  local log="${LOG_DIR}/${name}.log"

  echo "Run: ${name}"
  (
    set +e
    ${cmd}
    echo "<<<EXIT:$?>>>"
  ) > "${log}" 2>&1

  local code
  code="$(sed -n 's/.*<<<EXIT:\([0-9]\+\)>>>.*/\1/p' "${log}")"
  [[ -z "${code}" ]] && code=999

  if [[ "${code}" -eq 0 ]]; then
    echo "  -> OK"
  else
    echo "  -> FAIL: exit=${code}"
    echo "     log: ${log}"
  fi
}

# bucketed ddp only: run each bucket size separately with CUDA memory snapshot
for bs in 1 10 100 1000; do
  run_one "ddp_bucket_bs${bs}" "uv run python ${ROOT}/cs336_systems/ddp/ddp_bucket_benchmark.py --bucket-size-mb ${bs} --profile --profile-dir ${OUT_DIR}/ddp_bucket_bs${bs}"
done

echo "Done."
