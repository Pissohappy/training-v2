#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-eval/results/neutral_tools}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss-120b}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-any}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8015/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2000}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-3}"
LOG_DIR="${LOG_DIR:-}"

mapfile -t RESPONSE_FILES < <(find "${RESULT_ROOT}" -mindepth 2 -maxdepth 2 -type f -name "*.responses.jsonl" | sort)

if [ "${#RESPONSE_FILES[@]}" -eq 0 ]; then
  echo "No *.responses.jsonl found under ${RESULT_ROOT}" >&2
  exit 1
fi

echo "Found ${#RESPONSE_FILES[@]} response files under ${RESULT_ROOT}"

for responses_file in "${RESPONSE_FILES[@]}"; do
  dataset_dir="$(dirname "${responses_file}")"
  dataset_name="$(basename "${dataset_dir}")"
  judged_output="${dataset_dir}/${dataset_name}.judged.jsonl"
  summary_output="${dataset_dir}/${dataset_name}.judged.summary.json"

  echo "============================================================"
  echo "Judging dataset: ${dataset_name}"
  echo "Responses input: ${responses_file}"
  echo "Judged output: ${judged_output}"
  echo "Summary output: ${summary_output}"
  echo "============================================================"

  cmd=(
    python -m eval.run_omni_judge
    --responses-file "${responses_file}"
    --output "${judged_output}"
    --summary-output "${summary_output}"
    --judge-model "${JUDGE_MODEL}"
    --judge-provider "${JUDGE_PROVIDER}"
    --judge-base-url "${JUDGE_BASE_URL}"
    --judge-api-key "${JUDGE_API_KEY}"
    --judge-max-tokens "${JUDGE_MAX_TOKENS}"
    --judge-temperature "${JUDGE_TEMPERATURE}"
    --success-threshold "${SUCCESS_THRESHOLD}"
  )

  if [ -n "${LOG_DIR}" ]; then
    mkdir -p "${LOG_DIR}"
    log_file="${LOG_DIR}/judge_${dataset_name}.log"
    "${cmd[@]}" 2>&1 | tee "${log_file}"
  else
    "${cmd[@]}"
  fi
done

echo "All judge runs completed."
