#!/usr/bin/env bash
set -euo pipefail

OMNI_ROOT="${OMNI_ROOT:-/mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM}"
TEST_CASES_ROOT="${TEST_CASES_ROOT:-${OMNI_ROOT}/output_sample/test_cases}"
MODEL_NAME="${MODEL_NAME:-GLM-4.6V-Flash}"
SERVER_BASE_URL="${SERVER_BASE_URL:-http://127.0.0.1:8000/v1}"
SERVER_API_KEY="${SERVER_API_KEY:-EMPTY}"
ABLATION_MODE="${ABLATION_MODE:-self_vlm_tools}"
RESULT_ROOT="${RESULT_ROOT:-eval/results/omni_safe_vtool}"
INCLUDE_CONVERSATION_TRACE="${INCLUDE_CONVERSATION_TRACE:-1}"
LOG_DIR="${LOG_DIR:-}"

mapfile -t TEST_CASE_FILES < <(find "${TEST_CASES_ROOT}" -mindepth 1 -maxdepth 2 -type f -name "test_cases.jsonl" | sort)

if [ "${#TEST_CASE_FILES[@]}" -eq 0 ]; then
  echo "No test_cases.jsonl found under ${TEST_CASES_ROOT}" >&2
  exit 1
fi

echo "Found ${#TEST_CASE_FILES[@]} datasets under ${TEST_CASES_ROOT}"

for test_cases_file in "${TEST_CASE_FILES[@]}"; do
  dataset_name="$(basename "$(dirname "${test_cases_file}")")"
  output_dir="${RESULT_ROOT}/${dataset_name}"
  trace_output="${output_dir}/${dataset_name}.trace.jsonl"
  responses_output="${output_dir}/${dataset_name}.responses.jsonl"

  mkdir -p "${output_dir}"

  echo "============================================================"
  echo "Running dataset: ${dataset_name}"
  echo "Input: ${test_cases_file}"
  echo "Trace output: ${trace_output}"
  echo "Responses output: ${responses_output}"
  echo "============================================================"

  if [ -s "${responses_output}" ]; then
    echo "Skipping dataset: ${dataset_name} (existing non-empty responses file found at ${responses_output})"
    continue
  fi

  cmd=(
    python -m eval.run_omni_safe_vtool
    --test-cases-file "${test_cases_file}"
    --model "${MODEL_NAME}"
    --server-base-url "${SERVER_BASE_URL}"
    --server-api-key "${SERVER_API_KEY}"
    --output "${trace_output}"
    --responses-output "${responses_output}"
    --ablation-mode "${ABLATION_MODE}"
  )

  if [ "${INCLUDE_CONVERSATION_TRACE}" = "1" ]; then
    cmd+=(--include-conversation-trace)
  fi

  if [ -n "${LOG_DIR}" ]; then
    mkdir -p "${LOG_DIR}"
    log_file="${LOG_DIR}/generate_${dataset_name}.log"
    "${cmd[@]}" 2>&1 | tee "${log_file}"
  else
    "${cmd[@]}"
  fi
done

echo "All datasets completed."
