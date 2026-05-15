#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/disk1/szchen/VLMAlignment/training-v2"
MODEL_SETUP_SCRIPT="${MODEL_SETUP_SCRIPT:-/mnt/disk1/szchen/vllm/setup_multiturns.sh}"
JUDGE_SETUP_SCRIPT="${JUDGE_SETUP_SCRIPT:-/mnt/disk1/szchen/vllm/setup_oss.sh}"
CHECK_MODEL_PY="${CHECK_MODEL_PY:-/mnt/disk1/szchen/vllm/check_model.py}"
VLLM_ENV_NAME="${VLLM_ENV_NAME:-vllm_env}"
VTOOL_ENV_NAME="${VTOOL_ENV_NAME:-vtool}"

MODEL_NAME="${MODEL_NAME:-GLM-4.6V-Flash}"
MODEL_SERVER_BASE_URL="${MODEL_SERVER_BASE_URL:-http://127.0.0.1:8010/v1}"
MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-8010}"

JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss-120b}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-any}"
JUDGE_SERVER_BASE_URL="${JUDGE_SERVER_BASE_URL:-http://127.0.0.1:8015/v1}"
JUDGE_SERVER_PORT="${JUDGE_SERVER_PORT:-8015}"

ABLATION_MODE="${ABLATION_MODE:-self_vlm_tools}"
RESULT_ROOT="${RESULT_ROOT:-eval/results/omni_safe_vtool}"
INCLUDE_CONVERSATION_TRACE="${INCLUDE_CONVERSATION_TRACE:-1}"

HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-5}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/omni_pipeline}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${RUN_LOG_DIR}"

PIPELINE_LOG="${RUN_LOG_DIR}/pipeline.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

MODEL_SERVER_PID=""
JUDGE_SERVER_PID=""

send_email() {
  local model_name="$1"
  local attack_name="$2"
  local status="$3"
  local details="${4:-}"

  conda run -n "${VLLM_ENV_NAME}" python - "$CHECK_MODEL_PY" "$model_name" "$attack_name" "$status" "$details" <<'PY'
import importlib.util
import pathlib
import sys

check_model_py = pathlib.Path(sys.argv[1])
model_name = sys.argv[2]
attack_name = sys.argv[3]
status = sys.argv[4]
details = sys.argv[5]

spec = importlib.util.spec_from_file_location("check_model_email", check_model_py)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.send_email_notification(model_name, attack_name, status, details)
PY
}

cleanup_pid() {
  local pid="$1"
  if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup_all() {
  cleanup_pid "${MODEL_SERVER_PID}"
  cleanup_pid "${JUDGE_SERVER_PID}"
}

on_error() {
  local line_no="$1"
  send_email "${MODEL_NAME}" "all_datasets_pipeline" "FAILED" "Pipeline failed near line ${line_no}."
  cleanup_all
}

trap 'on_error $LINENO' ERR
trap cleanup_all EXIT

port_is_busy() {
  local port="$1"
  python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
result = sock.connect_ex(("127.0.0.1", port))
sock.close()
sys.exit(0 if result == 0 else 1)
PY
}

wait_for_server() {
  local base_url="$1"
  local timeout_seconds="$2"
  local poll_seconds="$3"
  local label="$4"

  python - "$base_url" "$timeout_seconds" "$poll_seconds" "$label" <<'PY'
import json
import sys
import time
from urllib import request, error

base_url = sys.argv[1].rstrip("/")
timeout_seconds = int(sys.argv[2])
poll_seconds = float(sys.argv[3])
label = sys.argv[4]
deadline = time.time() + timeout_seconds
models_url = f"{base_url}/models"
last_error = ""

while time.time() < deadline:
    try:
        with request.urlopen(models_url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict) and payload.get("data"):
            print(f"{label} is ready: {models_url}")
            sys.exit(0)
        last_error = f"empty models payload: {payload!r}"
    except Exception as exc:  # noqa: BLE001
        last_error = str(exc)
    time.sleep(poll_seconds)

print(f"Timed out waiting for {label} at {models_url}. Last error: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

start_server() {
  local setup_script="$1"
  local log_file="$2"
  conda run -n "${VLLM_ENV_NAME}" bash "${setup_script}" >"${log_file}" 2>&1 &
  echo $!
}

run_generation() {
  (
    cd "${ROOT_DIR}"
    conda run -n "${VTOOL_ENV_NAME}" \
      env \
      MODEL_NAME="${MODEL_NAME}" \
      SERVER_BASE_URL="${MODEL_SERVER_BASE_URL}" \
      SERVER_API_KEY="${SERVER_API_KEY:-EMPTY}" \
      ABLATION_MODE="${ABLATION_MODE}" \
      RESULT_ROOT="${RESULT_ROOT}" \
      INCLUDE_CONVERSATION_TRACE="${INCLUDE_CONVERSATION_TRACE}" \
      LOG_DIR="${RUN_LOG_DIR}" \
      bash "${ROOT_DIR}/bash/run_all_omni_generate.sh"
  )
}

run_judge() {
  (
    cd "${ROOT_DIR}"
    conda run -n "${VTOOL_ENV_NAME}" \
      env \
      RESULT_ROOT="${RESULT_ROOT}" \
      JUDGE_MODEL="${JUDGE_MODEL}" \
      JUDGE_PROVIDER="${JUDGE_PROVIDER}" \
      JUDGE_BASE_URL="${JUDGE_SERVER_BASE_URL}" \
      JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}" \
      LOG_DIR="${RUN_LOG_DIR}" \
      bash "${ROOT_DIR}/bash/run_all_omni_judge.sh"
  )
}

echo "Run ID: ${RUN_ID}"
echo "Run log dir: ${RUN_LOG_DIR}"

echo "Checking ports..."
if port_is_busy "${MODEL_SERVER_PORT}"; then
  echo "Model server port ${MODEL_SERVER_PORT} is already in use." >&2
  exit 1
fi
if port_is_busy "${JUDGE_SERVER_PORT}"; then
  echo "Judge server port ${JUDGE_SERVER_PORT} is already in use." >&2
  exit 1
fi

echo "Starting tested model server..."
MODEL_SERVER_LOG="${RUN_LOG_DIR}/model_server.log"
MODEL_SERVER_PID="$(start_server "${MODEL_SETUP_SCRIPT}" "${MODEL_SERVER_LOG}")"
wait_for_server "${MODEL_SERVER_BASE_URL}" "${HEALTH_TIMEOUT_SECONDS}" "${HEALTH_POLL_SECONDS}" "tested model server"
send_email "${MODEL_NAME}" "all_datasets_generate" "MODEL_READY" "Tested model server is ready at ${MODEL_SERVER_BASE_URL}."

echo "Running full generation..."
run_generation
send_email "${MODEL_NAME}" "all_datasets_generate" "GENERATE_DONE" "All datasets finished generation. Results stored under ${RESULT_ROOT}."

echo "Stopping tested model server..."
cleanup_pid "${MODEL_SERVER_PID}"
MODEL_SERVER_PID=""

echo "Starting judge model server..."
JUDGE_SERVER_LOG="${RUN_LOG_DIR}/judge_server.log"
JUDGE_SERVER_PID="$(start_server "${JUDGE_SETUP_SCRIPT}" "${JUDGE_SERVER_LOG}")"
wait_for_server "${JUDGE_SERVER_BASE_URL}" "${HEALTH_TIMEOUT_SECONDS}" "${HEALTH_POLL_SECONDS}" "judge model server"
send_email "${JUDGE_MODEL}" "all_datasets_judge" "JUDGE_READY" "Judge model server is ready at ${JUDGE_SERVER_BASE_URL}."

echo "Running full judge..."
run_judge
send_email "${JUDGE_MODEL}" "all_datasets_judge" "JUDGE_DONE" "All datasets finished judge evaluation. Results stored under ${RESULT_ROOT}."

echo "Stopping judge model server..."
cleanup_pid "${JUDGE_SERVER_PID}"
JUDGE_SERVER_PID=""

send_email "${MODEL_NAME}" "all_datasets_pipeline" "COMPLETED" "Full Omni pipeline finished successfully."
echo "Full Omni pipeline completed."
