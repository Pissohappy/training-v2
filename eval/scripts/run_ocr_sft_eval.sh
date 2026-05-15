#!/bin/bash

set -euo pipefail

MODE="${1:-all}"

PROJECT_DIR="/mnt/disk1/szchen/VLMAlignment/training-v2"
OMNI_ROOT="/mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/eval/sft_output/ocr_sft_merged}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-4.6V-Flash-ocr-sft}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/eval/results/omni_safe_vtool_sft}"
OVERWRITE_OUTPUTS="${OVERWRITE_OUTPUTS:-1}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
PORT="${PORT:-8010}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-4}"
ABLATION_MODE="${ABLATION_MODE:-self_vlm_tools}"
SERVER_BASE_URL="http://127.0.0.1:${PORT}/v1"

FIGSTEP_CASES="${FIGSTEP_CASES:-${OMNI_ROOT}/output_sample/test_cases/figstep/test_cases.jsonl}"
ARTTEXT_CASES="${ARTTEXT_CASES:-${OMNI_ROOT}/output_sample/test_cases/arttextfigstep/test_cases.jsonl}"

SERVER_PID=""

usage() {
    cat <<EOF
Usage:
  bash eval/scripts/run_ocr_sft_eval.sh serve
  bash eval/scripts/run_ocr_sft_eval.sh eval
  bash eval/scripts/run_ocr_sft_eval.sh all

Environment overrides:
  MODEL_DIR                default: ${MODEL_DIR}
  SERVED_MODEL_NAME        default: ${SERVED_MODEL_NAME}
  RESULTS_ROOT             default: ${RESULTS_ROOT}
  OVERWRITE_OUTPUTS        default: ${OVERWRITE_OUTPUTS} (1 deletes old outputs before eval)
  CUDA_VISIBLE_DEVICES     default: ${CUDA_DEVICES}
  PORT                     default: ${PORT}
  TENSOR_PARALLEL_SIZE     default: ${TENSOR_PARALLEL_SIZE}
  GPU_MEMORY_UTILIZATION   default: ${GPU_MEMORY_UTILIZATION}
  MAX_MODEL_LEN            default: ${MAX_MODEL_LEN}
  MAX_TOOL_CALLS           default: ${MAX_TOOL_CALLS}
  ABLATION_MODE            default: ${ABLATION_MODE}
EOF
}

ensure_paths() {
    test -d "${MODEL_DIR}" || { echo "Model dir not found: ${MODEL_DIR}" >&2; exit 1; }
    test -f "${FIGSTEP_CASES}" || { echo "figstep test cases not found: ${FIGSTEP_CASES}" >&2; exit 1; }
    test -f "${ARTTEXT_CASES}" || { echo "arttextfigstep test cases not found: ${ARTTEXT_CASES}" >&2; exit 1; }
    mkdir -p "${RESULTS_ROOT}/figstep" "${RESULTS_ROOT}/arttextfigstep"
}

wait_for_server() {
    local retries=60
    local url="http://127.0.0.1:${PORT}/v1/models"
    echo "Waiting for vLLM server at ${url} ..."
    for _ in $(seq 1 "${retries}"); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            echo "vLLM server is ready."
            return 0
        fi
        sleep 2
    done

    echo "Timed out waiting for vLLM server on port ${PORT}." >&2
    return 1
}

start_server() {
    ensure_paths
    echo "Starting merged model server"
    echo "  model: ${MODEL_DIR}"
    echo "  served name: ${SERVED_MODEL_NAME}"
    echo "  cuda devices: ${CUDA_DEVICES}"
    echo "  port: ${PORT}"

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" OMP_NUM_THREADS=1 \
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_DIR}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --port "${PORT}" \
        --trust-remote-code \
        --dtype auto \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --enable-auto-tool-choice \
        --tool-call-parser glm45 &

    SERVER_PID=$!
    wait_for_server
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}

run_eval_one() {
    local dataset_name="$1"
    local cases_file="$2"
    local output_file="${RESULTS_ROOT}/${dataset_name}/${dataset_name}.trace.jsonl"
    local responses_file="${output_file}.omni_responses.jsonl"

    echo "Running eval for ${dataset_name}"
    echo "  cases: ${cases_file}"
    echo "  output: ${output_file}"

    if [[ "${OVERWRITE_OUTPUTS}" == "1" ]]; then
        rm -f "${output_file}" "${responses_file}"
    fi

    python "${PROJECT_DIR}/eval/run_omni_safe_vtool.py" \
        --test-cases-file "${cases_file}" \
        --model "${MODEL_DIR}" \
        --server-base-url "${SERVER_BASE_URL}" \
        --server-model "${SERVED_MODEL_NAME}" \
        --ablation-mode "${ABLATION_MODE}" \
        --max-tool-calls "${MAX_TOOL_CALLS}" \
        --include-conversation-trace \
        --output "${output_file}"
}

run_eval() {
    ensure_paths
    run_eval_one "figstep" "${FIGSTEP_CASES}"
    run_eval_one "arttextfigstep" "${ARTTEXT_CASES}"
}

case "${MODE}" in
    serve)
        start_server
        wait "${SERVER_PID}"
        ;;
    eval)
        run_eval
        ;;
    all)
        trap stop_server EXIT
        start_server
        run_eval
        ;;
    *)
        usage
        exit 1
        ;;
esac
