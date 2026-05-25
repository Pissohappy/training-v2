#!/usr/bin/env bash
set -euo pipefail

OMNI_ROOT="${OMNI_ROOT:-/mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/disk1/szchen/miniconda3/envs/vtool/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-config/general_config_benchmark_defense_utility_evaluation.yaml}"
RESP_ROOT="${RESP_ROOT:-/mnt/disk1/szchen/VLMAlignment/training-v2/eval/results/omni_safe_vtool}"
OUT_ROOT="${OUT_ROOT:-/mnt/disk1/szchen/VLMAlignment/training-v2/eval/results/omni_benchmark_eval}"

run_eval() {
    local input_file="$1"
    local output_dir="$2"

    echo "=== Evaluating ${input_file} -> ${output_dir}"
    (
        cd "${OMNI_ROOT}"
        "${PYTHON_BIN}" run_pipeline.py \
            --config "${CONFIG_PATH}" \
            --stage evaluation \
            --input-file "${input_file}" \
            --output-dir "${output_dir}"
    )
}

run_eval "${RESP_ROOT}/mmbench.responses.jsonl" "${OUT_ROOT}/neutral"
run_eval "${RESP_ROOT}/mmmu.responses.jsonl" "${OUT_ROOT}/neutral"
run_eval "${RESP_ROOT}/mmbench_safety.responses.jsonl" "${OUT_ROOT}/safety"
run_eval "${RESP_ROOT}/mmmu_safety.responses.jsonl" "${OUT_ROOT}/safety"
