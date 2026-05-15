#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?model path required}"
DATA_FILE="${2:?data file required}"
ABLATION_MODE="${3:-full_safevtool}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="eval/results/safe_vtool/${ABLATION_MODE}"
mkdir -p "${OUT_DIR}"

python -m eval.safety_eval \
  --data-file "${DATA_FILE}" \
  --model "${MODEL_PATH}" \
  --ablation-mode "${ABLATION_MODE}" \
  --output "${OUT_DIR}/results_${STAMP}.jsonl" \
  --metrics-output "${OUT_DIR}/metrics_${STAMP}.json"
