#!/bin/bash
# OCR SFT Training Pipeline using LLaMA-Factory
#
# Steps:
#   1. Prepare SFT data from trace files
#   2. Train with LLaMA-Factory LoRA
#   3. (Manual) Merge LoRA and re-run eval
#
# Usage:
#   bash eval/scripts/run_ocr_sft.sh
#   CUDA_VISIBLE_DEVICES=0 bash eval/scripts/run_ocr_sft.sh
#   CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash eval/scripts/run_ocr_sft.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LLAMAFACTORY_DIR="/mnt/disk1/szchen/VLMAlignment/LlamaFactory"
RESULTS_DIR="${PROJECT_DIR}/eval/results/omni_safe_vtool"
SFT_DATA_DIR="${PROJECT_DIR}/eval/sft_data"
SFT_OUTPUT_DIR="${PROJECT_DIR}/eval/sft_output/ocr_lora"

echo "============================================"
echo "OCR SFT Training Pipeline"
echo "============================================"

# Step 1: Prepare SFT data
echo ""
echo "[Step 1/2] Preparing SFT data..."
python3 "${SCRIPT_DIR}/prepare_ocr_sft_data.py" \
    "${RESULTS_DIR}" \
    "${SFT_DATA_DIR}/ocr_sft_train.jsonl"
echo "  OpenAI tool-use data: ${SFT_DATA_DIR}/ocr_sft_train_openai.jsonl"
echo "  LLaMA-Factory data:   ${SFT_DATA_DIR}/ocr_sft_train_llamafactory.jsonl"

# Step 2: Train with LLaMA-Factory
echo ""
echo "[Step 2/2] Training with LLaMA-Factory..."
echo "  Config: ${SCRIPT_DIR}/ocr_sft_lora.yaml"
echo "  Output: ${SFT_OUTPUT_DIR}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE:-<auto>}"
echo ""

cd "${LLAMAFACTORY_DIR}"
PYTHONPATH=src python -m llamafactory.cli train "${SCRIPT_DIR}/ocr_sft_lora.yaml"

echo ""
echo "============================================"
echo "Training complete!"
echo "LoRA adapter saved to: ${SFT_OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Merge LoRA: llamafactory-cli export ${SCRIPT_DIR}/ocr_sft_merge.yaml"
echo "  2. Re-run eval on figstep/arttextfigstep with the fine-tuned model"
echo "  3. Compare: python ${SCRIPT_DIR}/compare_ocr_sft.py --baseline ... --results_dir ..."
echo "============================================"
