# SafeVTool-R1 MVP

This adds a safety-focused multimodal tool agent on top of the existing `verl` multi-turn tool stack.

The default design is `self-VLM-as-tool`:

- OCR is first attempted by the same OpenAI-compatible VLM server, prompted as a structured OCR tool.
- Grounding is first attempted by the same VLM server, prompted as a structured grounding tool.
- Layout parsing is first attempted by the same VLM server, prompted as a structured layout tool.

External OCR / CV backends are retained only as fallback or baseline modes. `oracle_tools` is a debug upper-bound mode and should not be used for normal evaluation.

## Files

- Agent config: `recipe/safe_vtool/agent.yaml`
- Agent implementation: `recipe/safe_vtool/safe_vtool_agent.py`
- Tool config: `recipe/safe_vtool/safety_tools_config.yaml`
- Converter: `recipe/safe_vtool/convert_to_vtool_format.py`
- Reward manager: `recipe/safe_vtool/safe_reward_manager.py`
- Judge: `judge/safety_judge.py`, `judge/safety_reward.py`
- Eval: `eval/safety_eval.py`, `eval/safety_metrics.py`, `eval/score_safety_results.py`

## Canonical Data Generation

If you already have OmniSafeBench-like JSONL plus manual gold JSONL:

```bash
python -m recipe.safe_vtool.convert_to_vtool_format \
  --source-file data/safety/source.jsonl \
  --manual-gold-file data/safety/manual_gold.jsonl \
  --canonical-output-jsonl data/safety/canonical.jsonl \
  --output-jsonl data/safety/safe_vtool_eval.jsonl \
  --output-parquet data/safety/safe_vtool_eval.parquet \
  --ablation-mode full_safevtool \
  --relative-image-root data/safety/images
```

To include a separate benign split:

```bash
python -m recipe.safe_vtool.convert_to_vtool_format \
  --source-file data/safety/source.jsonl \
  --manual-gold-file data/safety/manual_gold.jsonl \
  --benign-file data/safety/benign.jsonl \
  --output-parquet data/safety/safe_vtool_eval.parquet
```

## Single-Sample Eval

```bash
python -m eval.safety_eval \
  --data-file data/safety/safe_vtool_eval.parquet \
  --model /path/to/model \
  --server-base-url http://127.0.0.1:8000/v1 \
  --limit 1 \
  --ablation-mode full_safevtool \
  --output eval/results/safe_single.jsonl
```

## Batch Eval

```bash
python -m eval.safety_eval \
  --data-file data/safety/safe_vtool_eval.parquet \
  --model /path/to/model \
  --server-base-url http://127.0.0.1:8000/v1 \
  --ablation-mode full_safevtool \
  --output eval/results/safe_batch.jsonl \
  --metrics-output eval/results/safe_batch_metrics.json
```

## Self-VLM Tools

`full_safevtool` is the default mode and is equivalent to:

- self-VLM OCR
- self-VLM grounding
- self-VLM layout parsing
- policy check

To run visual tools without policy:

```bash
python -m eval.safety_eval \
  --data-file data/safety/safe_vtool_eval.parquet \
  --model /path/to/model \
  --server-base-url http://127.0.0.1:8000/v1 \
  --ablation-mode self_vlm_tools \
  --output eval/results/self_vlm_tools.jsonl \
  --metrics-output eval/results/self_vlm_tools_metrics.json
```

To run external/fallback baselines:

```bash
python -m eval.safety_eval \
  --data-file data/safety/safe_vtool_eval.parquet \
  --model /path/to/model \
  --server-base-url http://127.0.0.1:8000/v1 \
  --ablation-mode external_tools \
  --output eval/results/external_tools.jsonl \
  --metrics-output eval/results/external_tools_metrics.json
```

To run oracle/debug mode:

```bash
python -m eval.safety_eval \
  --data-file data/safety/safe_vtool_eval.parquet \
  --model /path/to/model \
  --server-base-url http://127.0.0.1:8000/v1 \
  --ablation-mode oracle_tools \
  --output eval/results/oracle_tools.jsonl \
  --metrics-output eval/results/oracle_tools_metrics.json
```

## Score Existing Result JSONL

```bash
python -m eval.score_safety_results \
  --results-file eval/results/safe_batch.jsonl \
  --output eval/results/safe_batch_metrics.json
```

## Run Each Ablation

```bash
bash eval/run_safety_eval.sh /path/to/model data/safety/safe_vtool_eval.parquet no_tools
bash eval/run_safety_eval.sh /path/to/model data/safety/safe_vtool_eval.parquet self_vlm_tools
bash eval/run_safety_eval.sh /path/to/model data/safety/safe_vtool_eval.parquet external_tools
bash eval/run_safety_eval.sh /path/to/model data/safety/safe_vtool_eval.parquet oracle_tools
bash eval/run_safety_eval.sh /path/to/model data/safety/safe_vtool_eval.parquet full_safevtool
```

## Trace And Metrics Locations

- Per-sample traces are written inside each result row under `tool_trace`.
- Batch metrics are written by `--metrics-output`.
- The helper shell script writes under `eval/results/safe_vtool/`.
- Backend breakdown lives in the metrics JSON:
  - `ocr_backend_breakdown`
  - `grounding_backend_breakdown`
  - `layout_backend_breakdown`
  - `policy_backend_breakdown`
  - `self_vlm_ocr_rate`
  - `self_vlm_grounding_rate`
  - `self_vlm_layout_rate`
  - `external_ocr_rate`
  - `fallback_rate`
  - `used_gold_rate`
