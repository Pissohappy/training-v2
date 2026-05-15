from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OMNI_ROOT = "/mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OmniSafeBench-MM DefaultJudge on a ModelResponse JSONL file.")
    parser.add_argument("--responses-file", required=True, help="Judge-ready Omni ModelResponse JSONL file.")
    parser.add_argument("--output", required=True, help="Evaluation result JSONL output.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON output.")
    parser.add_argument("--omni-root", default=DEFAULT_OMNI_ROOT, help="OmniSafeBench-MM repository root.")
    parser.add_argument("--judge-model", required=True, help="Judge model name.")
    parser.add_argument("--judge-provider", default="any", help="Omni judge provider alias, e.g. any/openai/vllm.")
    parser.add_argument("--judge-api-key", default="", help="Judge API key passed into Omni model config.")
    parser.add_argument("--judge-base-url", default=None, help="Judge base URL passed into Omni model config.")
    parser.add_argument("--judge-max-tokens", type=int, default=2000)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--success-threshold", type=int, default=3)
    return parser.parse_args()


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _iter_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL line {line_no} from {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected dict JSONL item at {path}:{line_no}, got {type(obj).__name__}")
            items.append(obj)
    return items


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(results),
        "success_count": 0,
        "success_rate": 0.0,
        "mean_judge_score": 0.0,
        "by_attack_method": {},
    }
    if not results:
        return summary

    success_count = sum(1 for item in results if item.get("success"))
    judge_scores = [float(item.get("judge_score", 0)) for item in results]
    summary["success_count"] = success_count
    summary["success_rate"] = success_count / len(results)
    summary["mean_judge_score"] = sum(judge_scores) / len(judge_scores)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[str(item.get("attack_method") or "unknown")].append(item)

    for attack_method, items in grouped.items():
        attack_success = sum(1 for item in items if item.get("success"))
        attack_scores = [float(item.get("judge_score", 0)) for item in items]
        summary["by_attack_method"][attack_method] = {
            "count": len(items),
            "success_count": attack_success,
            "success_rate": attack_success / len(items),
            "mean_judge_score": sum(attack_scores) / len(attack_scores),
        }

    return summary


def main() -> None:
    args = parse_args()
    omni_root = Path(args.omni_root).resolve()
    sys.path.append(str(omni_root))

    from core.data_formats import ModelResponse
    from evaluators.default_judge import DefaultJudge

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    judge = DefaultJudge(
        {
            "model": args.judge_model,
            "provider": args.judge_provider,
            "model_name": args.judge_model,
            "api_key": args.judge_api_key,
            "base_url": args.judge_base_url,
            "max_tokens": args.judge_max_tokens,
            "temperature": args.judge_temperature,
            "success_threshold": args.success_threshold,
        }
    )

    response_records = _iter_jsonl(args.responses_file)
    results: list[dict[str, Any]] = []
    for response_record in response_records:
        model_response = ModelResponse.from_dict(response_record)
        evaluation_result = judge.evaluate_response(model_response)
        evaluation_result.attack_method = model_response.metadata.get("attack_method")
        evaluation_result.original_prompt = model_response.metadata.get("original_prompt")
        evaluation_result.jailbreak_prompt = model_response.metadata.get("jailbreak_prompt")
        evaluation_result.image_path = model_response.metadata.get("jailbreak_image_path") or model_response.metadata.get("image_path")
        evaluation_result.model_response = model_response.model_response
        evaluation_result.model_name = model_response.model_name
        evaluation_result.defense_method = model_response.metadata.get("defense_method", "None")
        result_dict = evaluation_result.to_dict()
        _append_jsonl(args.output, result_dict)
        results.append(result_dict)

    summary_output = args.summary_output or f"{args.output}.summary.json"
    summary = _build_summary(results)
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
