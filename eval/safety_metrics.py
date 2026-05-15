from __future__ import annotations

from collections import defaultdict
from typing import Any


def aggregate_safety_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "overall_score": 0.0,
            "decision_accuracy": 0.0,
            "policy_accuracy": 0.0,
            "evidence_accuracy": 0.0,
            "tool_use_reasonableness": 0.0,
            "harmful_leakage_rate": 0.0,
            "over_refusal_rate": 0.0,
            "used_gold_rate": 0.0,
            "ocr_backend_breakdown": {},
            "grounding_backend_breakdown": {},
            "layout_backend_breakdown": {},
            "policy_backend_breakdown": {},
            "empty_ocr_rate": 0.0,
            "heuristic_grounding_rate": 0.0,
            "self_vlm_ocr_rate": 0.0,
            "self_vlm_grounding_rate": 0.0,
            "self_vlm_layout_rate": 0.0,
            "external_ocr_rate": 0.0,
            "fallback_rate": 0.0,
            "per_source_dataset": {},
            "per_ablation": {},
        }

    def _mean(key: str) -> float:
        return sum(float(record.get(key, 0.0)) for record in records) / len(records)

    def _group(field: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get(field) or "unknown")].append(record)
        return {
            key: {
                "count": len(items),
                "overall_score": sum(float(item.get("total_score", 0.0)) for item in items) / len(items),
                "decision_accuracy": sum(float(item.get("decision_correct", 0.0)) for item in items) / len(items),
                "harmful_leakage_rate": sum(float(item.get("harmful_leakage", 0.0)) for item in items) / len(items),
            }
            for key, items in grouped.items()
        }

    def _step_breakdown(prefix: str, tool_name: str) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for record in records:
            for step in (record.get("tool_trace") or {}).get("tool_steps") or []:
                if step.get("tool_name") != tool_name:
                    continue
                backend = str((step.get("metadata") or {}).get("backend") or "unknown")
                counts[backend] += 1
        return dict(counts)

    total_tool_steps = 0
    used_gold_steps = 0
    empty_ocr_steps = 0
    ocr_steps = 0
    grounding_steps = 0
    heuristic_grounding_steps = 0
    self_vlm_ocr_steps = 0
    self_vlm_grounding_steps = 0
    self_vlm_layout_steps = 0
    external_ocr_steps = 0
    fallback_steps = 0

    for record in records:
        for step in (record.get("tool_trace") or {}).get("tool_steps") or []:
            total_tool_steps += 1
            metadata = step.get("metadata") or {}
            backend = str(metadata.get("backend") or "")
            if bool(metadata.get("used_gold")):
                used_gold_steps += 1
            if step.get("tool_name") == "safety_ocr_tool":
                ocr_steps += 1
                if backend == "empty_fallback":
                    empty_ocr_steps += 1
                if backend == "self_vlm_ocr":
                    self_vlm_ocr_steps += 1
                if backend in {"easyocr", "pytesseract"}:
                    external_ocr_steps += 1
            if step.get("tool_name") == "grounding_tool":
                grounding_steps += 1
                if backend == "heuristic_fallback":
                    heuristic_grounding_steps += 1
                if backend == "self_vlm_grounding":
                    self_vlm_grounding_steps += 1
            if step.get("tool_name") == "layout_parse_tool" and backend == "self_vlm_layout":
                self_vlm_layout_steps += 1
            if backend in {"empty_fallback", "heuristic_fallback", "cv_connected_components", "ocr_block_layout", "ocr_layout_fallback"}:
                fallback_steps += 1

    return {
        "count": len(records),
        "overall_score": _mean("total_score"),
        "decision_accuracy": _mean("decision_correct"),
        "policy_accuracy": _mean("policy_correct"),
        "evidence_accuracy": _mean("evidence_correct"),
        "tool_use_reasonableness": _mean("tool_use_reasonable"),
        "harmful_leakage_rate": _mean("harmful_leakage"),
        "over_refusal_rate": _mean("over_refusal"),
        "used_gold_rate": (used_gold_steps / total_tool_steps) if total_tool_steps else 0.0,
        "ocr_backend_breakdown": _step_breakdown("ocr", "safety_ocr_tool"),
        "grounding_backend_breakdown": _step_breakdown("grounding", "grounding_tool"),
        "layout_backend_breakdown": _step_breakdown("layout", "layout_parse_tool"),
        "policy_backend_breakdown": _step_breakdown("policy", "policy_check_tool"),
        "empty_ocr_rate": (empty_ocr_steps / sum(_step_breakdown("ocr", "safety_ocr_tool").values()))
        if _step_breakdown("ocr", "safety_ocr_tool")
        else 0.0,
        "heuristic_grounding_rate": (heuristic_grounding_steps / grounding_steps) if grounding_steps else 0.0,
        "self_vlm_ocr_rate": (self_vlm_ocr_steps / ocr_steps) if ocr_steps else 0.0,
        "self_vlm_grounding_rate": (self_vlm_grounding_steps / grounding_steps) if grounding_steps else 0.0,
        "self_vlm_layout_rate": (
            self_vlm_layout_steps
            / sum(_step_breakdown("layout", "layout_parse_tool").values())
        )
        if _step_breakdown("layout", "layout_parse_tool")
        else 0.0,
        "external_ocr_rate": (external_ocr_steps / ocr_steps) if ocr_steps else 0.0,
        "fallback_rate": (fallback_steps / total_tool_steps) if total_tool_steps else 0.0,
        "per_source_dataset": _group("source_dataset"),
        "per_ablation": _group("ablation_mode"),
    }
