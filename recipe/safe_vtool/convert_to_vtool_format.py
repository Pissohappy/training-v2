from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recipe.safe_vtool.common import normalize_ablation_mode


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _require_field(record: dict[str, Any], candidates: list[str], *, label: str) -> Any:
    for key in candidates:
        if key in record and record[key] not in (None, ""):
            return record[key]
    raise ValueError(f"Missing required field {label}. Tried keys: {candidates}")


def normalize_source_row(row: dict[str, Any], *, relative_image_root: str | None = None) -> dict[str, Any]:
    sample_id = str(_require_field(row, ["sample_id", "id", "uid"], label="sample_id"))
    image_path = str(_require_field(row, ["image_path", "image", "path"], label="image_path"))
    if relative_image_root and not Path(image_path).is_absolute():
        image_path = str((Path(relative_image_root) / image_path).resolve())

    user_query = str(_require_field(row, ["user_query", "query", "question", "prompt"], label="user_query"))
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "user_query": user_query,
        "source_dataset": str(row.get("source_dataset") or row.get("dataset") or "omnisaferow"),
        "source_split": str(row.get("source_split") or row.get("split") or "unknown"),
        "metadata": metadata,
        "expected_decision": row.get("expected_decision"),
        "expected_policy_tags": list(row.get("expected_policy_tags") or []),
        "expected_response_style": row.get("expected_response_style"),
        "visual_text_gold": row.get("visual_text_gold"),
        "evidence_gold": row.get("evidence_gold"),
    }


def normalize_manual_gold(row: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(_require_field(row, ["sample_id", "id", "uid"], label="manual sample_id"))
    manual_gold = row.get("manual_gold") if isinstance(row.get("manual_gold"), dict) else row
    decision = str(_require_field(manual_gold, ["decision"], label="manual_gold.decision"))
    policy_tags = list(_require_field(manual_gold, ["policy_tags"], label="manual_gold.policy_tags"))
    rationale_short = str(_require_field(manual_gold, ["rationale_short"], label="manual_gold.rationale_short"))
    return {
        "sample_id": sample_id,
        "manual_gold": {
            "decision": decision,
            "policy_tags": policy_tags,
            "rationale_short": rationale_short,
            "evidence_text": manual_gold.get("evidence_text"),
            "evidence_regions": list(manual_gold.get("evidence_regions") or []),
            "allow_tool_modes": list(manual_gold.get("allow_tool_modes") or []),
        },
    }


def build_canonical_dataset(
    *,
    source_rows: list[dict[str, Any]],
    manual_gold_rows: list[dict[str, Any]],
    benign_rows: list[dict[str, Any]] | None = None,
    relative_image_root: str | None = None,
) -> list[dict[str, Any]]:
    normalized_manual = {
        row["sample_id"]: row["manual_gold"] for row in (normalize_manual_gold(item) for item in manual_gold_rows)
    }
    canonical: list[dict[str, Any]] = []

    for row in source_rows:
        normalized = normalize_source_row(row, relative_image_root=relative_image_root)
        sample_id = normalized["sample_id"]
        if sample_id not in normalized_manual:
            raise ValueError(f"Missing manual gold for sample_id={sample_id}")
        normalized["manual_gold"] = normalized_manual[sample_id]
        canonical.append(normalized)

    for benign in benign_rows or []:
        normalized = normalize_source_row(benign, relative_image_root=relative_image_root)
        sample_id = normalized["sample_id"]
        if sample_id not in normalized_manual:
            raise ValueError(f"Missing manual gold for benign sample_id={sample_id}")
        normalized["manual_gold"] = normalized_manual[sample_id]
        normalized["metadata"] = dict(normalized.get("metadata") or {}, benign_split=True)
        canonical.append(normalized)

    return canonical


def canonical_to_vtool_row(record: dict[str, Any], *, ablation_mode: str) -> dict[str, Any]:
    safety_gold = {
        "sample_id": record["sample_id"],
        "expected_decision": record.get("expected_decision"),
        "expected_policy_tags": record.get("expected_policy_tags") or [],
        "expected_response_style": record.get("expected_response_style"),
        "visual_text_gold": record.get("visual_text_gold"),
        "evidence_gold": record.get("evidence_gold"),
        "manual_gold": record["manual_gold"],
    }
    metadata = {
        "sample_id": record["sample_id"],
        "source_dataset": record["source_dataset"],
        "source_split": record["source_split"],
        "ablation_mode": ablation_mode,
        "visual_text_gold": record.get("visual_text_gold"),
        "evidence_gold": record.get("evidence_gold"),
        "evidence_regions": record["manual_gold"].get("evidence_regions") or [],
        "manual_gold": record["manual_gold"],
        "metadata": record.get("metadata") or {},
    }
    return {
        "data_source": record["source_dataset"],
        "agent_name": "safe_vtool_agent",
        "uid": record["sample_id"],
        "images": [record["image_path"]],
        "prompt": [{"role": "user", "content": f"<image>\n{record['user_query']}"}],
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(safety_gold, ensure_ascii=False),
        },
        "extra_info": {
            "index": record["sample_id"],
            "need_tools_kwargs": True,
            "tools_kwargs": {"metadata": json.dumps(metadata, ensure_ascii=False)},
            "safety_gold": safety_gold,
        },
    }


def write_records(records: list[dict[str, Any]], *, jsonl_path: str | None, parquet_path: str | None) -> None:
    if jsonl_path:
        output_path = Path(jsonl_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if parquet_path:
        import pandas as pd

        output_path = Path(parquet_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_parquet(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert safety JSONL into VTool-style RL rows.")
    parser.add_argument("--source-file", nargs="+", required=True, help="Source OmniSafeBench-style JSONL files.")
    parser.add_argument("--manual-gold-file", required=True, help="Required manual gold JSONL.")
    parser.add_argument("--benign-file", nargs="*", default=None, help="Optional benign JSONL files.")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-parquet", default=None)
    parser.add_argument("--canonical-output-jsonl", default=None)
    parser.add_argument("--ablation-mode", default="full_safevtool")
    parser.add_argument("--relative-image-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation_mode = normalize_ablation_mode(args.ablation_mode)

    source_rows: list[dict[str, Any]] = []
    for path in args.source_file:
        source_rows.extend(load_jsonl(path))

    benign_rows: list[dict[str, Any]] = []
    for path in args.benign_file or []:
        benign_rows.extend(load_jsonl(path))

    canonical = build_canonical_dataset(
        source_rows=source_rows,
        manual_gold_rows=load_jsonl(args.manual_gold_file),
        benign_rows=benign_rows,
        relative_image_root=args.relative_image_root,
    )
    if args.canonical_output_jsonl:
        write_records(canonical, jsonl_path=args.canonical_output_jsonl, parquet_path=None)

    vtool_rows = [canonical_to_vtool_row(item, ablation_mode=ablation_mode) for item in canonical]
    write_records(vtool_rows, jsonl_path=args.output_jsonl, parquet_path=args.output_parquet)


if __name__ == "__main__":
    main()
