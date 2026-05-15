#!/usr/bin/env python3
"""
Compare OCR tool calling rate and ASR before/after SFT.

Reads trace.jsonl and judged.jsonl files to compute:
  - OCR calling rate (tool accuracy)
  - ASR (attack success rate)
  - Breakdown by dataset

Usage:
    # Before SFT (baseline)
    python eval/scripts/compare_ocr_sft.py \
        --results_dir eval/results/omni_safe_vtool \
        --output before_sft.json

    # After SFT (after re-running eval with fine-tuned model)
    python eval/scripts/compare_ocr_sft.py \
        --results_dir eval/results/omni_safe_vtool_sft \
        --output after_sft.json \
        --baseline before_sft.json
"""

import argparse
import json
import sys
from pathlib import Path


def compute_metrics(results_dir: Path) -> dict:
    """Compute OCR rate and ASR for all datasets."""
    metrics = {}
    for ds_dir in sorted(results_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        trace_f = ds_dir / f"{ds_dir.name}.trace.jsonl"
        judged_f = ds_dir / f"{ds_dir.name}.judged.jsonl"
        if not trace_f.exists() or not judged_f.exists():
            continue

        # Load traces
        traces = {}
        with open(trace_f) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                uid = d.get("uid", "") or d.get("test_case_id", "")
                tools = [
                    s["tool_name"]
                    for s in d.get("tool_trace", {}).get("tool_steps", [])
                ]
                traces[uid] = {
                    "has_ocr": "safety_ocr_tool" in tools,
                    "has_grounding": "grounding_tool" in tools,
                    "tools": tools,
                }

        # Load judged
        total = 0
        ocr_count = 0
        grounding_count = 0
        success_count = 0
        sum_h = 0.0
        sum_a = 0.0
        sum_d = 0.0

        with open(judged_f) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                uid = j.get("test_case_id", "")
                total += 1

                t = traces.get(uid, {})
                if t.get("has_ocr"):
                    ocr_count += 1
                if t.get("has_grounding"):
                    grounding_count += 1

                if j.get("success"):
                    success_count += 1
                meta = j.get("metadata", {})
                sum_h += meta.get("harmfulness_score", 0)
                sum_a += meta.get("alignment_score", 0)
                sum_d += meta.get("detail_score", 0)

        metrics[ds_dir.name] = {
            "total": total,
            "ocr_count": ocr_count,
            "ocr_rate": ocr_count / total * 100 if total else 0,
            "grounding_count": grounding_count,
            "grounding_rate": grounding_count / total * 100 if total else 0,
            "success_count": success_count,
            "asr": success_count / total * 100 if total else 0,
            "avg_h": sum_h / total if total else 0,
            "avg_a": sum_a / total if total else 0,
            "avg_d": sum_d / total if total else 0,
        }

    # Overall
    overall = {
        "total": sum(m["total"] for m in metrics.values()),
        "ocr_count": sum(m["ocr_count"] for m in metrics.values()),
        "grounding_count": sum(m["grounding_count"] for m in metrics.values()),
        "success_count": sum(m["success_count"] for m in metrics.values()),
    }
    overall["ocr_rate"] = overall["ocr_count"] / overall["total"] * 100 if overall["total"] else 0
    overall["grounding_rate"] = overall["grounding_count"] / overall["total"] * 100 if overall["total"] else 0
    overall["asr"] = overall["success_count"] / overall["total"] * 100 if overall["total"] else 0

    return {"datasets": metrics, "overall": overall}


def print_table(metrics: dict, title: str, highlight_ds: list[str] | None = None):
    """Print a formatted table."""
    if highlight_ds is None:
        highlight_ds = ["figstep", "arttextfigstep"]

    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")

    header = (
        f"{'Dataset':<22} {'Total':>6} {'OCR':>6} {'OCR%':>7} "
        f"{'GND':>6} {'GND%':>7} {'Succ':>6} {'ASR':>7} "
        f"{'H':>6} {'A':>6} {'D':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for ds_name in sorted(metrics["datasets"].keys()):
        m = metrics["datasets"][ds_name]
        marker = " <<<" if ds_name in highlight_ds else ""
        print(
            f"{ds_name + marker:<22} "
            f"{m['total']:>6} "
            f"{m['ocr_count']:>6} {m['ocr_rate']:>6.1f}% "
            f"{m['grounding_count']:>6} {m['grounding_rate']:>6.1f}% "
            f"{m['success_count']:>6} {m['asr']:>6.2f}% "
            f"{m['avg_h']:>6.2f} {m['avg_a']:>6.2f} {m['avg_d']:>6.2f}"
        )

    print(sep)
    ov = metrics["overall"]
    print(
        f"{'OVERALL':<22} "
        f"{ov['total']:>6} "
        f"{ov['ocr_count']:>6} {ov['ocr_rate']:>6.1f}% "
        f"{ov['grounding_count']:>6} {ov['grounding_rate']:>6.1f}% "
        f"{ov['success_count']:>6} {ov['asr']:>6.2f}%"
    )
    print(sep)


def print_diff(before: dict, after: dict, highlight_ds: list[str] | None = None):
    """Print before/after comparison."""
    if highlight_ds is None:
        highlight_ds = ["figstep", "arttextfigstep"]

    print(f"\n{'=' * 130}")
    print(f"  SFT Before vs After Comparison")
    print(f"{'=' * 130}")

    header = (
        f"{'Dataset':<22} {'OCR% Before':>12} {'OCR% After':>11} {'ΔOCR%':>8}  "
        f"{'ASR Before':>11} {'ASR After':>10} {'ΔASR':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    all_ds = sorted(set(before["datasets"].keys()) | set(after["datasets"].keys()))
    for ds_name in all_ds:
        b = before["datasets"].get(ds_name, {})
        a = after["datasets"].get(ds_name, {})
        ocr_b = b.get("ocr_rate", 0)
        ocr_a = a.get("ocr_rate", 0)
        asr_b = b.get("asr", 0)
        asr_a = a.get("asr", 0)
        delta_ocr = ocr_a - ocr_b
        delta_asr = asr_a - asr_b

        marker = " <<<" if ds_name in highlight_ds else ""
        print(
            f"{ds_name + marker:<22} "
            f"{ocr_b:>11.1f}% {ocr_a:>10.1f}% {delta_ocr:>+7.1f}%  "
            f"{asr_b:>10.2f}% {asr_a:>9.2f}% {delta_asr:>+6.2f}%"
        )

    print(sep)
    b_ov = before.get("overall", {})
    a_ov = after.get("overall", {})
    ocr_b_ov = b_ov.get("ocr_rate", 0)
    ocr_a_ov = a_ov.get("ocr_rate", 0)
    asr_b_ov = b_ov.get("asr", 0)
    asr_a_ov = a_ov.get("asr", 0)
    print(
        f"{'OVERALL':<22} "
        f"{ocr_b_ov:>11.1f}% {ocr_a_ov:>10.1f}% {ocr_a_ov - ocr_b_ov:>+7.1f}%  "
        f"{asr_b_ov:>10.2f}% {asr_a_ov:>9.2f}% {asr_a_ov - asr_b_ov:>+6.2f}%"
    )
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Path to eval results directory")
    parser.add_argument("--output", help="Save metrics to JSON file")
    parser.add_argument("--baseline", help="Path to baseline metrics JSON for comparison")
    parser.add_argument("--label", default="Results", help="Label for display")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} not found")
        sys.exit(1)

    metrics = compute_metrics(results_dir)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output}")

    print_table(metrics, args.label)

    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        print_diff(baseline, metrics)


if __name__ == "__main__":
    main()
