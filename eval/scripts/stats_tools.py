#!/usr/bin/env python3
"""Extract tool usage statistics from trace files and print a table."""

import json
import os
import sys
from collections import Counter
from pathlib import Path


def extract_tools_from_trace(filepath: Path) -> dict:
    """Extract tool usage counts from a trace jsonl file.

    Returns dict with per-sample tool set and total sample count.
    """
    tool_counter = Counter()
    total_samples = 0
    fallback_samples = 0
    truncated_samples = 0
    recovered_tool_calls = 0

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_samples += 1
            parse_stats = record.get("tool_call_parse_stats", {})
            if parse_stats.get("fallback_used"):
                fallback_samples += 1
            if parse_stats.get("truncated_tool_call_block"):
                truncated_samples += 1
            recovered_tool_calls += int(parse_stats.get("recovered_tool_calls", 0) or 0)
            steps = record.get("tool_trace", {}).get("tool_steps", [])
            if not steps:
                continue
            seen = set()
            for step in steps:
                name = step.get("tool_name", "unknown")
                if name not in seen:
                    tool_counter[name] += 1
                    seen.add(name)

    return {
        "total": total_samples,
        "tools": tool_counter,
        "fallback_samples": fallback_samples,
        "truncated_samples": truncated_samples,
        "recovered_tool_calls": recovered_tool_calls,
    }


def main(results_dir: str):
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    trace_files = sorted(results_dir.rglob("*.trace.jsonl"))
    if not trace_files:
        print(f"No *.trace.jsonl files found in {results_dir}")
        sys.exit(1)

    # Collect stats per dataset
    rows = []
    all_tool_names = set()
    for fpath in trace_files:
        ds_name = fpath.stem.replace(".trace", "")
        stats = extract_tools_from_trace(fpath)
        stats["dataset"] = ds_name
        rows.append(stats)
        all_tool_names.update(stats["tools"].keys())

    tool_names = sorted(all_tool_names)

    # Print table
    col_w = max(len(t) for t in tool_names) + 2 if tool_names else 12
    header = f"{'Dataset':<22} {'Total':>6} {'FB':>5} {'TR':>5} {'REC':>5}"
    for t in tool_names:
        header += f" {t:>{col_w}}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    grand_total = 0
    grand_fallback = 0
    grand_truncated = 0
    grand_recovered = 0
    grand_tools = Counter()
    for row in rows:
        line = (
            f"{row['dataset']:<22} {row['total']:>6} "
            f"{row['fallback_samples']:>5} {row['truncated_samples']:>5} {row['recovered_tool_calls']:>5}"
        )
        for t in tool_names:
            c = row["tools"].get(t, 0)
            line += f" {c:>{col_w}}"
        print(line)
        grand_total += row["total"]
        grand_fallback += row["fallback_samples"]
        grand_truncated += row["truncated_samples"]
        grand_recovered += row["recovered_tool_calls"]
        grand_tools.update(row["tools"])

    print(sep)
    line = f"{'OVERALL':<22} {grand_total:>6} {grand_fallback:>5} {grand_truncated:>5} {grand_recovered:>5}"
    for t in tool_names:
        line += f" {grand_tools.get(t, 0):>{col_w}}"
    print(line)
    print(sep)

    # Print per-tool usage rate
    print()
    if grand_total:
        print(f"  fallback parser used: {grand_fallback}/{grand_total} = {grand_fallback / grand_total * 100:.1f}%")
        print(f"  truncated <tool_call> samples: {grand_truncated}/{grand_total} = {grand_truncated / grand_total * 100:.1f}%")
        print(f"  recovered tool calls: {grand_recovered}")
        print()
    for t in tool_names:
        rate = grand_tools.get(t, 0) / grand_total * 100 if grand_total else 0
        print(f"  {t}: {grand_tools.get(t, 0)}/{grand_total} = {rate:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "omni_safe_vtool",
        )
    main(target)
