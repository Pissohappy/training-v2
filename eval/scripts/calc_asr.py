#!/usr/bin/env python3
"""Calculate ASR (Attack Success Rate) from judged_jsonl files and print a table."""

import json
import os
import sys
from pathlib import Path


def calc_asr(results_dir: str):
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    # Collect all judged.jsonl files
    judged_files = sorted(results_dir.rglob("*.judged.jsonl"))
    if not judged_files:
        print(f"No *.judged.jsonl files found in {results_dir}")
        sys.exit(1)

    rows = []
    for fpath in judged_files:
        dataset_name = fpath.stem.replace(".judged", "")
        total = 0
        success = 0
        sum_score = 0.0
        sum_h = 0.0
        sum_a = 0.0
        sum_d = 0.0
        defense_methods = set()

        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if record.get("success", False):
                    success += 1
                sum_score += record.get("judge_score", 0)
                meta = record.get("metadata", {})
                sum_h += meta.get("harmfulness_score", 0)
                sum_a += meta.get("alignment_score", 0)
                sum_d += meta.get("detail_score", 0)
                dm = record.get("defense_method", "Unknown")
                defense_methods.add(dm)

        asr = (success / total * 100) if total > 0 else 0
        avg_score = sum_score / total if total > 0 else 0
        avg_h = sum_h / total if total > 0 else 0
        avg_a = sum_a / total if total > 0 else 0
        avg_d = sum_d / total if total > 0 else 0
        rows.append({
            "dataset": dataset_name,
            "total": total,
            "success": success,
            "asr": asr,
            "avg_score": avg_score,
            "avg_h": avg_h,
            "avg_a": avg_a,
            "avg_d": avg_d,
            "defenses": ", ".join(sorted(defense_methods)),
        })

    # Print table
    header = f"{'Dataset':<22} {'Total':>6} {'Success':>8} {'ASR':>8}  {'H':>6}  {'A':>6}  {'D':>6}  {'Defenses'}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    total_all = 0
    success_all = 0
    h_all = 0.0
    a_all = 0.0
    d_all = 0.0
    for row in rows:
        print(
            f"{row['dataset']:<22} "
            f"{row['total']:>6} "
            f"{row['success']:>8} "
            f"{row['asr']:>7.2f}% "
            f"{row['avg_h']:>6.2f} "
            f"{row['avg_a']:>6.2f} "
            f"{row['avg_d']:>6.2f} "
            f"{row['defenses']}"
        )
        total_all += row["total"]
        success_all += row["success"]
        h_all += row["total"] * row["avg_h"]
        a_all += row["total"] * row["avg_a"]
        d_all += row["total"] * row["avg_d"]

    print(sep)
    overall_asr = (success_all / total_all * 100) if total_all > 0 else 0
    overall_h = h_all / total_all if total_all > 0 else 0
    overall_a = a_all / total_all if total_all > 0 else 0
    overall_d = d_all / total_all if total_all > 0 else 0
    print(
        f"{'OVERALL':<22} "
        f"{total_all:>6} "
        f"{success_all:>8} "
        f"{overall_asr:>7.2f}% "
        f"{overall_h:>6.2f} "
        f"{overall_a:>6.2f} "
        f"{overall_d:>6.2f}"
    )
    print(sep)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "omni_safe_vtool"
        )
    calc_asr(target)
