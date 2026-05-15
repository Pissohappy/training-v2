from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.result_io import iter_jsonl
from eval.safety_metrics import aggregate_safety_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SafeVTool result JSONL files.")
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = aggregate_safety_records(list(iter_jsonl(args.results_file)))
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
