from __future__ import annotations

import json
from typing import Any

from judge.safety_judge import judge_safety_response


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    del data_source
    extra_info = dict(extra_info or {})
    tool_extra_fields = dict(extra_info.get("tool_extra_fields") or {})
    tool_trace = tool_extra_fields.get("tool_trace") or extra_info.get("tool_trace")

    judge_result = judge_safety_response(
        response_text=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        tool_trace=tool_trace,
    )
    if isinstance(tool_trace, dict):
        tool_trace["judge_result"] = judge_result.to_dict()

    return {
        "score": judge_result.total_score,
        "reward_extra_info": judge_result.to_dict(),
        **judge_result.to_dict(),
        "ground_truth": json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth,
    }
