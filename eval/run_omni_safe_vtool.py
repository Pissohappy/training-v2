from __future__ import annotations

import argparse
import asyncio
import copy
import io
import json
import re
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval.config import parse_json_dict
    from eval.server import OpenAIChatServerManager
    from eval.result_io import append_jsonl
    from recipe.safe_vtool.common import (
        build_filtered_tool_config,
        ensure_safety_prompt,
        normalize_ablation_mode,
    )
    from verl.tools.utils.tool_registry import initialize_tools_from_config
else:
    from .config import parse_json_dict
    from .server import OpenAIChatServerManager
    from .result_io import append_jsonl
    from recipe.safe_vtool.common import (
        build_filtered_tool_config,
        ensure_safety_prompt,
        normalize_ablation_mode,
    )
    from verl.tools.utils.tool_registry import initialize_tools_from_config


DEFAULT_SERVER_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TOOL_CONFIG_PATH = str(
    Path(__file__).resolve().parent.parent / "recipe" / "safe_vtool" / "safety_tools_config.yaml"
)
DEFAULT_OMNI_ROOT = "/mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM"


@dataclass
class OmniExample:
    index: int
    uid: str
    query: str
    raw_prompt: list[dict[str, Any]]
    multi_modal_data: dict[str, Any]
    tools_kwargs: dict[str, Any]
    original_record: dict[str, Any]
    absolute_image_path: str


@dataclass
class SimpleFunctionCall:
    name: str
    arguments: str


def _maybe_parse_json_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _extract_text_tool_calls(content: str) -> tuple[list[SimpleFunctionCall], dict[str, Any]]:
    """Best-effort fallback parser for malformed <tool_call> text blocks."""
    if "<tool_call>" not in content:
        return [], {
            "fallback_used": False,
            "recovered_tool_calls": 0,
            "truncated_tool_call_block": False,
            "text_tool_call_detected": False,
        }

    blocks = []
    complete_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", content, flags=re.DOTALL)
    truncated = False
    if complete_blocks:
        blocks.extend(complete_blocks)

    if not complete_blocks:
        start = content.find("<tool_call>")
        if start != -1:
            blocks.append(content[start + len("<tool_call>") :])
            truncated = True

    recovered = []
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue

        name_match = re.match(r"\s*([^\n<]+)", stripped)
        if not name_match:
            continue
        name = name_match.group(1).strip()
        args = {}
        arg_pattern = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            flags=re.DOTALL,
        )
        for key, value in re.findall(arg_pattern, stripped):
            args[key.strip()] = _maybe_parse_json_value(value)

        if name:
            recovered.append(SimpleFunctionCall(name=name, arguments=json.dumps(args, ensure_ascii=False)))

    return recovered, {
        "fallback_used": bool(recovered),
        "recovered_tool_calls": len(recovered),
        "truncated_tool_call_block": truncated,
        "text_tool_call_detected": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SafeVTool on OmniSafeBench-MM test cases.")
    parser.add_argument("--test-cases-file", nargs="+", required=True, help="OmniSafeBench-MM test_cases.jsonl file(s).")
    parser.add_argument("--omni-root", default=DEFAULT_OMNI_ROOT, help="OmniSafeBench-MM repository root used to resolve relative image paths.")
    parser.add_argument("--model", required=True, help="Model identifier used for request/model bookkeeping.")
    parser.add_argument("--server-base-url", default=DEFAULT_SERVER_BASE_URL)
    parser.add_argument("--server-api-key", default="EMPTY")
    parser.add_argument("--server-model", default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--request-extra-body", default="{}")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True, help="Detailed SafeVTool trace JSONL output.")
    parser.add_argument(
        "--responses-output",
        default=None,
        help="Judge-ready Omni ModelResponse JSONL output. Defaults to <output>.omni_responses.jsonl.",
    )
    parser.add_argument("--tool-config-path", default=DEFAULT_TOOL_CONFIG_PATH)
    parser.add_argument("--ablation-mode", default="full_safevtool")
    parser.add_argument("--max-tool-calls", type=int, default=4)
    parser.add_argument(
        "--include-conversation-trace",
        action="store_true",
        help="Persist full per-turn message/tool trace in the detailed output.",
    )
    return parser.parse_args()


def build_sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": 1024,
    }


def _resolve_image_path(image_path: str, *, omni_root: Path) -> str:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((omni_root / candidate).resolve())


def _load_jsonl_records(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_str in paths:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse JSONL line {line_no} from {path}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(record).__name__}")
                records.append(record)
    return records


def _build_examples(
    *,
    records: list[dict[str, Any]],
    omni_root: Path,
    ablation_mode: str,
    offset: int,
    limit: int | None,
) -> list[OmniExample]:
    total = len(records)
    start = min(max(offset, 0), total)
    stop = total if limit is None else min(total, start + max(limit, 0))
    examples: list[OmniExample] = []

    for dataset_index in range(start, stop):
        record = copy.deepcopy(records[dataset_index])
        test_case_id = str(record.get("test_case_id") or dataset_index)
        prompt = str(record.get("prompt") or "")
        absolute_image_path = _resolve_image_path(str(record.get("image_path") or ""), omni_root=omni_root)
        image = Image.open(absolute_image_path).convert("RGB")

        metadata = copy.deepcopy(record.get("metadata") or {})
        metadata.update(
            {
                "sample_id": test_case_id,
                "test_case_id": test_case_id,
                "source_dataset": "OmniSafeBench-MM",
                "ablation_mode": ablation_mode,
                "image_path": absolute_image_path,
                "jailbreak_image_path": metadata.get("jailbreak_image_path") or str(record.get("image_path") or ""),
            }
        )

        raw_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        examples.append(
            OmniExample(
                index=dataset_index,
                uid=test_case_id,
                query=prompt,
                raw_prompt=raw_prompt,
                multi_modal_data={"image": [image]},
                tools_kwargs={"metadata": json.dumps(metadata, ensure_ascii=False)},
                original_record=record,
                absolute_image_path=absolute_image_path,
            )
        )

    return examples


def _to_endpoint_messages(messages: list[dict[str, Any]], images: list[Any]) -> list[dict[str, Any]]:
    import base64
    import io

    endpoint_messages: list[dict[str, Any]] = []
    image_index = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            endpoint_messages.append({"role": message["role"], "content": content})
            continue
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                image = images[image_index]
                image_index += 1
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append({"type": "text", "text": str(item.get("text", ""))})
        endpoint_messages.append({"role": message["role"], "content": parts})
    return endpoint_messages


def _render_tool_message(tool_response) -> dict[str, Any]:
    if tool_response.image:
        content: list[dict[str, Any]] = [{"type": "image"}]
        if tool_response.text:
            content.append({"type": "text", "text": tool_response.text})
        return {"role": "tool", "content": content}
    return {"role": "tool", "content": tool_response.text or ""}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _snapshot_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalized_content: Any = content
        elif isinstance(content, list):
            normalized_parts: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    normalized_parts.append({"type": "text", "text": str(item)})
                    continue
                item_type = item.get("type")
                if item_type == "image":
                    normalized_parts.append({"type": "image"})
                elif item_type == "text":
                    normalized_parts.append({"type": "text", "text": str(item.get("text", ""))})
                else:
                    normalized_parts.append(_json_safe(item))
            normalized_content = normalized_parts
        else:
            normalized_content = str(content)
        snapshots.append({"role": str(message.get("role", "")), "content": normalized_content})
    return snapshots


def _extract_reasoning_and_answer(response_text: str) -> tuple[str | None, str | None, str]:
    text = response_text or ""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        reasoning = (match.group(1) or "").strip() or None
        stripped = re.sub(r"<think>.*?</think>", "", text, count=1, flags=re.IGNORECASE | re.DOTALL).strip()
        return reasoning, stripped or text, "split_by_tag"
    return None, None, "no_reasoning_tag"


def _extract_safe_vtool_final_answer(response_text: str) -> str | None:
    if not response_text:
        return None
    match = re.search(r"FINAL_RESPONSE:\s*(.*)", response_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    value = (match.group(1) or "").strip()
    return value or None


async def _run_sample(
    *,
    example: OmniExample,
    server_manager: OpenAIChatServerManager,
    sampling_params: dict[str, Any],
    request_extra_body: dict[str, Any],
    tools: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    ablation_mode: str,
    max_tool_calls: int,
    include_conversation_trace: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = ensure_safety_prompt(copy.deepcopy(example.raw_prompt), ablation_mode=ablation_mode)
    images = list(example.multi_modal_data.get("image") or example.multi_modal_data.get("images") or [])
    eval_agent_data = type(
        "EvalAgentData",
        (),
        {
            "image_data": images,
            "tools_kwargs": example.tools_kwargs,
            "messages": messages,
            "extra_fields": {},
        },
    )()

    tool_trace = {
        "sample_id": example.uid,
        "ablation_mode": ablation_mode,
        "tool_steps": [],
        "final_response": "",
        "tool_call_parse_stats": {
            "fallback_used": False,
            "recovered_tool_calls": 0,
            "truncated_tool_call_block": False,
            "text_tool_call_detected": False,
        },
    }
    conversation_trace: list[dict[str, Any]] = []

    generated_text = ""
    for turn_index in range(max_tool_calls + 1):
        trace_turn: dict[str, Any] = {
            "turn_index": turn_index,
            "messages_before_request": _snapshot_messages(messages),
        }
        output = await server_manager.generate(
            messages=_to_endpoint_messages(messages, images),
            sampling_params=sampling_params,
            tools=tool_schemas or None,
            extra_body=request_extra_body,
        )
        generated_text = output.text.strip()
        trace_turn["assistant_text"] = generated_text
        trace_turn["finish_reason"] = output.finish_reason

        tool_calls = []
        raw_tool_calls = ((output.raw_response.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
        trace_turn["raw_tool_calls"] = _json_safe(raw_tool_calls)
        for call in raw_tool_calls:
            function = call.get("function") or {}
            tool_calls.append(SimpleFunctionCall(name=str(function.get("name")), arguments=str(function.get("arguments") or "{}")))
        parse_stats = {
            "fallback_used": False,
            "recovered_tool_calls": 0,
            "truncated_tool_call_block": False,
            "text_tool_call_detected": False,
        }
        if not tool_calls:
            tool_calls, parse_stats = _extract_text_tool_calls(generated_text)
        trace_turn["parsed_tool_calls"] = [{"name": tool_call.name, "arguments": tool_call.arguments} for tool_call in tool_calls]
        trace_turn["tool_call_parse_stats"] = parse_stats
        for key, value in parse_stats.items():
            if isinstance(value, bool):
                tool_trace["tool_call_parse_stats"][key] = tool_trace["tool_call_parse_stats"][key] or value
            elif isinstance(value, int):
                tool_trace["tool_call_parse_stats"][key] += value

        messages.append({"role": "assistant", "content": generated_text})
        if not tool_calls:
            trace_turn["tool_results"] = []
            trace_turn["messages_after_turn"] = _snapshot_messages(messages)
            conversation_trace.append(trace_turn)
            break

        tool_results: list[dict[str, Any]] = []
        for tool_call in tool_calls[:max_tool_calls]:
            tool = tools.get(tool_call.name)
            if tool is None:
                tool_results.append(
                    {
                        "tool_name": tool_call.name,
                        "arguments": _json_safe(tool_call.arguments),
                        "success": False,
                        "error": "Tool schema was provided to the model, but no local tool implementation was found.",
                    }
                )
                continue
            try:
                arguments = json.loads(tool_call.arguments)
                if not isinstance(arguments, dict):
                    arguments = {}
            except json.JSONDecodeError:
                arguments = {}
            instance_id, _ = await tool.create()
            tool_response, _, metadata = await tool.execute(
                instance_id,
                arguments,
                agent_data=eval_agent_data,
            )
            await tool.release(instance_id)
            tool_trace["tool_steps"].append(
                {
                    "tool_name": tool_call.name,
                    "arguments": arguments,
                    "observation": {
                        "text": tool_response.text,
                        "image_count": len(tool_response.image or []),
                        "video_count": len(tool_response.video or []),
                    },
                    "success": not str(tool_response.text or "").startswith("Error"),
                    "metadata": metadata or {},
                }
            )
            tool_results.append(
                {
                    "tool_name": tool_call.name,
                    "arguments": _json_safe(arguments),
                    "observation": {
                        "text": tool_response.text,
                        "image_count": len(tool_response.image or []),
                        "video_count": len(tool_response.video or []),
                    },
                    "success": not str(tool_response.text or "").startswith("Error"),
                    "metadata": _json_safe(metadata or {}),
                }
            )
            messages.append(_render_tool_message(tool_response))
            for image in tool_response.image or []:
                images.append(image)
            eval_agent_data.image_data = images
            eval_agent_data.messages = messages
            eval_agent_data.extra_fields.setdefault("safe_vtool_tool_state", {}).setdefault("trace", []).append(
                {"tool_name": tool_call.name, "metadata": metadata or {}}
            )
        trace_turn["tool_results"] = tool_results
        trace_turn["messages_after_turn"] = _snapshot_messages(messages)
        conversation_trace.append(trace_turn)

    reasoning_trace, answer_after_reasoning, response_parse_status = _extract_reasoning_and_answer(generated_text)
    final_answer_text = _extract_safe_vtool_final_answer(answer_after_reasoning or generated_text) or answer_after_reasoning or generated_text

    metadata = copy.deepcopy(example.original_record.get("metadata") or {})
    metadata.update(
        {
            "attack_method": metadata.get("attack_method"),
            "original_prompt": metadata.get("original_prompt"),
            "jailbreak_prompt": metadata.get("jailbreak_prompt") or example.query,
            "jailbreak_image_path": metadata.get("jailbreak_image_path") or example.original_record.get("image_path"),
            "original_image_path": metadata.get("original_image_path"),
            "image_path": example.absolute_image_path,
            "source_dataset": "OmniSafeBench-MM",
            "ablation_mode": ablation_mode,
            "test_case_id": example.uid,
            "defense_method": "None",
        }
    )
    tool_trace["final_response"] = generated_text

    detailed_record = {
        "index": example.index,
        "uid": example.uid,
        "test_case_id": example.uid,
        "model": server_manager.model,
        "query": example.query,
        "image_path": example.absolute_image_path,
        "ablation_mode": ablation_mode,
        "generated_text": generated_text,
        "final_answer_text": final_answer_text,
        "reasoning_trace": reasoning_trace,
        "response_parse_status": response_parse_status,
        "metadata": metadata,
        "tool_trace": tool_trace,
        "tool_call_parse_stats": tool_trace["tool_call_parse_stats"],
    }
    if include_conversation_trace:
        detailed_record["conversation_trace"] = conversation_trace

    judge_ready_record = {
        "test_case_id": example.uid,
        "model_response": generated_text,
        "model_name": server_manager.model,
        "metadata": metadata,
        "reasoning_trace": reasoning_trace,
        "final_answer": final_answer_text,
        "response_parse_status": response_parse_status,
    }
    return detailed_record, judge_ready_record


async def main_async() -> None:
    args = parse_args()
    ablation_mode = normalize_ablation_mode(args.ablation_mode)
    omni_root = Path(args.omni_root).resolve()
    request_extra_body = parse_json_dict(args.request_extra_body)
    records = _load_jsonl_records(args.test_cases_file)
    examples = _build_examples(
        records=records,
        omni_root=omni_root,
        ablation_mode=ablation_mode,
        offset=args.offset,
        limit=args.limit,
    )

    responses_output = args.responses_output or f"{args.output}.omni_responses.jsonl"
    server_manager = OpenAIChatServerManager.create(
        base_url=args.server_base_url,
        api_key=args.server_api_key,
        model=args.server_model,
        request_timeout=args.request_timeout,
        default_extra_body=request_extra_body,
    )
    if not server_manager.model:
        server_manager.model = await server_manager.resolve_model(args.model)

    with tempfile.TemporaryDirectory(prefix="safe_vtool_omni_tools_") as temp_dir:
        filtered_tool_config = build_filtered_tool_config(
            base_config_path=args.tool_config_path,
            ablation_mode=ablation_mode,
            output_path=str(Path(temp_dir) / f"{ablation_mode}.yaml"),
            vlm_tool_client_overrides={
                "base_url": args.server_base_url,
                "model": server_manager.model,
                "api_key": args.server_api_key,
            },
        )
        with redirect_stdout(io.StringIO()):
            tool_list = initialize_tools_from_config(filtered_tool_config)
        tools = {tool.name: tool for tool in tool_list}
        tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

        progress = tqdm(examples, desc="SafeVTool eval", unit="sample")
        for example in progress:
            progress.set_postfix_str(f"id={example.uid}")
            detailed_record, judge_ready_record = await _run_sample(
                example=example,
                server_manager=server_manager,
                sampling_params=build_sampling_params(args),
                request_extra_body=request_extra_body,
                tools=tools,
                tool_schemas=tool_schemas,
                ablation_mode=ablation_mode,
                max_tool_calls=args.max_tool_calls,
                include_conversation_trace=args.include_conversation_trace,
            )
            append_jsonl(args.output, detailed_record)
            append_jsonl(responses_output, judge_ready_record)
        progress.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
