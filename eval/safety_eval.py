from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from eval.config import parse_json_dict
    from eval.dataset import load_source_dataset, prepare_dataset_examples
    from eval.result_io import append_jsonl
    from eval.safety_metrics import aggregate_safety_records
    from eval.server import OpenAIChatServerManager
    from judge.safety_judge import judge_safety_response
    from recipe.safe_vtool.common import build_filtered_tool_config, ensure_safety_prompt, normalize_ablation_mode, parse_metadata_blob
    from verl.tools.utils.tool_registry import initialize_tools_from_config
else:
    from .config import parse_json_dict
    from .dataset import load_source_dataset, prepare_dataset_examples
    from .result_io import append_jsonl
    from .safety_metrics import aggregate_safety_records
    from .server import OpenAIChatServerManager
    from judge.safety_judge import judge_safety_response
    from recipe.safe_vtool.common import build_filtered_tool_config, ensure_safety_prompt, normalize_ablation_mode, parse_metadata_blob
    from verl.tools.utils.tool_registry import initialize_tools_from_config


DEFAULT_SERVER_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TOOL_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "recipe" / "safe_vtool" / "safety_tools_config.yaml")


@dataclass
class SimpleFunctionCall:
    name: str
    arguments: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone SafeVTool evaluation runner.")
    parser.add_argument("--data-file", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-base-url", default=DEFAULT_SERVER_BASE_URL)
    parser.add_argument("--server-api-key", default="EMPTY")
    parser.add_argument("--server-model", default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--request-extra-body", default="{}")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--tool-config-path", default=DEFAULT_TOOL_CONFIG_PATH)
    parser.add_argument("--ablation-mode", default="full_safevtool")
    parser.add_argument("--max-tool-calls", type=int, default=4)
    return parser.parse_args()


def build_sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": 1024,
    }


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


async def _run_sample(
    *,
    example,
    server_manager: OpenAIChatServerManager,
    sampling_params: dict[str, Any],
    tool_config_path: str,
    ablation_mode: str,
    max_tool_calls: int,
) -> dict[str, Any]:
    messages = ensure_safety_prompt(copy.deepcopy(example.raw_prompt), ablation_mode=ablation_mode)
    images = list(example.multi_modal_data.get("image") or example.multi_modal_data.get("images") or [])
    tool_list = initialize_tools_from_config(tool_config_path)
    tools = {tool.name: tool for tool in tool_list}
    tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
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
        "judge_result": None,
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
        trace_turn["parsed_tool_calls"] = [{"name": tool_call.name, "arguments": tool_call.arguments} for tool_call in tool_calls]

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
                    "observation": {"text": tool_response.text, "image_count": len(tool_response.image or [])},
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

    metadata = parse_metadata_blob(example.tools_kwargs.get("metadata"))
    judge_result = judge_safety_response(
        response_text=generated_text,
        ground_truth=example.original_record["reward_model"]["ground_truth"],
        extra_info={"safety_gold": example.original_record.get("extra_info", {}).get("safety_gold"), "ablation_mode": ablation_mode},
        tool_trace=tool_trace,
    )
    tool_trace["final_response"] = generated_text
    tool_trace["judge_result"] = judge_result.to_dict()

    return {
        "index": example.index,
        "uid": example.uid,
        "query": example.query,
        "source_dataset": metadata.get("source_dataset") or example.original_record.get("data_source"),
        "ablation_mode": ablation_mode,
        "generated_text": generated_text,
        "conversation_trace": conversation_trace,
        "tool_trace": tool_trace,
        **judge_result.to_dict(),
    }


async def main_async() -> None:
    args = parse_args()
    ablation_mode = normalize_ablation_mode(args.ablation_mode)
    dataset = load_source_dataset(
        data_files=args.data_file,
        dataset_name=None,
        dataset_subset=None,
        dataset_split="train",
    )
    examples = prepare_dataset_examples(dataset, prompt_key=args.prompt_key, image_key=args.image_key, offset=args.offset, limit=args.limit)
    server_manager = OpenAIChatServerManager.create(
        base_url=args.server_base_url,
        api_key=args.server_api_key,
        model=args.server_model,
        request_timeout=args.request_timeout,
        default_extra_body=parse_json_dict(args.request_extra_body),
    )
    if not server_manager.model:
        server_manager.model = await server_manager.resolve_model(args.model)

    with tempfile.TemporaryDirectory(prefix="safe_vtool_tools_") as temp_dir:
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
        results = []
        for example in examples:
            result = await _run_sample(
                example=example,
                server_manager=server_manager,
                sampling_params=build_sampling_params(args),
                tool_config_path=filtered_tool_config,
                ablation_mode=ablation_mode,
                max_tool_calls=args.max_tool_calls,
            )
            append_jsonl(args.output, result)
            results.append(result)

    if args.metrics_output:
        metrics = aggregate_safety_records(results)
        Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
