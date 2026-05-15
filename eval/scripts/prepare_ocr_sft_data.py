#!/usr/bin/env python3
"""
Prepare OCR SFT data in two variants:

1. Canonical OpenAI tool-use format
2. LLaMA-Factory-compatible openai-format dataset

The two outputs intentionally differ:
- OpenAI tool-use keeps `assistant.tool_calls` + `tool` messages.
- LLaMA-Factory variant keeps the role/tag layout expected by its current converter.
"""

import json
import re
import sys
from pathlib import Path


LLAMAFACTORY_SRC = "/mnt/disk1/szchen/VLMAlignment/LlamaFactory/src"


def normalize_tool_arguments(arguments):
    """Return tool-call arguments as a dict."""
    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        arguments = arguments.strip()
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tool arguments JSON: {arguments}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"Tool arguments must decode to a dict, got {type(parsed).__name__}")

        return parsed

    raise ValueError(f"Unsupported tool arguments type: {type(arguments).__name__}")


def normalize_tool_calls(raw_tool_calls: list[dict]) -> list[dict] | None:
    """Normalize raw tool calls into a shared intermediate representation."""
    tool_calls = []
    for tc in raw_tool_calls:
        try:
            arguments = normalize_tool_arguments(tc["function"]["arguments"])
            name = tc["function"]["name"]
        except (KeyError, ValueError, TypeError) as exc:
            print(f"Warning: skipping sample with malformed tool arguments: {exc}")
            return None

        tool_calls.append(
            {
                "id": tc.get("id", f"call_{len(tool_calls)}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments_dict": arguments,
                    "arguments_json": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )

    return tool_calls


def normalize_thought_block(content: str) -> str:
    """Normalize <think> blocks to the exact boundary expected by glm4_5v."""
    if not content:
        return content

    match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    if not match:
        return content

    thought = (match.group(1) or "").strip("\n")
    normalized = f"<think>\n{thought}\n</think>\n\n"
    return content[: match.start()] + normalized + content[match.end() :].lstrip("\n")


def extract_normalized_thought(content: str) -> str:
    """Extract only the normalized thought block, if present."""
    content = normalize_thought_block(content)
    match = re.search(r"<think>\n(.*?)\n</think>\n\n", content, flags=re.DOTALL)
    if not match:
        return ""
    return f"<think>\n{match.group(1)}\n</think>\n\n"


def extract_system_and_user(turn0_msgs: list[dict]) -> tuple[str, str, list[dict] | str]:
    """Extract system content and both LF/OpenAI user content variants."""
    system_content = ""
    user_content_lf = ""
    user_content_openai: list[dict] | str = ""

    for msg in turn0_msgs:
        if msg["role"] == "system" and not system_content:
            system_content = msg.get("content", "")
        elif msg["role"] == "user" and not user_content_lf:
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                normalized_content = []
                for c in content:
                    if c.get("type") == "image":
                        parts.append("<image>")
                        normalized_content.append({"type": "image"})
                    elif c.get("type") == "text":
                        text = c["text"]
                        parts.append(text)
                        normalized_content.append({"type": "text", "text": text})
                user_content_lf = "".join(parts)
                user_content_openai = normalized_content
            else:
                user_content_lf = str(content)
                user_content_openai = str(content)

    return system_content, user_content_lf, user_content_openai


def extract_llamafactory_sample(trace_record: dict) -> dict | None:
    """Extract a LLaMA-Factory-compatible sample."""
    conv = trace_record.get("conversation_trace", [])
    if len(conv) < 2:
        return None

    image_path = trace_record.get("image_path", "")
    if not image_path:
        return None

    turn0_msgs = conv[0].get("messages_before_request", [])
    system_content, user_content, _ = extract_system_and_user(turn0_msgs)
    messages = []

    if system_content:
        messages.append({"role": "system", "content": system_content})

    if not user_content:
        return None
    messages.append({"role": "user", "content": user_content})

    turn0 = conv[0]
    assistant_text_0 = normalize_thought_block(turn0.get("assistant_text", ""))
    raw_tool_calls = turn0.get("raw_tool_calls", [])
    if not raw_tool_calls:
        return None

    tool_calls = normalize_tool_calls(raw_tool_calls)
    if tool_calls is None:
        return None

    tool_calls_payload = [
        {
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments_dict"],
        }
        for tc in tool_calls
    ]
    function_content = extract_normalized_thought(assistant_text_0) + json.dumps(tool_calls_payload, ensure_ascii=False)

    messages.append(
        {
            "role": "function",
            "content": function_content,
        }
    )

    tool_results = turn0.get("tool_results", [])
    for tr in tool_results:
        obs_text = tr.get("observation", {}).get("text", "")
        messages.append({"role": "observation", "content": obs_text})

    turn1 = conv[1] if len(conv) > 1 else None
    if turn1 and turn1.get("finish_reason") == "stop":
        final_text = normalize_thought_block(turn1.get("assistant_text", ""))
        messages.append({"role": "assistant", "content": final_text})
    else:
        final_text = normalize_thought_block(trace_record.get("generated_text", ""))
        if final_text:
            messages.append({"role": "assistant", "content": final_text})

    return {
        "messages": messages,
        "images": [image_path],
    }


def extract_openai_tool_use_sample(trace_record: dict) -> dict | None:
    """Extract a canonical OpenAI tool-use sample."""
    conv = trace_record.get("conversation_trace", [])
    if len(conv) < 2:
        return None

    image_path = trace_record.get("image_path", "")
    if not image_path:
        return None

    turn0_msgs = conv[0].get("messages_before_request", [])
    system_content, _, user_content = extract_system_and_user(turn0_msgs)
    messages = []

    if system_content:
        messages.append({"role": "system", "content": system_content})
    if not user_content:
        return None
    messages.append({"role": "user", "content": user_content})

    turn0 = conv[0]
    assistant_text_0 = normalize_thought_block(turn0.get("assistant_text", ""))
    raw_tool_calls = turn0.get("raw_tool_calls", [])
    if not raw_tool_calls:
        return None

    tool_calls = normalize_tool_calls(raw_tool_calls)
    if tool_calls is None:
        return None

    messages.append(
        {
            "role": "assistant",
            "content": assistant_text_0,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments_json"],
                    },
                }
                for tc in tool_calls
            ],
        }
    )

    tool_results = turn0.get("tool_results", [])
    for tc, tr in zip(tool_calls, tool_results):
        obs_text = tr.get("observation", {}).get("text", "")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["function"]["name"],
                "content": obs_text,
            }
        )

    turn1 = conv[1] if len(conv) > 1 else None
    if turn1 and turn1.get("finish_reason") == "stop":
        final_text = normalize_thought_block(turn1.get("assistant_text", ""))
        messages.append({"role": "assistant", "content": final_text})
    else:
        final_text = normalize_thought_block(trace_record.get("generated_text", ""))
        if final_text:
            messages.append({"role": "assistant", "content": final_text})

    return {
        "messages": messages,
        "images": [image_path],
    }


def resolve_output_paths(output_path: Path) -> tuple[Path, Path]:
    """Convert a user-provided path into paired OpenAI/LF output paths."""
    if output_path.suffix == ".jsonl":
        base = output_path.with_suffix("")
    else:
        base = output_path
    return (
        base.with_name(base.name + "_openai.jsonl"),
        base.with_name(base.name + "_llamafactory.jsonl"),
    )


def render_glm4_5v_messages(messages: list[dict]) -> str:
    """Render messages with the actual glm4_5v template formatters."""
    if LLAMAFACTORY_SRC not in sys.path:
        sys.path.insert(0, LLAMAFACTORY_SRC)

    from llamafactory.data.data_utils import Role
    from llamafactory.data.template import TEMPLATES

    template = TEMPLATES["glm4_5v"]
    rendered_parts = []
    system = ""
    start_idx = 0

    if messages and messages[0]["role"] == "system":
        system = messages[0]["content"]
        start_idx = 1

    for i, message in enumerate(messages[start_idx:]):
        elements = []
        if i == 0:
            elements += template.format_prefix.apply()
            elements += template.format_system.apply(content=system)

        role = message["role"]
        if role == "user":
            elements += template.format_user.apply(content=message["content"], idx=str(i // 2))
        elif role == "assistant":
            elements += template.format_assistant.apply(content=message["content"])
        elif role == "observation":
            elements += template.format_observation.apply(content=message["content"])
        elif role == "function":
            elements += template.format_function.apply(
                content=message["content"],
                thought_words=template.thought_words,
                tool_call_words=template.tool_call_words,
            )
        else:
            raise ValueError(f"Unsupported role for rendering: {role}")

        rendered_parts.extend([item for item in elements if isinstance(item, str)])

    return "".join(rendered_parts)


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/mnt/disk1/szchen/VLMAlignment/training-v2/eval/results/omni_safe_vtool"
    )
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "/mnt/disk1/szchen/VLMAlignment/training-v2/eval/sft_data/ocr_sft_train.jsonl"
    )
    openai_output_path, llamafactory_output_path = resolve_output_paths(output_path)

    target_datasets = ["figstep", "arttextfigstep"]

    openai_samples = []
    llamafactory_samples = []
    stats = {}

    for ds_name in target_datasets:
        trace_f = results_dir / ds_name / f"{ds_name}.trace.jsonl"
        if not trace_f.exists():
            print(f"Warning: {trace_f} not found, skipping")
            continue

        ds_ocr_yes = 0
        ds_ocr_no = 0
        ds_total = 0

        with open(trace_f) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ds_total += 1

                tools = [
                    s["tool_name"]
                    for s in record.get("tool_trace", {}).get("tool_steps", [])
                ]
                has_ocr = "safety_ocr_tool" in tools

                if has_ocr:
                    ds_ocr_yes += 1
                    openai_sample = extract_openai_tool_use_sample(record)
                    llamafactory_sample = extract_llamafactory_sample(record)
                    if openai_sample:
                        openai_samples.append(openai_sample)
                    if llamafactory_sample:
                        llamafactory_samples.append(llamafactory_sample)
                else:
                    ds_ocr_no += 1

        stats[ds_name] = {
            "total": ds_total,
            "ocr_yes": ds_ocr_yes,
            "ocr_no": ds_ocr_no,
            "ocr_rate": ds_ocr_yes / ds_total * 100 if ds_total else 0,
        }

    # Print stats
    print("=" * 60)
    print("OCR Calling Statistics (before SFT)")
    print("=" * 60)
    for ds_name, s in stats.items():
        print(
            f"  {ds_name:<20}: {s['ocr_yes']}/{s['total']} = {s['ocr_rate']:.1f}% "
            f"(correct={s['ocr_yes']}, missed={s['ocr_no']})"
        )
    print(f"\n  Total OpenAI tool-use samples: {len(openai_samples)}")
    print(f"  Total LLaMA-Factory samples: {len(llamafactory_samples)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(openai_output_path, "w", encoding="utf-8") as f:
        for sample in openai_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(llamafactory_output_path, "w", encoding="utf-8") as f:
        for sample in llamafactory_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(openai_samples)} OpenAI tool-use samples to {openai_output_path}")
    print(f"Saved {len(llamafactory_samples)} LLaMA-Factory samples to {llamafactory_output_path}")

    if openai_samples:
        print("\n" + "=" * 60)
        print("Sample OpenAI tool-use example:")
        print("=" * 60)
        ex = openai_samples[0]
        print(f"  Image: {ex['images'][0]}")
        for i, msg in enumerate(ex["messages"]):
            role = msg["role"]
            content = str(msg["content"])[:300]
            print(f"  [{i}] {role}: {content}...")
        print()

    if llamafactory_samples:
        print("\n" + "=" * 60)
        print("Sample LLaMA-Factory example:")
        print("=" * 60)
        ex = llamafactory_samples[0]
        print(f"  Image: {ex['images'][0]}")
        for i, msg in enumerate(ex["messages"]):
            role = msg["role"]
            content = str(msg["content"])[:300]
            print(f"  [{i}] {role}: {content}...")
        print()

        print("=" * 60)
        print("Rendered glm4_5v training text preview:")
        print("=" * 60)
        print(render_glm4_5v_messages(ex["messages"])[:4000])
        print()


if __name__ == "__main__":
    main()
