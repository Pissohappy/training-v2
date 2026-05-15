from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

TOOL_ALIAS_TO_NAME = {
    "safety_ocr": "safety_ocr_tool",
    "crop_zoom": "crop_zoom_tool",
    "grounding": "grounding_tool",
    "layout_parse": "layout_parse_tool",
    "policy_check": "policy_check_tool",
}

ABLATION_MODES = (
    "no_tools",
    "self_vlm_tools",
    "external_tools",
    "full_safevtool",
    "oracle_tools",
)

ABLATION_TOOLS = {
    "no_tools": [],
    "self_vlm_tools": [
        "safety_ocr_tool",
        "crop_zoom_tool",
        "grounding_tool",
        "layout_parse_tool",
    ],
    "external_tools": [
        "safety_ocr_tool",
        "crop_zoom_tool",
        "grounding_tool",
        "layout_parse_tool",
    ],
    "full_safevtool": [
        "safety_ocr_tool",
        "crop_zoom_tool",
        "grounding_tool",
        "layout_parse_tool",
        "policy_check_tool",
    ],
    "oracle_tools": [
        "safety_ocr_tool",
        "grounding_tool",
        "policy_check_tool",
    ],
}


def normalize_ablation_mode(value: str | None) -> str:
    if not value:
        return "full_safevtool"
    if value not in ABLATION_TOOLS:
        raise ValueError(f"Unknown ablation mode: {value}. Expected one of {ABLATION_MODES}")
    return value


def get_tool_names_for_ablation(ablation_mode: str | None) -> list[str]:
    return list(ABLATION_TOOLS[normalize_ablation_mode(ablation_mode)])


def get_tool_notice(ablation_mode: str | None) -> str:
    normalized = normalize_ablation_mode(ablation_mode)
    tool_names = get_tool_names_for_ablation(normalized)
    if not tool_names:
        return "Available tools in this run: none. Rely only on the visible image and the user request."
    return f"Available tools in this run: {', '.join(tool_names)}."


def load_safety_prompt() -> str:
    prompt_path = Path(__file__).resolve().with_name("safety_agent_prompt.txt")
    return prompt_path.read_text(encoding="utf-8").strip()


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return str(content or "").strip()


def ensure_safety_prompt(messages: list[dict[str, Any]], *, ablation_mode: str | None) -> list[dict[str, Any]]:
    normalized = normalize_ablation_mode(ablation_mode)
    system_text = f"{load_safety_prompt()}\n\n{get_tool_notice(normalized)}"
    if messages and messages[0].get("role") == "system":
        first_text = extract_text_content(messages[0].get("content"))
        if load_safety_prompt() in first_text:
            return messages
        updated = dict(messages[0])
        updated["content"] = f"{system_text}\n\n{first_text}".strip()
        return [updated] + messages[1:]
    return [{"role": "system", "content": system_text}] + messages


def parse_metadata_blob(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return dict(metadata)
    if isinstance(metadata, str) and metadata.strip():
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def build_filtered_tool_config(
    *,
    base_config_path: str,
    ablation_mode: str | None,
    output_path: str,
    vlm_tool_client_overrides: dict[str, Any] | None = None,
) -> str:
    normalized_mode = normalize_ablation_mode(ablation_mode)
    allowed = set(get_tool_names_for_ablation(normalized_mode))
    config = OmegaConf.load(base_config_path)
    if vlm_tool_client_overrides:
        for key, value in vlm_tool_client_overrides.items():
            config.vlm_tool_client[key] = value
    tools = []
    for entry in config.tools:
        tool_alias = entry.get("config", {}).get("tool_alias")
        name = TOOL_ALIAS_TO_NAME.get(tool_alias or "", entry.get("tool_schema", {}).get("function", {}).get("name"))
        if name is None:
            name = entry.class_name.rsplit(".", 1)[-1]
        if name in allowed:
            if normalized_mode == "oracle_tools" and tool_alias in {"safety_ocr", "grounding", "policy_check"}:
                entry.config.allow_gold = True
            elif normalized_mode == "external_tools":
                if tool_alias == "grounding":
                    entry.config.default_backend = "ocr_layout_fallback"
                    entry.config.fallback_order = ["ocr_layout_fallback", "heuristic_fallback"]
                elif tool_alias == "layout_parse":
                    entry.config.default_backend = "ocr_block_layout"
                    entry.config.fallback_order = ["ocr_block_layout", "cv_connected_components", "heuristic_fallback"]
            tools.append(entry)
    filtered = OmegaConf.create(
        {
            "vlm_tool_client": config.get("vlm_tool_client"),
            "tool_backends": config.get("tool_backends"),
            "ablation_modes": config.get("ablation_modes"),
            "tools": tools,
        }
    )
    OmegaConf.save(filtered, output_path)
    return output_path
