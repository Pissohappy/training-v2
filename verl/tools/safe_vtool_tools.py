from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from PIL import Image, ImageOps

from recipe.safe_vtool.vlm_tool_client import build_vlm_tool_client
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None


def _tool_schema(
    *,
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> OpenAIFunctionToolSchema:
    parsed_properties = {
        key: OpenAIFunctionPropertySchema(
            type=value["type"],
            description=value.get("description"),
            enum=value.get("enum"),
        )
        for key, value in properties.items()
    }
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name=name,
            description=description,
            parameters=OpenAIFunctionParametersSchema(type="object", properties=parsed_properties, required=required),
        ),
    )


def _safe_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _parse_metadata(agent_data) -> dict[str, Any]:
    if agent_data is None:
        return {}
    metadata = (agent_data.tools_kwargs or {}).get("metadata")
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


def _get_tool_state(agent_data) -> dict[str, Any]:
    if agent_data is None:
        return {}
    if not hasattr(agent_data, "extra_fields") or agent_data.extra_fields is None:
        agent_data.extra_fields = {}
    return agent_data.extra_fields.setdefault("safe_vtool_tool_state", {})


def _state_set(agent_data, key: str, value: Any) -> None:
    _get_tool_state(agent_data)[key] = value


def _state_get(agent_data, key: str, default: Any = None) -> Any:
    return _get_tool_state(agent_data).get(key, default)


def _clip_box(box: list[float] | tuple[float, float, float, float], width: int, height: int) -> list[int]:
    left = max(0, min(width, int(round(float(box[0])))))
    top = max(0, min(height, int(round(float(box[1])))))
    right = max(left + 1, min(width, int(round(float(box[2])))))
    bottom = max(top + 1, min(height, int(round(float(box[3])))))
    return [left, top, right, bottom]


def _flatten_box_values(box: Any) -> list[float]:
    if isinstance(box, (int, float)):
        return [float(box)]
    if isinstance(box, str):
        text = box.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [float(text)]
        return _flatten_box_values(parsed)
    if isinstance(box, dict):
        for key in ("box", "bbox", "bbox_2d", "coords", "coordinates", "points"):
            if key in box:
                return _flatten_box_values(box[key])
        return []
    if isinstance(box, (list, tuple)):
        values: list[float] = []
        for item in box:
            values.extend(_flatten_box_values(item))
        return values
    raise TypeError(f"Unsupported box value type: {type(box).__name__}")


def _normalize_bbox(bbox_2d: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(bbox_2d) != 4:
        raise ValueError("bbox_2d must have four values [x1, y1, x2, y2].")
    left, top, right, bottom = _clip_box(bbox_2d, width, height)
    if left >= right or top >= bottom:
        raise ValueError(f"Invalid bbox after normalization: {(left, top, right, bottom)}")
    return left, top, right, bottom


def _normalize_image_path_value(image_path: Any) -> str | None:
    if image_path is None:
        return None
    if isinstance(image_path, Path):
        return str(image_path)
    text = str(image_path).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _get_image(agent_data, image_path: str | None = None, image_index: int = -1) -> Image.Image:
    image_path = _normalize_image_path_value(image_path)
    images = list((agent_data.image_data if agent_data is not None else None) or [])
    if images:
        return images[image_index].convert("RGB")
    if image_path:
        return Image.open(image_path).convert("RGB")
    raise ValueError("No image is available for this tool call.")


def _ensure_image_path(image: Image.Image, image_path: str | None) -> str:
    image_path = _normalize_image_path_value(image_path)
    if image_path:
        return image_path
    tmp = tempfile.NamedTemporaryFile(prefix="safe_vtool_", suffix=".png", delete=False)
    image.save(tmp.name, format="PNG")
    return tmp.name


def _crop_with_region(image: Image.Image, region: Optional[list[int]]) -> tuple[Image.Image, Optional[list[int]]]:
    if not region:
        return image, None
    left, top, right, bottom = _normalize_bbox(region, image.size[0], image.size[1])
    return image.crop((left, top, right, bottom)), [left, top, right, bottom]


def _resolve_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _extract_user_query(agent_data, fallback: str = "") -> str:
    if agent_data is None:
        return fallback
    for message in reversed(agent_data.messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"]
            merged = " ".join(part for part in texts if part.strip()).strip()
            if merged:
                return merged
    return fallback


def _mean_confidence(blocks: list[dict[str, Any]]) -> float:
    confidences = [float(block.get("confidence", 0.0)) for block in blocks]
    return float(mean(confidences)) if confidences else 0.0


def _redact_text(text: str) -> str:
    return "[redacted]" if text.strip() else ""


def _summarize_ocr_text(raw_text: str) -> str:
    lowered = raw_text.lower()
    if any(token in lowered for token in ["step 1", "step 2", "1.", "2.", "procedure", "instructions"]):
        return "The image contains step-like procedural text."
    if any(token in lowered for token in ["from:", "to:", "subject:", "@", "dear ", "regards", "user:", "assistant:"]):
        return "The image contains chat/email-like text."
    if raw_text.strip():
        return "The image contains text-like content."
    return "No reliable text could be extracted from the image."


def _match_query_tokens(query: str) -> set[str]:
    lowered = query.lower()
    tokens = set(re.findall(r"[a-z0-9_@.-]+", lowered))
    if any(token in lowered for token in ["text", "instruction", "steps", "procedure"]):
        tokens.add("text")
    if any(token in lowered for token in ["email", "mail", "subject", "from", "to"]):
        tokens.add("email")
    if any(token in lowered for token in ["chat", "conversation", "message", "reply"]):
        tokens.add("chat")
    if any(token in lowered for token in ["warning", "poster", "caution", "notice"]):
        tokens.add("poster")
    if any(token in lowered for token in ["table", "grid", "rows", "columns"]):
        tokens.add("table")
    if any(token in lowered for token in ["flowchart", "workflow", "process", "diagram"]):
        tokens.add("flowchart")
    return tokens


def _build_success_payload(*, backend: str, used_gold: bool, error: str | None = None, **payload: Any) -> dict[str, Any]:
    return {"success": error is None, "backend": backend, "used_gold": used_gold, "error": error, **payload}


DEFAULT_POLICY_LIBRARY = {
    "violent_harm": (
        "Do not provide instructions, operational guidance, material lists, optimization advice, or encouragement "
        "for violence, assault, weapons use, poisoning, or other forms of physical harm."
    ),
    "sexual_minor": (
        "Do not assist with sexual, exploitative, or suggestive content involving minors or young-looking persons."
    ),
    "privacy_pii": (
        "Do not expose, identify, aggregate, or facilitate misuse of personal data such as addresses, phone numbers, "
        "emails, ID numbers, or passports."
    ),
    "hate_or_harassment": (
        "Do not generate or amplify hateful, discriminatory, degrading, or harassing content targeting protected groups or individuals."
    ),
}


def _detect_structure_from_ocr(ocr_blocks: list[dict[str, Any]]) -> tuple[str, float]:
    if not ocr_blocks:
        return "unknown", 0.1
    texts = " ".join(str(block.get("text", "")) for block in ocr_blocks).lower()
    if any(token in texts for token in ["from:", "to:", "subject:", "date:"]):
        return "email", 0.85
    if len([block for block in ocr_blocks if re.match(r"^\s*(alice|bob|user|assistant|agent|system)\s*:", str(block.get("text", "")).lower())]) >= 2:
        return "chat", 0.8
    if any(token in texts for token in ["warning", "caution", "safety", "danger"]):
        return "poster", 0.75
    if any(token in texts for token in ["step 1", "step 2", "process", "workflow", "->", "next"]):
        if any(token in texts for token in ["flow", "node", "branch", "arrow", "decision"]):
            return "flowchart", 0.72
        return "document", 0.68
    if len(ocr_blocks) >= 4:
        tops = [block["box"][1] for block in ocr_blocks]
        lefts = [block["box"][0] for block in ocr_blocks]
        if len({top // 25 for top in tops}) >= 2 and len({left // 25 for left in lefts}) >= 2:
            return "table", 0.7
    return "document", 0.55


def _build_relations(blocks: list[dict[str, Any]], structure_type: str) -> tuple[list[dict[str, Any]], list[str]]:
    sorted_blocks = sorted(blocks, key=lambda block: (block["box"][1], block["box"][0]))
    reading_order = [block["id"] for block in sorted_blocks]
    relations = [{"source": current["id"], "target": nxt["id"], "relation": "next"} for current, nxt in zip(sorted_blocks, sorted_blocks[1:], strict=False)]
    if structure_type == "table":
        for left_block in sorted_blocks:
            for right_block in sorted_blocks:
                if left_block["id"] == right_block["id"]:
                    continue
                if abs(left_block["box"][1] - right_block["box"][1]) < 20:
                    relations.append({"source": left_block["id"], "target": right_block["id"], "relation": "same_row"})
                if abs(left_block["box"][0] - right_block["box"][0]) < 20:
                    relations.append({"source": left_block["id"], "target": right_block["id"], "relation": "same_column"})
    elif structure_type == "chat":
        for current, nxt in zip(sorted_blocks, sorted_blocks[1:], strict=False):
            relations.append({"source": current["id"], "target": nxt["id"], "relation": "reply_to"})
    return relations, reading_order


def _connected_components_regions(image: Image.Image) -> list[list[int]]:
    grayscale = ImageOps.grayscale(image)
    if cv2 is not None and np is not None:
        array = np.array(grayscale)
        _, threshold = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 400:
                continue
            regions.append([x, y, x + w, y + h])
        return regions[:20]
    binary = grayscale.point(lambda value: 255 if value < 180 else 0)
    bbox = binary.getbbox()
    return [list(bbox)] if bbox else []


def _convert_box_from_vlm(box: list[Any], width: int, height: int) -> list[int]:
    values = _flatten_box_values(box)
    if len(values) == 8:
        xs = values[0::2]
        ys = values[1::2]
        values = [min(xs), min(ys), max(xs), max(ys)]
    if len(values) != 4:
        raise ValueError(f"Expected 4 box coordinates, got {len(values)} from {box!r}")
    if max(abs(value) for value in values) <= 1.5:
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    return _clip_box(values, width, height)


class _VLMCapableTool(BaseTool):
    def _call_self_vlm(self, *, image_path: str, prompt: str, system_prompt: str, max_tokens: int = 512) -> dict[str, Any]:
        config = dict(self.config.get("vlm_tool_client") or {})
        if not config.get("model"):
            return {"success": False, "backend": "self_vlm", "raw_response": "", "json": None, "error": "No VLM tool model configured."}
        client = build_vlm_tool_client(config)
        return client.call(image_path=image_path, prompt=prompt, system_prompt=system_prompt, temperature=float(config.get("temperature", 0.0)), max_tokens=max_tokens)


class SafetyOCRTool(_VLMCapableTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _tool_schema(
            name="safety_ocr_tool",
            description=(
                "Read the visible text in the image and provide the extracted text in the `ocr_text` argument. "
                "Use this tool when text in the image is relevant for understanding the user's request. "
                "The `ocr_text` field must contain the text you read from the image and must not be empty."
            ),
            properties={
                "region": {"type": "array", "description": "Optional OCR region [x1, y1, x2, y2]."},
                "ocr_text": {
                    "type": "string",
                    "description": "Required. The visible text read from the image. Must not be empty.",
                },
                "ocr_blocks": {"type": "array", "description": "Optional OCR blocks, each with text, box, and confidence."},
                "text_summary": {"type": "string", "description": "Optional summary of the OCR result."},
                "structure_hint": {"type": "string", "description": "Optional structure hint such as document, chat, table, or poster."},
                "confidence": {"type": "number", "description": "Optional overall OCR confidence."},
                "redact_output": {"type": "boolean", "description": "Whether to redact raw OCR text in public observation."},
                "save_private_trace": {"type": "boolean", "description": "Whether to preserve raw OCR text in metadata."},
            },
            required=["ocr_text"],
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        del instance_id
        agent_data = kwargs.get("agent_data")
        metadata = _parse_metadata(agent_data)
        image_path = str(metadata.get("image_path") or "")
        region = parameters.get("region")
        redact_output = _resolve_bool(parameters.get("redact_output"), True)
        save_private_trace = _resolve_bool(parameters.get("save_private_trace"), False)
        raw_text = str(
            parameters.get("ocr_text")
            or parameters.get("text")
            or parameters.get("visual_text_gold")
            or metadata.get("visual_text_gold")
            or ""
        ).strip()
        raw_blocks = list(parameters.get("ocr_blocks") or [])
        text_summary = str(parameters.get("text_summary") or "").strip()
        structure_hint = str(parameters.get("structure_hint") or "").strip()
        confidence = parameters.get("confidence")

        image = None
        if image_path or getattr(agent_data, "image_data", None):
            try:
                image = _get_image(agent_data, image_path=image_path or None)
            except Exception:  # noqa: BLE001
                image = None

        region_used = None
        if region is not None and image is not None:
            try:
                _, region_used = _crop_with_region(image, region)
            except Exception:  # noqa: BLE001
                region_used = None
        elif isinstance(region, list) and len(region) == 4:
            region_used = [int(round(float(value))) for value in region]

        blocks: list[dict[str, Any]] = []
        for block in raw_blocks:
            if not isinstance(block, dict):
                continue
            text = str(block.get("text", "")).strip()
            if not text:
                continue
            raw_box = block.get("box") or region_used
            if raw_box and len(raw_box) == 4:
                if image is not None:
                    box = _clip_box(raw_box, image.size[0], image.size[1])
                else:
                    box = [int(round(float(value))) for value in raw_box]
            elif region_used:
                box = list(region_used)
            elif image is not None:
                box = [0, 0, image.size[0], image.size[1]]
            else:
                box = [0, 0, 0, 0]
            blocks.append(
                {
                    "text": text,
                    "box": box,
                    "confidence": float(block.get("confidence", 1.0)),
                }
            )

        if not blocks and raw_text:
            if region_used:
                box = list(region_used)
            elif image is not None:
                box = [0, 0, image.size[0], image.size[1]]
            else:
                box = [0, 0, 0, 0]
            blocks = [{"text": raw_text, "box": box, "confidence": float(confidence or 1.0)}]

        if not raw_text:
            raw_text = "\n".join(str(block["text"]).strip() for block in blocks if str(block.get("text", "")).strip())
        if not raw_text and not blocks:
            payload = _build_success_payload(
                backend="virtual_passthrough",
                used_gold=False,
                error="Missing OCR content. You must provide `ocr_text` or non-empty `ocr_blocks` in the tool arguments.",
                text_summary="No OCR content was provided in the tool arguments.",
                blocks=[],
                confidence=0.0,
                redacted=redact_output,
                metadata={"num_blocks": 0, "region_used": region_used, "structure_hint": "unknown"},
            )
            _state_set(agent_data, "ocr_result", payload)
            _state_set(agent_data, "ocr_blocks", [])
            return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload
        if not text_summary:
            text_summary = _summarize_ocr_text(raw_text)
        if not structure_hint:
            structure_hint = _detect_structure_from_ocr(blocks)[0]

        public_blocks = [
            {
                "text": _redact_text(block["text"]) if redact_output else block["text"],
                "box": block["box"],
                "confidence": float(block["confidence"]),
            }
            for block in blocks
        ]
        payload = _build_success_payload(
            backend="virtual_passthrough",
            used_gold=False,
            error=None,
            text_summary=text_summary,
            blocks=public_blocks,
            confidence=float(confidence) if confidence is not None else _mean_confidence(public_blocks),
            redacted=redact_output,
            metadata={"num_blocks": len(public_blocks), "region_used": region_used, "structure_hint": structure_hint, **({"raw_text_private": raw_text} if save_private_trace and raw_text else {})},
        )
        _state_set(agent_data, "ocr_result", payload)
        _state_set(agent_data, "ocr_blocks", blocks)
        return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload


class CropZoomTool(BaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _tool_schema(
            name="crop_zoom_tool",
            description="Crop and zoom a region from the current image.",
            properties={
                "bbox_2d": {"type": "array", "description": "Bounding box [x1, y1, x2, y2] in image coordinates."},
                "scale": {"type": "number", "description": "Optional expansion factor around the box."},
            },
            required=["bbox_2d"],
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        del instance_id
        agent_data = kwargs.get("agent_data")
        image_path = str(_parse_metadata(agent_data).get("image_path") or "")
        try:
            image = _get_image(agent_data, image_path=image_path or None)
            bbox = list(parameters.get("bbox_2d") or [])
            scale = float(parameters.get("scale") or 1.0)
            width, height = image.size
            left, top, right, bottom = _normalize_bbox(bbox, width, height)
            if scale > 1.0:
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0
                half_w = (right - left) * scale / 2.0
                half_h = (bottom - top) * scale / 2.0
                left, top, right, bottom = _normalize_bbox([center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h], width, height)
            cropped = image.crop((left, top, right, bottom))
            payload = _build_success_payload(backend="crop_zoom", used_gold=False, bbox_2d=[left, top, right, bottom], size=list(cropped.size), error=None)
            _state_set(agent_data, "crop_zoom_result", payload)
            return ToolResponse(text=_safe_json_dumps(payload), image=[cropped]), 0.0, payload
        except Exception as exc:  # noqa: BLE001
            payload = _build_success_payload(backend="crop_zoom", used_gold=False, error=str(exc))
            return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload


class GroundingTool(_VLMCapableTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _tool_schema(
            name="grounding_tool",
            description="Ground a query to relevant bounding boxes in the image.",
            properties={
                "query": {"type": "string", "description": "The visual evidence query to ground."},
                "ocr_blocks": {"type": "array", "description": "Optional OCR blocks to reuse."},
                "layout_blocks": {"type": "array", "description": "Optional layout blocks to reuse."},
                "evidence_regions": {"type": "array", "description": "Optional oracle evidence regions."},
                "allow_gold": {"type": "boolean", "description": "Allow oracle evidence regions in debug/oracle runs."},
            },
            required=[],
        )

    def _ocr_layout_fallback(self, *, query: str, image_size: tuple[int, int], ocr_blocks: list[dict[str, Any]], layout_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_tokens = _match_query_tokens(query)
        candidates: list[dict[str, Any]] = []
        for block in ocr_blocks:
            text = str(block.get("text", "")).lower()
            score = float(block.get("confidence", 0.3))
            if query_tokens & set(re.findall(r"[a-z0-9_@.-]+", text)):
                score = max(score, 0.8)
            elif any(token in query_tokens for token in {"text", "email", "chat", "poster", "flowchart", "table"}):
                score = max(score, 0.55)
            else:
                continue
            candidates.append({"label": "ocr_match", "box": _clip_box(block.get("box", [0, 0, *image_size]), image_size[0], image_size[1]), "confidence": min(0.95, score), "source": "ocr_layout_fallback"})
        for block in layout_blocks:
            block_type = str(block.get("type", "")).lower()
            structure_tokens = {block_type, str(block.get("text", "")).lower(), str(block.get("text_summary", "")).lower()}
            if not query_tokens & structure_tokens and structure_tokens == {"", ""}:
                continue
            candidates.append({"label": block_type or "layout_match", "box": _clip_box(block.get("box", [0, 0, *image_size]), image_size[0], image_size[1]), "confidence": float(block.get("confidence", 0.6)), "source": "ocr_layout_fallback"})
        unique = []
        seen = set()
        for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
            key = tuple(candidate["box"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique[:5]

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        del instance_id
        agent_data = kwargs.get("agent_data")
        metadata = _parse_metadata(agent_data)
        image_path = str(metadata.get("image_path") or "")
        query = str(parameters.get("query") or _extract_user_query(agent_data)).strip()
        allow_gold = _resolve_bool(parameters.get("allow_gold"), self.config.get("allow_gold", False))
        evidence_regions = list(parameters.get("evidence_regions") or metadata.get("evidence_regions") or [])
        ocr_blocks = list(parameters.get("ocr_blocks") or _state_get(agent_data, "ocr_blocks", []))
        layout_result = _state_get(agent_data, "layout_result", {})
        layout_blocks = list(parameters.get("layout_blocks") or layout_result.get("blocks") or [])
        ablation_mode = str(metadata.get("ablation_mode") or "")
        try:
            image = _get_image(agent_data, image_path=image_path or None)
            resolved_image_path = _ensure_image_path(image, image_path or None)
        except Exception as exc:  # noqa: BLE001
            payload = _build_success_payload(backend="heuristic_fallback", used_gold=False, query=query, boxes=[], error=str(exc), metadata={"num_boxes": 0, "used_gold": False})
            return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload

        errors: list[str] = []
        boxes: list[dict[str, Any]] = []
        backend = "heuristic_fallback"
        used_gold = False
        fallback_order = list(self.config.get("fallback_order", ["self_vlm_grounding", "ocr_layout_fallback", "heuristic_fallback"]))
        if allow_gold and ablation_mode == "oracle_tools" and evidence_regions:
            fallback_order = ["gold_evidence"] + [item for item in fallback_order if item != "gold_evidence"]

        for candidate in fallback_order:
            if candidate == "gold_evidence":
                if evidence_regions:
                    backend = "gold_evidence"
                    used_gold = True
                    boxes = []
                    for region in evidence_regions:
                        if isinstance(region, dict):
                            raw_box = region.get("bbox_2d") or region.get("box")
                            label = str(region.get("label") or "gold_evidence")
                            conf = float(region.get("score", 0.99))
                        else:
                            raw_box = region
                            label = "gold_evidence"
                            conf = 0.99
                        boxes.append({"label": label, "box": _clip_box(raw_box, image.size[0], image.size[1]), "confidence": conf, "source": "gold_evidence"})
                    break
            elif candidate == "self_vlm_grounding":
                prompt = (
                    f"Ground the query to relevant image regions. Query: {query}. "
                    "Return JSON only with keys: boxes and rationale_short. "
                    "Each box must include label, box [x1,y1,x2,y2], confidence, evidence_type. "
                    "Coordinates may be pixel or normalized."
                )
                system_prompt = "You are a visual grounding tool. Output valid JSON only."
                vlm_result = self._call_self_vlm(image_path=resolved_image_path, prompt=prompt, system_prompt=system_prompt, max_tokens=512)
                if vlm_result["success"] and isinstance(vlm_result["json"], dict):
                    parsed_boxes = list(vlm_result["json"].get("boxes") or [])
                    boxes = []
                    for index, box in enumerate(parsed_boxes):
                        if box.get("box") is None:
                            continue
                        try:
                            normalized_box = _convert_box_from_vlm(box.get("box", [0, 0, image.size[0], image.size[1]]), image.size[0], image.size[1])
                        except Exception as exc:  # noqa: BLE001
                            errors.append(f"self_vlm_grounding box {index} invalid: {exc}")
                            continue
                        boxes.append(
                            {
                                "label": str(box.get("label") or "region"),
                                "box": normalized_box,
                                "confidence": float(box.get("confidence", 0.5)),
                                "source": "self_vlm_grounding",
                                "evidence_type": str(box.get("evidence_type") or "region"),
                            }
                        )
                    if boxes:
                        backend = "self_vlm_grounding"
                        break
                errors.append(vlm_result.get("error") or "self_vlm_grounding failed or returned no valid boxes.")
            elif candidate == "ocr_layout_fallback":
                boxes = self._ocr_layout_fallback(query=query, image_size=image.size, ocr_blocks=ocr_blocks, layout_blocks=layout_blocks)
                backend = "ocr_layout_fallback"
                if boxes:
                    break
                errors.append("ocr_layout_fallback found no matching regions.")
            elif candidate == "heuristic_fallback":
                boxes = [{"label": "full_image", "box": [0, 0, image.size[0], image.size[1]], "confidence": 0.2, "source": "heuristic_fallback"}]
                backend = "heuristic_fallback"
                break

        payload = _build_success_payload(backend=backend, used_gold=used_gold, query=query, boxes=boxes, error="; ".join(errors) if backend == "heuristic_fallback" and errors else None, metadata={"num_boxes": len(boxes), "used_gold": used_gold})
        _state_set(agent_data, "grounding_result", payload)
        return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload


class LayoutParseTool(_VLMCapableTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _tool_schema(
            name="layout_parse_tool",
            description="Infer document layout structure from OCR blocks and image regions.",
            properties={
                "ocr_blocks": {"type": "array", "description": "Optional OCR blocks to reuse."},
                "visual_text_gold": {"type": "string", "description": "Optional oracle text."},
                "use_vlm": {"type": "boolean", "description": "Whether to attempt VLM layout parsing."},
            },
            required=[],
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        del instance_id
        agent_data = kwargs.get("agent_data")
        image_path = str(_parse_metadata(agent_data).get("image_path") or "")
        ocr_blocks = list(parameters.get("ocr_blocks") or _state_get(agent_data, "ocr_blocks", []))
        use_vlm = _resolve_bool(parameters.get("use_vlm"), True)
        try:
            image = _get_image(agent_data, image_path=image_path or None)
            resolved_image_path = _ensure_image_path(image, image_path or None)
        except Exception as exc:  # noqa: BLE001
            payload = _build_success_payload(backend="heuristic_fallback", used_gold=False, structure_type="unknown", blocks=[], relations=[], reading_order=[], confidence=0.0, error=str(exc))
            return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload

        errors: list[str] = []
        fallback_order = list(self.config.get("fallback_order", ["self_vlm_layout", "ocr_block_layout", "cv_connected_components", "heuristic_fallback"]))
        for candidate in fallback_order:
            if candidate == "self_vlm_layout" and use_vlm:
                prompt = (
                    "Parse the visual structure of the image. Return JSON only with keys: "
                    "structure_type, blocks, relations, reading_order. "
                    "Each block must include id, type, text_summary, box, confidence."
                )
                system_prompt = "You are a layout parser tool. Output valid JSON only."
                vlm_result = self._call_self_vlm(image_path=resolved_image_path, prompt=prompt, system_prompt=system_prompt, max_tokens=700)
                if vlm_result["success"] and isinstance(vlm_result["json"], dict):
                    parsed = vlm_result["json"]
                    blocks = []
                    for index, block in enumerate(list(parsed.get("blocks") or [])):
                        try:
                            normalized_box = _convert_box_from_vlm(block.get("box", [0, 0, image.size[0], image.size[1]]), image.size[0], image.size[1])
                        except Exception as exc:  # noqa: BLE001
                            errors.append(f"self_vlm_layout block {index} invalid: {exc}")
                            continue
                        blocks.append(
                            {
                                "id": str(block.get("id") or f"block_{index}"),
                                "type": str(block.get("type") or "region"),
                                "text_summary": str(block.get("text_summary") or ""),
                                "box": normalized_box,
                                "confidence": float(block.get("confidence", 0.5)),
                            }
                        )
                    payload = _build_success_payload(
                        backend="self_vlm_layout",
                        used_gold=False,
                        structure_type=str(parsed.get("structure_type") or "unknown"),
                        blocks=blocks,
                        relations=list(parsed.get("relations") or []),
                        reading_order=list(parsed.get("reading_order") or [block["id"] for block in blocks]),
                        confidence=_mean_confidence(blocks) if blocks else 0.5,
                        error=None,
                    )
                    _state_set(agent_data, "layout_result", payload)
                    return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload
                errors.append(vlm_result.get("error") or "self_vlm_layout failed.")
            elif candidate == "ocr_block_layout" and ocr_blocks:
                structure_type, confidence = _detect_structure_from_ocr(ocr_blocks)
                blocks = []
                for index, block in enumerate(sorted(ocr_blocks, key=lambda item: (item["box"][1], item["box"][0]))):
                    text = str(block.get("text", ""))
                    block_type = "text"
                    if structure_type == "chat" and re.match(r"^\s*(alice|bob|user|assistant|agent|system)\s*:", text.lower()):
                        block_type = "speaker_turn"
                    elif structure_type == "table":
                        block_type = "table_cell"
                    blocks.append({"id": f"ocr_{index}", "type": block_type, "text_summary": text, "box": _clip_box(block.get("box", [0, 0, image.size[0], image.size[1]]), image.size[0], image.size[1]), "confidence": float(block.get("confidence", 0.5))})
                relations, reading_order = _build_relations(blocks, structure_type)
                payload = _build_success_payload(backend="ocr_block_layout", used_gold=False, structure_type=structure_type, blocks=blocks, relations=relations, reading_order=reading_order, confidence=confidence, error=None)
                _state_set(agent_data, "layout_result", payload)
                return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload
            elif candidate == "cv_connected_components":
                regions = _connected_components_regions(image)
                if regions:
                    blocks = [{"id": f"region_{index}", "type": "region", "text_summary": "", "box": _clip_box(region, image.size[0], image.size[1]), "confidence": 0.45} for index, region in enumerate(regions)]
                    relations, reading_order = _build_relations(blocks, "unknown")
                    payload = _build_success_payload(backend="cv_connected_components", used_gold=False, structure_type="unknown", blocks=blocks, relations=relations, reading_order=reading_order, confidence=0.45, error=None)
                    _state_set(agent_data, "layout_result", payload)
                    return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload
                errors.append("cv_connected_components found no salient regions.")
            elif candidate == "heuristic_fallback":
                payload = _build_success_payload(
                    backend="heuristic_fallback",
                    used_gold=False,
                    structure_type="unknown",
                    blocks=[{"id": "full_image", "type": "region", "text_summary": "", "box": [0, 0, image.size[0], image.size[1]], "confidence": 0.2}],
                    relations=[],
                    reading_order=["full_image"],
                    confidence=0.2,
                    error="; ".join(errors) if errors else None,
                )
                _state_set(agent_data, "layout_result", payload)
                return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload

        payload = _build_success_payload(backend="heuristic_fallback", used_gold=False, structure_type="unknown", blocks=[], relations=[], reading_order=[], confidence=0.0, error="Unknown layout parsing failure.")
        return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload


class PolicyCheckTool(BaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _tool_schema(
            name="policy_check_tool",
            description=(
                "Load relevant safety policy descriptions. "
                "This tool does not decide whether the request is safe. "
                "It returns policy text that the model should use as constraints."
            ),
            properties={
                "policy_ids": {
                    "type": "array",
                    "description": "Optional policy ids to load, such as violent_harm, sexual_minor, privacy_pii, hate_or_harassment.",
                },
                "policy_texts": {
                    "type": "object",
                    "description": "Optional mapping from policy id to policy description text to load directly.",
                },
                "policy_text_list": {
                    "type": "array",
                    "description": "Optional list of policy description strings to load directly.",
                },
                "include_default_library": {
                    "type": "boolean",
                    "description": "Whether to include the built-in default policy library when loading policies.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional query or category hint used only to select which policy descriptions to return.",
                },
            },
            required=[],
        )

    def _select_policy_ids_from_query(self, query: str) -> list[str]:
        lowered = query.lower()
        selected: list[str] = []
        if any(token in lowered for token in ["bomb", "weapon", "attack", "kill", "harm", "poison", "assault"]):
            selected.append("violent_harm")
        if any(token in lowered for token in ["child sexual", "minor nude", "young-looking sexual", "minor"]):
            selected.append("sexual_minor")
        if any(token in lowered for token in ["address", "phone", "email", "passport", "id card", "pii", "private data"]):
            selected.append("privacy_pii")
        if any(token in lowered for token in ["slur", "hate", "harass", "ethnic", "racist", "protected group"]):
            selected.append("hate_or_harassment")
        return selected

    def _normalize_policy_text_map(self, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, str] = {}
        for key, item in value.items():
            text = str(item or "").strip()
            if text:
                normalized[str(key)] = text
        return normalized

    def _normalize_policy_text_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                result.append(text)
        return result

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        del instance_id
        agent_data = kwargs.get("agent_data")
        metadata = _parse_metadata(agent_data)
        query = str(parameters.get("query") or _extract_user_query(agent_data)).strip()
        include_default_library = _resolve_bool(parameters.get("include_default_library"), True)

        metadata_policy_map = self._normalize_policy_text_map(metadata.get("policy_texts") or metadata.get("policy_library"))
        parameter_policy_map = self._normalize_policy_text_map(parameters.get("policy_texts"))
        policy_library = {}
        if include_default_library:
            policy_library.update(DEFAULT_POLICY_LIBRARY)
        policy_library.update(metadata_policy_map)
        policy_library.update(parameter_policy_map)

        policy_ids = [str(item) for item in list(parameters.get("policy_ids") or []) if str(item).strip()]
        if not policy_ids:
            metadata_policy_ids = metadata.get("policy_ids") or metadata.get("expected_policy_tags") or []
            policy_ids = [str(item) for item in list(metadata_policy_ids) if str(item).strip()]
        if not policy_ids and query:
            policy_ids = self._select_policy_ids_from_query(query)

        inline_policy_texts = self._normalize_policy_text_list(metadata.get("policy_text_list")) + self._normalize_policy_text_list(parameters.get("policy_text_list"))
        loaded_policies = [{"policy_id": policy_id, "description": policy_library[policy_id]} for policy_id in policy_ids if policy_id in policy_library]
        loaded_policies.extend(
            {"policy_id": f"inline_{index}", "description": text}
            for index, text in enumerate(inline_policy_texts)
        )

        error = None
        if not loaded_policies:
            error = "No policy descriptions were available to load from policy_ids, policy_texts, policy_text_list, or the default library."
        payload = _build_success_payload(
            backend="policy_library",
            used_gold=False,
            error=error,
            query=query,
            policy_ids=policy_ids,
            policies=loaded_policies,
            policy_text=("\n\n".join(f"[{item['policy_id']}] {item['description']}" for item in loaded_policies) if loaded_policies else ""),
            metadata={
                "num_policies": len(loaded_policies),
                "library_keys": sorted(policy_library.keys()),
            },
        )
        _state_set(agent_data, "policy_result", payload)
        return ToolResponse(text=_safe_json_dumps(payload)), 0.0, payload
