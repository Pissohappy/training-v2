from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from PIL import Image


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url must be non-empty")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _extract_json_candidate(raw_text: str) -> dict[str, Any] | None:
    raw_text = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw_text = fenced.group(1).strip()
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    loose = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if not loose:
        return None
    try:
        parsed = json.loads(loose.group(1))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _image_to_data_url(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class VLMToolClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 60.0,
        retry: int = 2,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.retry = max(0, int(retry))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def call(
        self,
        *,
        image_path: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                ],
            }
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }

        last_error: str | None = None
        for attempt in range(self.retry + 1):
            try:
                raw = self._post_json("/chat/completions", payload)
                message = ((raw.get("choices") or [{}])[0].get("message") or {})
                content = message.get("content")
                if isinstance(content, list):
                    raw_response = "".join(
                        str(item.get("text", "")) if isinstance(item, dict) else str(item)
                        for item in content
                    ).strip()
                else:
                    raw_response = str(content or "").strip()
                parsed_json = _extract_json_candidate(raw_response)
                return {
                    "success": parsed_json is not None,
                    "backend": "self_vlm",
                    "raw_response": raw_response,
                    "json": parsed_json,
                    "error": None if parsed_json is not None else "Failed to parse JSON from VLM response.",
                }
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {detail}"
            except error.URLError as exc:
                last_error = f"URL error: {exc.reason}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            if attempt < self.retry:
                time.sleep(1.0)

        return {
            "success": False,
            "backend": "self_vlm",
            "raw_response": "",
            "json": None,
            "error": last_error or "Unknown VLM tool client error.",
        }


def build_vlm_tool_client(config: dict[str, Any]) -> VLMToolClient:
    return VLMToolClient(
        base_url=str(config.get("base_url") or "http://127.0.0.1:8000/v1"),
        model=str(config.get("model") or ""),
        api_key=str(config.get("api_key") or "EMPTY"),
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(config.get("max_tokens", 512)),
        timeout=float(config.get("timeout", 60)),
        retry=int(config.get("retry", 2)),
    )
