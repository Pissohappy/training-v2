from __future__ import annotations

import copy
import json
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

from recipe.safe_vtool.common import (
    ensure_safety_prompt,
    get_tool_names_for_ablation,
    normalize_ablation_mode,
    parse_metadata_blob,
)


@register("safe_vtool_agent")
class SafeVToolAgentLoop(ToolAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_trace: dict[str, Any] = {}
        self._all_tools = dict(self.tools)
        self._all_tool_schemas = list(self.tool_schemas)

    def _configure_tools_for_ablation(self, ablation_mode: str) -> None:
        allowed = set(get_tool_names_for_ablation(ablation_mode))
        self.tools = {name: tool for name, tool in self._all_tools.items() if name in allowed}
        self.tool_schemas = [
            schema for schema in self._all_tool_schemas if schema.get("function", {}).get("name") in allowed
        ]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_prompt = copy.deepcopy(list(kwargs["raw_prompt"]))
        tools_kwargs = kwargs.get("tools_kwargs", {}) or {}
        metadata = parse_metadata_blob(tools_kwargs.get("metadata"))
        ablation_mode = normalize_ablation_mode(metadata.get("ablation_mode"))
        prompt_variant = str(metadata.get("prompt_variant") or "safety")
        sample_id = metadata.get("sample_id") or kwargs.get("index")
        self._configure_tools_for_ablation(ablation_mode)

        kwargs["raw_prompt"] = ensure_safety_prompt(raw_prompt, ablation_mode=ablation_mode, prompt_variant=prompt_variant)
        self._active_trace = {
            "sample_id": sample_id,
            "ablation_mode": ablation_mode,
            "prompt_variant": prompt_variant,
            "tool_steps": [],
            "final_response": "",
            "judge_result": None,
        }
        output = await super().run(sampling_params, **kwargs)

        final_response = await self._decode_final_response(output)
        output.extra_fields["safe_vtool_final_response_text"] = final_response
        output.extra_fields["final_response"] = final_response
        output.extra_fields["tool_trace"] = dict(self._active_trace, final_response=final_response)
        output.extra_fields["sample_id"] = sample_id
        output.extra_fields["ablation_mode"] = ablation_mode
        output.extra_fields["prompt_variant"] = prompt_variant
        return output

    async def _decode_final_response(self, output: AgentLoopOutput) -> str:
        response_ids = list(output.response_ids)
        response_mask = list(output.response_mask)
        final_ids = [token_id for token_id, mask in zip(response_ids, response_mask, strict=False) if mask]
        if not final_ids:
            final_ids = response_ids
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(final_ids, skip_special_tokens=True).strip(),
        )

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        next_state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        assistant_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True).strip(),
        )
        turn_metadata = agent_data.extra_fields.setdefault("turn_metadata", [])
        turn_metadata.append(
            {
                "assistant_turn": agent_data.assistant_turns,
                "state_after_generation": next_state.value,
                "tool_calls": [tool_call.name for tool_call in agent_data.tool_calls],
                "response_text": assistant_text,
            }
        )
        return next_state

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        parsed_arguments: dict[str, Any]
        try:
            parsed_arguments = json.loads(tool_call.arguments)
            if not isinstance(parsed_arguments, dict):
                parsed_arguments = {}
        except json.JSONDecodeError:
            parsed_arguments = {}

        tool_response, tool_reward, metadata = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        success = not str(tool_response.text or "").startswith("Error when executing tool:")
        self._active_trace["tool_steps"].append(
            {
                "tool_name": tool_call.name,
                "arguments": parsed_arguments,
                "observation": {
                    "text": tool_response.text,
                    "image_count": len(tool_response.image or []),
                    "video_count": len(tool_response.video or []),
                },
                "success": success,
                "metadata": metadata or {},
            }
        )
        agent_data.extra_fields["tool_trace"] = dict(self._active_trace)
        return tool_response, tool_reward, metadata
