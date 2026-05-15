from __future__ import annotations

import inspect
from typing import Any

from recipe.vtool.vtool import _normalize_object_field
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from judge.safety_reward import compute_score as safety_compute_score


class SafeVToolRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or safety_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        tool_extra_fields = _normalize_object_field(data_item.non_tensor_batch.get("tool_extra_fields", {}))
        if not isinstance(tool_extra_fields, dict):
            tool_extra_fields = {}

        response_str = tool_extra_fields.get("safe_vtool_final_response_text")
        if response_str is None:
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            if "response_mask" in data_item.batch.keys():
                valid_response_mask = data_item.batch["response_mask"][:valid_response_length].bool()
                final_response_ids = valid_response_ids[valid_response_mask]
                if final_response_ids.numel() == 0:
                    final_response_ids = valid_response_ids
            else:
                final_response_ids = valid_response_ids
            response_str = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(final_response_ids.tolist(), skip_special_tokens=True),
            )

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["tool_extra_fields"] = tool_extra_fields

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info: dict[str, Any] = {}
        if isinstance(result, dict):
            reward_score = float(result["score"])
            reward_extra_info.update(result)
        else:
            reward_score = float(result)
            reward_extra_info["score"] = reward_score

        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}
