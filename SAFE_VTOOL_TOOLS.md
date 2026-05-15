# SafeVTool Tools Reference

This document describes the tools currently defined for `safe_vtool`: what each tool does, which argument slots it accepts, how the model is expected to call it, what it returns, and how tool state flows across turns.

Relevant source files:

- `verl/tools/safe_vtool_tools.py`
- `recipe/safe_vtool/safety_tools_config.yaml`
- `recipe/safe_vtool/common.py`
- `recipe/safe_vtool/vlm_tool_client.py`
- `recipe/safe_vtool/safe_vtool_agent.py`
- `eval/safety_eval.py`

## Overview

The current tool set is:

- `safety_ocr_tool`
- `crop_zoom_tool`
- `grounding_tool`
- `layout_parse_tool`
- `policy_check_tool`

At runtime, the model sees OpenAI-style function schemas and may emit `tool_calls` in an assistant turn. The local runner then:

1. Parses `message.tool_calls`
2. Executes the corresponding Python tool implementation
3. Appends the tool observation back into the message chain as a `tool` message
4. Sends the updated message list back to the model for the next assistant turn

This creates the intended `think -> tools -> think -> tools -> final answer` loop.

## Invocation Model

Tools are exposed to the model as OpenAI function tools. In `eval.safety_eval`, the request payload includes:

- `messages`
- `tools`

If the model decides to call a tool, it should return `tool_calls` in the assistant message. Each call contains:

- `function.name`
- `function.arguments`

The local code then maps `name` to the matching class in `verl/tools/safe_vtool_tools.py`.

## Tool Availability By Ablation

Configured in `recipe/safe_vtool/common.py` and `recipe/safe_vtool/safety_tools_config.yaml`.

- `no_tools`
  No tools available.
- `self_vlm_tools`
  `safety_ocr_tool`, `crop_zoom_tool`, `grounding_tool`, `layout_parse_tool`
- `external_tools`
  Same tools as above. OCR remains a virtual passthrough tool; grounding/layout fallback order is changed to prefer non-VLM backends.
- `full_safevtool`
  `safety_ocr_tool`, `crop_zoom_tool`, `grounding_tool`, `layout_parse_tool`, `policy_check_tool`
- `oracle_tools`
  `safety_ocr_tool`, `grounding_tool`, `policy_check_tool`, with oracle-style tool availability but without a separate policy classification backend.

## Shared Runtime State

Tools read and write temporary state through `agent_data.extra_fields["safe_vtool_tool_state"]`.

Important state keys:

- `ocr_result`
- `ocr_blocks`
- `crop_zoom_result`
- `grounding_result`
- `layout_result`
- `policy_result`

This means later tools may omit some arguments and reuse prior tool output.

## Tool 1: `safety_ocr_tool`

### Purpose

Virtual OCR passthrough. The model is expected to read visible text itself and place the OCR result directly into the tool arguments. The tool does not run a real OCR backend.

### Slots

- `image_path: string`
  Optional explicit image path. If omitted, uses the current image from agent context.
- `region: array`
  Optional OCR region `[x1, y1, x2, y2]`.
- `ocr_text: string`
  OCR text content filled directly by the caller.
- `ocr_blocks: array`
  Optional OCR blocks, each typically containing `text`, `box`, and `confidence`.
- `text_summary: string`
  Optional OCR summary.
- `structure_hint: string`
  Optional structure hint such as `document`, `chat`, `table`, or `poster`.
- `confidence: number`
  Optional overall OCR confidence.
- `redact_output: boolean`
  Whether public tool output should hide raw OCR text.
- `save_private_trace: boolean`
  Whether to preserve raw OCR text in metadata.

### How It Works

1. Read `ocr_text` and/or `ocr_blocks` from tool arguments.
2. Optionally resolve image context only to normalize or clip boxes.
3. If `ocr_blocks` is absent but `ocr_text` is present, synthesize a single OCR block.
4. If `text_summary` or `structure_hint` is absent, derive a lightweight fallback summary from the provided text/blocks.
5. Save:
   - `ocr_result`
   - `ocr_blocks`

### Runtime Behavior

- `backend` is always reported as `virtual_passthrough`.
- No OCR backend is invoked.
- The tool still accepts legacy `visual_text_gold` as a compatibility fallback if older data pipelines populate it.
- Public `blocks[*].text` may be redacted to `[redacted]`.
- Raw text is only preserved when `save_private_trace=true`.

### Return Payload

Typical fields:

- `success`
- `backend`
- `used_gold`
- `error`
- `text_summary`
- `blocks`
- `confidence`
- `redacted`
- `metadata`

`metadata` may include:

- `num_blocks`
- `region_used`
- `structure_hint`
- `raw_text_private`

### Notes

- OCR blocks are reused later by grounding and layout parsing.
- The main safe-vtool prompt explicitly instructs the model to treat this as a virtual tool call and to place OCR results directly into `ocr_text` or `ocr_blocks`.

### Example Call

```json
{
  "name": "safety_ocr_tool",
  "arguments": {
    "region": [120, 80, 760, 420],
    "ocr_text": "Step 1: Open Settings. Step 2: Disable protections.",
    "ocr_blocks": [
      {
        "text": "Step 1: Open Settings.",
        "box": [120, 80, 760, 180],
        "confidence": 0.96
      },
      {
        "text": "Step 2: Disable protections.",
        "box": [120, 190, 760, 300],
        "confidence": 0.95
      }
    ],
    "text_summary": "The image contains step-by-step procedural text.",
    "redact_output": false,
    "save_private_trace": true
  }
}
```

## Tool 2: `crop_zoom_tool`

### Purpose

Crop and zoom a selected region from the current image.

### Slots

- `bbox_2d: array`
  Required. Bounding box `[x1, y1, x2, y2]`.
- `scale: number`
  Optional expansion factor around the box.
- `image_path: string`
  Optional explicit image path.

### How It Works

1. Resolve the current image.
2. Normalize and clip `bbox_2d`.
3. If `scale > 1.0`, expand around the box center.
4. Crop the region.
5. Return the cropped image as a tool image observation.
6. Save `crop_zoom_result`.

### Return Payload

- `success`
- `backend: "crop_zoom"`
- `used_gold: false`
- `bbox_2d`
- `size`
- `error`

### Example Call

```json
{
  "name": "crop_zoom_tool",
  "arguments": {
    "bbox_2d": [210, 140, 510, 330],
    "scale": 1.3
  }
}
```

## Tool 3: `grounding_tool`

### Purpose

Map a query or evidence request to one or more image regions.

### Slots

- `image_path: string`
  Optional explicit image path.
- `query: string`
  What visual evidence should be grounded.
- `ocr_blocks: array`
  Optional OCR blocks. If omitted, reuse `ocr_blocks` from state.
- `layout_blocks: array`
  Optional layout blocks. If omitted, reuse layout state.
- `evidence_regions: array`
  Optional oracle evidence regions.
- `allow_gold: boolean`
  Whether oracle evidence is allowed.

### Default Backend Order

1. `self_vlm_grounding`
2. `ocr_layout_fallback`
3. `heuristic_fallback`

In `oracle_tools`, `gold_evidence` may be inserted first.

### How It Works

1. Resolve image.
2. Read query from arguments, or fall back to the last user query.
3. Read OCR blocks from arguments or tool state.
4. Read layout blocks from arguments or `layout_result`.
5. Try grounding backends in order.
6. Save `grounding_result`.

### Backend Details

#### `self_vlm_grounding`

Calls the VLM with a prompt asking for JSON keys:

- `boxes`
- `rationale_short`

Each box should include:

- `label`
- `box`
- `confidence`
- `evidence_type`

#### `ocr_layout_fallback`

Heuristic grounding using:

- token overlap between query and OCR text
- layout block types such as `table`, `chat`, `poster`, `flowchart`

Returns up to 5 unique boxes.

#### `heuristic_fallback`

Returns the full image as a low-confidence region.

#### `gold_evidence`

Uses provided oracle evidence boxes in debug mode.

### Return Payload

- `success`
- `backend`
- `used_gold`
- `query`
- `boxes`
- `error`
- `metadata.num_boxes`

### Example Call

```json
{
  "name": "grounding_tool",
  "arguments": {
    "query": "Where is the dangerous instruction text in the image?"
  }
}
```

## Tool 4: `layout_parse_tool`

### Purpose

Infer document structure from the image and/or OCR results.

### Slots

- `image_path: string`
  Optional explicit image path.
- `ocr_blocks: array`
  Optional OCR blocks. If omitted, reuse state.
- `visual_text_gold: string`
  Declared in schema, but not actively used in the current implementation.
- `use_vlm: boolean`
  Whether to attempt VLM layout parsing first.

### Default Backend Order

1. `self_vlm_layout`
2. `ocr_block_layout`
3. `cv_connected_components`
4. `heuristic_fallback`

### How It Works

1. Resolve image.
2. Read OCR blocks from arguments or state.
3. If `use_vlm=true`, try VLM layout parsing first.
4. Otherwise or on failure, infer structure from OCR or connected components.
5. Save `layout_result`.

### Backend Details

#### `self_vlm_layout`

Calls the VLM asking for JSON keys:

- `structure_type`
- `blocks`
- `relations`
- `reading_order`

Each block should include:

- `id`
- `type`
- `text_summary`
- `box`
- `confidence`

#### `ocr_block_layout`

Derives structure from OCR blocks.

Possible `structure_type` values include:

- `email`
- `chat`
- `poster`
- `document`
- `table`
- `flowchart`
- `unknown`

It also builds:

- `relations`
- `reading_order`

#### `cv_connected_components`

Uses connected-component style region extraction to find salient blocks.

#### `heuristic_fallback`

Returns a single full-image block.

### Return Payload

- `success`
- `backend`
- `used_gold`
- `structure_type`
- `blocks`
- `relations`
- `reading_order`
- `confidence`
- `error`

### Example Call

```json
{
  "name": "layout_parse_tool",
  "arguments": {
    "use_vlm": true
  }
}
```

## Tool 5: `policy_check_tool`

### Purpose

Load relevant safety policy descriptions or constraints. This tool does not decide whether the request is safe. It returns policy text that the model should use when producing the final decision and response.

### Slots

- `policy_ids: array`
  Optional policy ids to load, such as `violent_harm`, `sexual_minor`, `privacy_pii`, or `hate_or_harassment`.
- `policy_texts: object`
  Optional mapping from policy id to policy description text.
- `policy_text_list: array`
  Optional list of policy description strings to inject directly.
- `include_default_library: boolean`
  Whether to include the built-in default policy library.
- `query: string`
  Optional query or category hint used only to select which policy descriptions to return.

### Policy Sources

The tool can load policy descriptions from:

1. direct tool arguments
2. `tools_kwargs.metadata.policy_texts` or `tools_kwargs.metadata.policy_library`
3. `tools_kwargs.metadata.policy_text_list`
4. a built-in default policy library

The built-in default library currently includes:

- `violent_harm`
- `sexual_minor`
- `privacy_pii`
- `hate_or_harassment`

### How It Works

1. Build a policy library from built-in defaults plus any policy text supplied in metadata or direct arguments.
2. Read explicit `policy_ids` if provided.
3. Otherwise, optionally fall back to metadata policy ids or lightweight query-based selection.
4. Return the matched policy descriptions and save `policy_result`.

### Return Payload

- `success`
- `backend`
- `used_gold`
- `query`
- `policy_ids`
- `policies`
- `policy_text`
- `error`
- `metadata.num_policies`

### Example Call

```json
{
  "name": "policy_check_tool",
  "arguments": {
    "policy_ids": ["violent_harm", "privacy_pii"],
    "include_default_library": true
  }
}
```

## Current Prompt-Level Guidance

The system prompt currently tells the model:

- inspect the image
- decide whether to answer, refuse, clarify, or safe-redirect
- ground claims in the visible image or tool observations
- use tools only when they add evidence

This means the model is allowed to skip tools when the text request alone is sufficient to decide.

## Important Implementation Notes

- `generated_text` in eval output is the final assistant text, not the full turn history.
- Full interleaved message history is now saved under `conversation_trace` in `eval.safety_eval`.
- `safety_ocr_tool` and other VLM-backed tools depend on `vlm_tool_client.model` being configured.
- In `external_tools` mode, the filtered config rewrites backend preference so OCR/layout/grounding rely less on self-VLM.

## Practical Debugging Checklist

When a tool is not used, check:

1. Was the tool schema present in the request?
2. Did the assistant message return `tool_calls`?
3. Did `conversation_trace[*].raw_tool_calls` contain anything?
4. Did `tool_results` get populated?
5. Did the tool backend succeed, or did it fall through to an empty/heuristic fallback?

When OCR quality looks wrong, check:

1. Was `self_vlm_ocr` actually configured with a model?
2. Was the image cropped by `region` before OCR?
3. Did the tool fall back to `easyocr` or `pytesseract`?
4. Was `redact_output` hiding the useful text from the public trace?
5. Was `save_private_trace` enabled if raw OCR text is needed for debugging?
