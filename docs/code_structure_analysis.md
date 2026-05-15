# `training-v2` 代码结构审计

本文只分析当前仓库已经存在的结构，不引入新功能代码。目标是回答 SafeVTool-R1 若要最小侵入式接入，应该依附哪些现有路径，哪些地方已经具备扩展点，哪些地方和现有 `recipe/vtool` 明显不同。

## 结论摘要

- `verl.experimental.agent_loop` 已经是一套通用多轮 tool-calling 框架，包含：
  - tool 定义与注册；
  - tool schema；
  - 模型输出里的 tool_call 解析；
  - tool observation 回灌给模型；
  - `agent_name` 驱动的 agent loop 选择；
  - reward loop 对接。
- `recipe/vtool` 没有复用这套通用 tool registry / `ToolAgentLoop` 主流程。它实现的是一条专门的 one-shot “生成 Python 代码 -> 本地执行 refocus -> 再生成 final answer” 路径。
- 因此，SafeVTool-R1 如果要“最小侵入式”，不应在 `recipe/vtool/vtool.py` 上继续堆复杂逻辑，而应尽量建立在 `verl.experimental.agent_loop.tool_agent_loop.ToolAgentLoop` 之上，同时保留 `recipe/vtool` 当前的数据和 reward 接口风格作为参考。

## 1. Tool 是在哪里定义和注册的？

### 1.1 Tool 抽象基类

核心入口在：

- `verl/tools/base_tool.py`

`BaseTool` 约定了一个 tool 至少需要这些方法：

- `get_openai_tool_schema()`
- `create()`
- `execute()`
- `calc_reward()`
- `release()`

其中真正关键的是：

- `execute()` 返回 `(ToolResponse, float, dict)`
- `ToolResponse` 定义在 `verl/tools/schemas.py`

也就是说，tool 不是简单函数，而是一个类实例，支持：

- 每条 trajectory 创建实例；
- 执行；
- 可选 step reward；
- 释放资源。

### 1.2 Tool 的“注册”方式

当前仓库的 tool 不是通过全局 Python decorator 注册，而是通过配置文件按类名加载。

入口在：

- `verl/tools/utils/tool_registry.py`

核心流程：

1. 读取 tool config YAML/JSON。
2. 对每个 `tools[]` 项读取：
   - `class_name`
   - `config.type`
3. `get_tool_class()` 用 `importlib` 按全限定类名动态导入。
4. `initialize_tools_from_config()` 实例化 tool。

支持两类 tool：

- `native`
- `mcp`

对于 `native`：

- 直接实例化 Python 类；
- `class_name` 指向类似 `verl.tools.gsm8k_tool.Gsm8kTool` 的类。

对于 `mcp`：

- 通过 MCP client 获取 schema，再实例化 wrapper。

因此，“注册”本质上是：

- 先写一个继承 `BaseTool` 的类；
- 再把它写进 tool config 文件；
- 再由 `ToolAgentLoop` 在启动时加载该 config。

### 1.3 现有参考 Tool

现有原生 tool 例子可参考：

- `verl/tools/gsm8k_tool.py`
- `verl/tools/geo3k_tool.py`
- `verl/tools/image_zoom_in_tool.py`
- `verl/tools/search_tool.py`
- `verl/tools/sandbox_fusion_tools.py`

测试里的最小示例：

- `tests/experimental/agent_loop/test_basic_agent_loop.py`
- `tests/experimental/agent_loop/test_multi_modal.py`

## 2. Tool schema 如何写？

### 2.1 Schema 数据结构

schema 定义在：

- `verl/tools/schemas.py`

关键类型：

- `OpenAIFunctionToolSchema`
- `OpenAIFunctionSchema`
- `OpenAIFunctionParametersSchema`
- `OpenAIFunctionPropertySchema`

整体遵循 OpenAI function/tool calling 风格。

### 2.2 Tool 类里如何提供 schema

通常有两种写法。

#### 写法 A：tool 自己实现 `get_openai_tool_schema()`

例如测试文件中常用：

- 用 `transformers.utils.get_json_schema` 从 Python 函数签名自动导出；
- 再包装成 `OpenAIFunctionToolSchema`。

这一模式可参考：

- `tests/experimental/agent_loop/test_basic_agent_loop.py`
- `tests/experimental/agent_loop/test_multi_modal.py`

#### 写法 B：在 tool config 里直接写 `tool_schema`

`tool_registry.initialize_tools_from_config()` 会优先检查配置项里是否显式提供 `tool_schema`：

- 如果有，就用配置中的 schema；
- 如果没有，就调用 tool 类自己的 `get_openai_tool_schema()`。

所以 schema 来源可以是：

- 类内生成；
- 配置文件显式提供。

### 2.3 Tool schema 的字段要求

至少要满足：

- `type: "function"`
- `function.name`
- `function.description`
- `function.parameters.type = "object"`
- `function.parameters.properties`
- `function.parameters.required`

参数最终会被模型当成 tool-calling 的 JSON schema 使用。

## 3. `ToolAgentLoop` 的入口在哪里？

### 3.1 Agent loop 注册点

注册机制在：

- `verl/experimental/agent_loop/agent_loop.py`

里面有：

- `_agent_loop_registry`
- `register(agent_name)`

`ToolAgentLoop` 在这里注册：

- `verl/experimental/agent_loop/tool_agent_loop.py`

具体是：

- `@register("tool_agent")`

因此 `agent_name="tool_agent"` 时，最终会实例化 `ToolAgentLoop`。

### 3.2 真正执行入口

整体运行入口是：

- `AgentLoopWorker.generate_sequences()` in `verl/experimental/agent_loop/agent_loop.py`

执行流程是：

1. 从 batch 里读出每个样本的 `agent_name`。
2. 用 `_agent_loop_registry[agent_name]` 找到对应 agent loop 配置。
3. `hydra.utils.instantiate(...)` 实例化对应 AgentLoop。
4. 调用 `agent_loop.run(sampling_params, **kwargs)`。

对于通用 tool agent：

- `agent_name="tool_agent"` -> `ToolAgentLoop.run()`

对于当前 VTool recipe：

- `agent_name="vtool_agent"` -> `recipe.vtool.vtool.VToolAgentLoop.run()`

### 3.3 `tool_agent` 和 `vtool_agent` 的区别

- `tool_agent`：通用多轮 tool-calling agent。
- `vtool_agent`：VTool 自己的 one-shot 特化 loop，不走通用 tool registry 主流程。

## 4. 模型生成的 `tool_call` 如何被解析？

### 4.1 解析入口

在：

- `verl/experimental/agent_loop/tool_agent_loop.py`

模型生成完一轮 token 后，`_handle_generating_state()` 会执行：

- `self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)`

这里的 `self.tool_parser` 来自：

- `ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)`

### 4.2 Tool parser 定义

在：

- `verl/experimental/agent_loop/tool_parser.py`

当前内置至少三种 parser：

- `hermes`
- `gpt-oss`
- `qwen3_coder`

### 4.3 解析逻辑

不同 parser 对模型输出格式的假设不同：

- `HermesToolParser`
  - 从 `<tool_call> ... </tool_call>` 中抽 JSON；
- `GptOssToolParser`
  - 从 Harmony 风格 `<|start|>assistant ... to=functions.xxx ...` 中抽；
- `Qwen3XMLToolParser`
  - 从 XML-like `<tool_call><function=...><parameter=...>` 中抽。

统一产物是：

- `FunctionCall(name, arguments)`

其中：

- `name` 是 tool 名；
- `arguments` 是 JSON 字符串。

后续真正执行时，`ToolAgentLoop._call_tool()` 再 `json.loads(tool_call.arguments)`。

## 5. tool observation 如何返回给模型？

### 5.1 Tool 执行

执行入口在：

- `ToolAgentLoop._call_tool()`

流程：

1. 取 `tool_call.name`
2. `json.loads(tool_call.arguments)`
3. `tool = self.tools[tool_name]`
4. `tool.create(...)`
5. `tool.execute(instance_id, tool_args, agent_data=agent_data)`
6. `tool.release(instance_id)`

tool 的返回值是：

- `ToolResponse(text=..., image=..., video=...)`

### 5.2 Observation 组装回消息

在：

- `ToolAgentLoop._handle_processing_tools_state()`

执行结果会被包装为一条新的 `{"role": "tool", "content": ...}` 消息。

两种情况：

- 纯文本 tool：
  - `{"role": "tool", "content": tool_response.text}`
- 多模态 tool：
  - `{"role": "tool", "content": [{"type": "image"}, {"type": "text", ...}, ...]}`

### 5.3 Observation 再次 tokenization

然后 tool observation 会重新编码进上下文，具体分两类：

- `gpt-oss`：走 `build_gpt_oss_tool_response_text(...)` 手工格式化；
- 其它 parser：走 `apply_chat_template(add_messages, remove_system_prompt=True)`

生成的 observation token：

- 被追加到 `agent_data.prompt_ids`
- 对应 `response_mask` 为 `0`

也就是说，在训练/rollout 视角：

- 模型自己生成的 token：`response_mask=1`
- tool observation 插进去的 token：`response_mask=0`

这是 PPO / GRPO 里只训练模型输出、不训练 observation token 的关键。

### 5.4 多模态 observation

如果 tool 返回新图片：

- `tool_response.image` 会被追加进 `agent_data.image_data`
- 下一轮生成时，这些图片会继续作为新的视觉上下文传给模型

## 6. `agent_name` 如何在数据里指定？

### 6.1 数据行里直接写 `agent_name`

最直接的方式是在训练/评测数据里放一列：

- `agent_name`

例如：

- `examples/data_preprocess/gsm8k_tool_agent_loop.py`

里面构造数据时直接写：

- `"agent_name": "tool_agent"`

当前 `recipe/vtool` 则要求：

- `"agent_name": "vtool_agent"`

### 6.2 如果不写 `agent_name`

在：

- `AgentLoopWorker.generate_sequences()`

如果 batch 里没有 `agent_name`，会回退到：

- `config.actor_rollout_ref.rollout.agent.default_agent_loop`

因此可以：

- 数据里显式指定每条样本用哪个 agent；
- 或者整个任务共用一个默认 agent。

### 6.3 `agent_name` 对应什么

对应的是 agent loop registry 的 key。

来源可以是两种：

- 代码里 `@register("tool_agent")` 这种 decorator；
- `agent_loop_config_path` 对应的 Hydra YAML。

`recipe/vtool/agent.yaml` 就是后者：

- name: `vtool_agent`
- `_target_`: `recipe.vtool.vtool.VToolAgentLoop`

## 7. eval 数据格式是什么？

### 7.1 Standalone eval 的读取入口

主要在：

- `eval/run_eval.py`
- `eval/dataset.py`

`eval/dataset.py` 定义了 `PreparedExample`，里面关心的字段有：

- `uid`
- `query`
- `ground_truth`
- `raw_prompt`
- `multi_modal_data`
- `tools_kwargs`
- `original_record`

### 7.2 评测样本最核心字段

从 `eval/dataset.py` 看，最稳妥的数据字段是：

- `prompt` 或 `messages`
- `images`
- `reward_model.ground_truth` 或等价 answer 字段
- `extra_info.tools_kwargs`

更具体地说：

#### 文本/消息

优先读：

- `prompt`
- 否则 `messages`
- 否则退化为 `query/question/prompt`

#### 图像

默认图像列名：

- `images`

支持多种 image payload：

- PIL
- `{"bytes": ...}`
- `{"path": ...}`
- `{"array": ...}`
- 本地路径字符串

#### Ground truth

`eval/dataset.py` 会从这些候选里抽：

- `ground_truth`
- `answer`
- `extra_info.answer`
- `extra_info.ground_truth`

#### Tools kwargs

优先读：

- `row["tools_kwargs"]`
- 否则 `extra_info["tools_kwargs"]`

并会把：

- `metadata` dict 序列化成 JSON string

### 7.3 训练侧 RL dataset 的格式约束

训练侧在：

- `verl/utils/dataset/rl_dataset.py`

`__getitem__()` 会补充：

- `raw_prompt`
- `index`
- `tools_kwargs`
- `interaction_kwargs`

并依赖：

- `extra_info.need_tools_kwargs`
- `extra_info.tools_kwargs`

所以如果后续新 recipe 想沿用现有机制，建议继续遵守这一 contract。

## 8. `reward_manager` / `reward_fn` 如何接入？

### 8.1 总入口

在：

- `verl/trainer/ppo/reward.py`

核心函数：

- `get_custom_reward_fn(config)`
- `load_reward_manager(config, tokenizer, **reward_kwargs)`

### 8.2 reward function 的接法

配置里可以指定：

- `reward.custom_reward_function.path`
- `reward.custom_reward_function.name`

然后 `load_extern_object()` 动态导入。

最终 `compute_score(...)` 会被包装成统一 callable。

### 8.3 reward manager 的接法

配置里还可以指定 reward manager：

- `reward.reward_manager.source`
- `reward.reward_manager.name`
- `reward.reward_manager.module.path`

两种来源：

- `source: register`
- `source: importlib`

`recipe/vtool/refocus_multiturn_grpo.yaml` 里走的是 `importlib`：

- reward function: `recipe.vtool.vtool.compute_score`
- reward manager: `recipe.vtool.vtool.VToolRewardManager`

### 8.4 reward manager 在什么时机拿到 agent loop 输出

`AgentLoopWorker._compute_score()` 会把这些打包进 `DataProto`：

- `prompts`
- `responses`
- `attention_mask`
- `response_mask`
- `tool_extra_fields`
- `__num_turns__`
- 原始 `non_tensor_batch` 字段

`tool_extra_fields` 就是 `AgentLoopOutput.extra_fields`。

因此自定义 agent loop 若需要 reward 使用额外信息，最直接的方法是：

- 在 `output.extra_fields` 里写入；
- reward manager 在 `run_single()` 里从 `data_item.non_tensor_batch["tool_extra_fields"]` 取回。

### 8.5 `recipe/vtool` 的 reward 设计

`VToolRewardManager` 做了两件事：

1. 优先从 `tool_extra_fields["vtool_final_response_text"]` 取 final answer；
2. 如果没有，再按 `response_mask` 从 response token 中抽最后可训练段解码。

这说明：

- reward manager 可以只奖 final answer；
- 不一定要按整段 response 解码。

## 9. `recipe/vtool` 和 `verl.experimental.agent_loop` 的关系是什么？

### 9.1 相同点

二者都在用同一个大框架：

- 都实现 `AgentLoopBase.run()`
- 都被 `agent_name` 调度
- 都输出 `AgentLoopOutput`
- 都接入统一 reward loop

也就是说，`recipe/vtool` 并不是完全脱离 `agent_loop` 系统，而是这个系统里的一个自定义 agent loop。

### 9.2 不同点

`ToolAgentLoop` 是通用多轮工具框架：

- 工具来源于 tool config；
- 工具 schema 走 `OpenAIFunctionToolSchema`；
- 模型输出显式 `tool_call`；
- parser 负责解析；
- observation 自动回灌；
- 可并行多个 tool call。

`recipe/vtool.VToolAgentLoop` 是专用 one-shot 路径：

- 不走 tool registry；
- 不要求模型输出 OpenAI tool_call；
- 而是让模型输出 Python 代码；
- 用 `RefocusCodeParser` 解析；
- 本地执行 refocus 代码；
- 拼接固定 observation；
- 再要求模型输出 final answer。

### 9.3 关系判断

可以把二者理解成：

- `verl.experimental.agent_loop.tool_agent_loop.ToolAgentLoop`：通用基础设施；
- `recipe/vtool.vtool.VToolAgentLoop`：历史/专项 recipe，在基础设施之上另起了一条专门 loop。

SafeVTool-R1 如果要支持多种安全工具、可插拔 schema、多轮视觉证据收集，那么更适合靠近前者，而不是继续沿用 `VToolAgentLoop` 这种“生成代码再执行”的单一形态。

## 10. SafeVTool-R1 最小侵入式应该新增哪些文件、修改哪些文件？

这里只给结构建议，不写实现。

### 10.1 应尽量新增，不动现有通用框架逻辑

最小侵入式原则下，优先新增这些文件：

#### A. 新 recipe 目录

- `recipe/safe_vtool/agent.yaml`
- `recipe/safe_vtool/safe_vtool_agent.py`
- `recipe/safe_vtool/safe_reward_manager.py`
- `recipe/safe_vtool/safety_agent_prompt.txt`
- `recipe/safe_vtool/safety_tools_config.yaml`
- `recipe/safe_vtool/convert_to_vtool_format.py`

原因：

- `agent.yaml` 用来注册新的 `agent_name`
- `safe_vtool_agent.py` 作为 `ToolAgentLoop` 的薄封装，而不是重写整套通用框架
- reward / 数据转换 / prompt / tool config 都放 recipe 自己目录，隔离影响面

#### B. 新 tool 实现

新增到：

- `verl/tools/`

例如一个单独文件：

- `verl/tools/safe_vtool_tools.py`

原因：

- 现有 tool registry 就是从这里类加载；
- 不需要改 registry 框架本身。

#### C. 新 judge / reward 纯逻辑

可以新增：

- `judge/safety_judge.py`
- `judge/safety_reward.py`

或者放到 `recipe/safe_vtool/` 内部也可以，但应保持：

- 一个纯 judge 模块；
- 一个 reward entrypoint。

#### D. 新 eval 脚本

如需 standalone eval，建议新增：

- `eval/safety_eval.py`
- `eval/safety_metrics.py`
- `eval/score_safety_results.py`
- `eval/run_safety_eval.sh`

这样不污染现有 `eval/run_eval.py` 的 VTool 专项逻辑。

### 10.2 应尽量少改的现有文件

理论上可以做到完全不改或只改文档：

- `verl/tools/base_tool.py`
- `verl/tools/utils/tool_registry.py`
- `verl/experimental/agent_loop/tool_agent_loop.py`
- `verl/experimental/agent_loop/tool_parser.py`
- `verl/trainer/ppo/reward.py`

原因：

- 这些已经提供了足够的扩展点。

### 10.3 可能需要的轻量修改

如果 SafeVTool-R1 需要更强 trace 持久化、或某些字段标准化，可能需要非常轻量地修改：

- `eval/dataset.py`
  - 如果现有 eval 数据字段不足以承载安全 gold / trace；
- `eval/run_eval.py`
  - 如果希望复用同一 CLI，而不是单独写 `safety_eval.py`；
- 文档或 recipe README

但从当前结构看，首选仍然是：

- 新增文件；
- 少改已有框架。

### 10.4 不建议直接修改的地方

#### 不建议把 SafeVTool-R1 直接堆进 `recipe/vtool/vtool.py`

原因：

- 当前 `VToolAgentLoop` 是代码执行型 one-shot loop；
- 与通用 tool schema / 多 tool call / 多轮视觉工具链并不一致；
- 会把“图像 refocus recipe”与“安全多工具 agent”耦合在一起。

#### 不建议为了 SafeVTool-R1 改动通用 parser / registry 核心逻辑

除非发现明确缺口，否则：

- tool registry 已经够用；
- parser 已经支持多个格式；
- reward manager 接口也已存在。

SafeVTool-R1 最小侵入式应优先“复用 + 包装”，而不是“改框架底层”。

## 推荐实施方向

如果后续进入实现阶段，建议路线是：

1. 复用 `ToolAgentLoop`，新建 `safe_vtool_agent.py`。
2. 把安全工具实现为新的 `BaseTool` 子类，并通过新的 tool config 注册。
3. 数据继续遵守现有 RL/eval contract：
   - `agent_name`
   - `prompt`
   - `images`
   - `reward_model.ground_truth`
   - `extra_info.tools_kwargs`
4. reward 走 `reward.py` 现有导入机制。
5. `recipe/vtool` 仅作为：
   - 当前 VTool 数据/奖励风格参考；
   - 不作为 SafeVTool-R1 主执行框架。

这条路线与当前仓库结构最一致，侵入最小，后续也最容易接到 PPO / GRPO / standalone eval。
