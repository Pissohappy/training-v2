# SafeVTool 中文说明

本仓库是Safely Thinking with Image的代码仓库，从training-v2的verl代码仓库fork和改写而来。本README目标是说明以下四件事：

1. 当前仓库中已经实现了什么。
2. 之前已经完成并跑通了哪些实验。
3. 如果需要复现已有结果，应该如何操作。
4. 如果要继续扩展这条链路，应该从哪些代码位置入手。

本文档聚焦 `training-v2` 仓库内与 `safe` / `safe_vtool` 直接相关的代码、脚本和结果目录，不展开介绍仓库其他训练 recipe，如需查看请见README.md。

## 项目概览

当前仓库已经具备一条完整的视觉安全 agent 评测链路，核心能力包括：

- 基于 `verl` 多轮 tool agent 框架构建安全视觉助手。
- 支持按消融模式动态启停 OCR、grounding、layout parsing、policy check 等工具。
- 支持使用同一个 OpenAI-compatible VLM 作为自举式视觉工具 backend。
- 支持对 OmniSafeBench-MM 风格测试集进行批量生成、批量 judge 和结果汇总。
- 支持将评测结果进一步送入 benchmark evaluator，得到更标准化的通用能力评估结果。

这套实现当前最成熟的部分是“评测与打分链路”。仓库里已经具备数据转换、agent loop、工具实现、judge、reward manager、批量运行脚本与结果汇总脚本。相比之下，`safe` 专用的一键 RL 训练 recipe 暂时还未做详细的兼容实现，因此如果后续要继续做训练侧集成，需要在现有评测链路基础上继续补强。

## 当前已经完成的工作

目前 `safe_vtool` 方向已经完成的工作可以概括为四类。

### 1. 安全视觉 agent 框架

仓库中已经实现了一个名为 `safe_vtool_agent` 的多轮安全视觉 agent。该 agent 能够：

- 自动注入 safety prompt 或 neutral prompt。
- 按 `ablation_mode` 控制可用工具集合。
- 在多轮推理过程中记录完整 tool trace。

### 2. 安全工具链

当前已经实现并接入的工具包括：

- `SafetyOCRTool`
- `CropZoomTool`
- `GroundingTool`
- `LayoutParseTool`
- `PolicyCheckTool`

这些工具支持不同的Tool backend，可扩展。

### 3. 安全 judge 与 reward

仓库里已经实现：

- 配合OmniSafetyBench-MM的Judge；（需要先在本代码仓库roll-out生成，然后接入OmniSafetyBench-MM）
- 基于结构化输出字段的安全 judge；
- 按 decision / policy / evidence / tool use / leakage 等维度打分；
- 将 judge 结果进一步包装为 reward，供 reward loop 使用。

### 4. 端到端实验脚本

仓库中已经可以直接完成：

- 批量生成安全 benchmark 响应；
- 批量运行 judge；
- 一键拉起模型服务并串联 generation + judge；
- 对 `mmbench` / `mmmu` 类 benchmark 做 neutral/safety 两套 prompt 下的结果评估；
- 将生成结果送入 benchmark evaluator，得到汇总报告。

## 仓库结构与关键代码位置

### `recipe/safe_vtool/`

这是 `safe_vtool` 的核心目录。

- `recipe/safe_vtool/agent.yaml`
  - agent 注册入口。
  - 当前只注册了一个 agent：`safe_vtool_agent`。
- `recipe/safe_vtool/safe_vtool_agent.py`
  - `SafeVToolAgentLoop` 的实现。
  - 负责 prompt 注入、工具筛选、tool trace 记录、最终回答保存。
- `recipe/safe_vtool/common.py`
  - 统一维护：
    - ablation mode 定义；
    - prompt variant 定义；
    - tool alias 到真实 tool name 的映射；
    - 按模式过滤工具配置的逻辑。
- `recipe/safe_vtool/safety_tools_config.yaml`
  - 所有 safe 工具的主配置文件。
  - 定义默认 backend、fallback 顺序、VLM tool client 参数等。
- `recipe/safe_vtool/safety_agent_prompt.txt`
  - 默认安全 prompt。
- `recipe/safe_vtool/safety_agent_prompt_neutral.txt`
  - neutral prompt。
- `recipe/safe_vtool/convert_to_vtool_format.py`
  - 将原始 JSONL 与人工标注 gold 数据转换成 `verl`/`vtool` 可用格式。
- `recipe/safe_vtool/safe_reward_manager.py`
  - 安全任务的 reward manager。
- `recipe/safe_vtool/vlm_tool_client.py`
  - self-VLM 工具后端的接口封装。

### `verl/tools/safe_vtool_tools.py`

这是所有 safe 工具实现的主文件。

当前已经实现的主要工具类包括：

- `SafetyOCRTool`
- `CropZoomTool`
- `GroundingTool`
- `LayoutParseTool`
- `PolicyCheckTool`

如果后续要新增工具、修改 fallback 策略、增加 backend，优先从这个文件入手。

### `judge/`

- `judge/safety_judge.py`
  - 安全评测打分主逻辑。
- `judge/safety_reward.py`
  - 将 judge 结果转为 reward 输出。

如果后续要调整评分标准、修改 refusal/over-refusal/harmful leakage 的定义，应从这里修改。

### `eval/`

- `eval/run_omni_safe_vtool.py`
  - 目前最重要的安全评测生成入口。
  - 负责读取 OmniSafeBench-MM 测试样本并生成：
    - `*.trace.jsonl`
    - `*.responses.jsonl`
- `eval/run_omni_judge.py`
  - 对 `responses.jsonl` 批量 judge。
- `eval/safety_eval.py`
  - 面向更通用数据集格式的 safe eval 入口。
- `eval/safety_metrics.py`
  - 汇总安全相关 metrics。
- `eval/score_safety_results.py`
  - 对已有结果文件重新计算 metrics。

### `bash/`

这是当前最常用的实验入口脚本目录。

- `bash/run_all_omni_generate.sh`
  - 批量生成 OmniSafeBench-MM 各子集的响应和 trace。
- `bash/run_all_omni_judge.sh`
  - 对生成结果批量 judge。
- `bash/run_all_omni_pipeline.sh`
  - 一键跑完整 pipeline：起服务、生成、judge、日志归档。
- `bash/run_all_general_bench.sh`
  - 运行 `mmbench` / `mmmu` 的 neutral/safety 两组实验。
- `bash/eval_general_bench_results.sh`
  - 将 `mmbench/mmmu` 的响应送入 benchmark evaluator，输出正式汇总结果。

### `tests/safe_vtool/`

- `tests/safe_vtool/test_safe_vtool.py`

这里已经覆盖了当前 safe 模块最关键的基本能力，包括：

- converter 是否生成正确格式；
- tool config 是否能成功注册全部工具；
- 各工具的基础 contract；
- agent trace 是否正确累积；
- judge 的核心 edge case；
- ablation mode 映射；
- backend metrics 聚合。

如果后续修改 safe 主逻辑，建议优先补和更新这里的测试。

## 路径设计说明

当前 `safe_vtool` 相关实现采用分层设计，而不是把所有逻辑集中在单个脚本中。

### 第一层：任务定义与配置

位于 `recipe/safe_vtool/`。

这一层负责：

- 定义 agent；
- 定义 prompt；
- 定义工具配置；
- 定义 reward manager；
- 定义数据转换逻辑。

### 第二层：工具执行层

位于 `verl/tools/safe_vtool_tools.py`。

这一层负责真正的 OCR、grounding、layout parsing、policy lookup 等执行逻辑。

### 第三层：判分层

位于 `judge/`。

这一层负责把模型的结构化回答转换为安全评分。

### 第四层：评测编排层

位于 `eval/`。

这一层负责读数据、调模型、调工具、写结果、聚合指标。

### 第五层：批量运行与复现入口

位于 `bash/`。

这一层负责将上面各层串联成可以重复执行的实验命令。

这种结构的好处是：

- agent 行为和工具实现解耦；
- 工具启停可以通过配置和 metadata 控制，不必频繁改代码；
- judge/reward 可以独立演进；
- 结果文件分层清晰，便于排查与复算；
- 后续扩展训练链路时，也可以复用现有的 agent、tools 和 reward 逻辑。

## 关键配置：ablation mode 与 prompt variant

### ablation mode

在 `recipe/safe_vtool/common.py` 中，当前已经定义五种模式：

- `no_tools`
- `self_vlm_tools`
- `external_tools`
- `full_safevtool`
- `oracle_tools`

可以简单理解为：

- `no_tools`
  - 不使用工具，只靠图像与文本直接回答。
- `self_vlm_tools`
  - 使用视觉工具，但不启用 policy check。
- `external_tools`
  - 更偏向外部/启发式 fallback 路径。
- `full_safevtool`
  - 视觉工具加 `policy_check_tool`，是最完整的 safe 模式。
- `oracle_tools`
  - 用于 debug 或上界分析，不适合作为正式主结果。

### prompt variant

当前支持：

- `safety`
- `neutral`
- `none`

其中：

- `safety` 用于强化安全导向行为；
- `neutral` 更适合对通用 benchmark 做对照实验；
- `none` 一般仅用于调试。

## 当前已有结果目录

目前与这条链路直接相关、且已经实际产生结果的目录主要有两个。

### 1. SafeVTool 主结果目录

`/mnt/disk1/szchen/VLMAlignment/training-v2/eval/results/omni_safe_vtool`

这个目录保存模型的原始生成结果、tool trace 和 judge 结果，是第一层实验产物目录。

当前该目录下已经可以看到：

- benchmark 结果文件：
  - `mmbench.trace.jsonl`
  - `mmbench.responses.jsonl`
  - `mmmu.trace.jsonl`
  - `mmmu.responses.jsonl`
  - `mmbench_safety.trace.jsonl`
  - `mmbench_safety.responses.jsonl`
  - `mmmu_safety.trace.jsonl`
  - `mmmu_safety.responses.jsonl`
- 各安全子集目录：
  - `advbenchm/`
  - `arttextfigstep/`
  - `figstep/`
  - `holisafe/`
  - `jailbreakv28k/`
  - `mmsafetybench/`
  - `mossbench/`
  - `mssbench/`
  - `sd35_figstep/`
  - `spa_vl/`

各子集目录中通常包含：

- `xxx.trace.jsonl`
- `xxx.responses.jsonl`
- `xxx.judged.jsonl`
- `xxx.judged.summary.json`

含义分别是：

- `trace.jsonl`
  - 最完整的逐样本执行轨迹。
- `responses.jsonl`
  - 面向 judge / benchmark evaluator 的精简响应输出。
- `judged.jsonl`
  - judge 后的逐样本结果。
- `judged.summary.json`
  - judge 汇总结果。

### 2. benchmark 汇总目录

`/mnt/disk1/szchen/VLMAlignment/training-v2/eval/results/omni_benchmark_eval`

这个目录保存的是第二层产物，即将 `omni_safe_vtool` 中的 benchmark 响应再送入 evaluator 之后得到的汇总结果。

当前目录结构中已经能看到：

- `neutral/benchmark_defense_utility_glm46_20260521/...`
- `safety/benchmark_defense_utility_glm46_20260521/...`

每一组实验下最重要的文件通常是：

- `evaluation_report.json`
- `attack_mmbench_model_GLM-4.6V-Flash_defense_None_evaluator_benchmark_eval.jsonl`
- `attack_mmmu_model_GLM-4.6V-Flash_defense_None_evaluator_benchmark_eval.jsonl`

这两个结果目录的关系需要明确区分：

- `omni_safe_vtool` 是模型执行与 judge 的直接输出。
- `omni_benchmark_eval` 是将其中 benchmark 响应进一步送入 evaluator 后得到的汇总结果。

前者更适合排查模型行为和工具调用，后者更适合正式汇报 benchmark 表现。

## 如何复现之前已经完成的工作

下面给出推荐的复现路径。不同层次的复现需求，可以选择不同脚本。

### 复现目标一：批量生成 OmniSafeBench 安全子集结果

进入仓库根目录：

```bash
cd /mnt/disk1/szchen/VLMAlignment/training-v2
```

直接运行：

```bash
bash bash/run_all_omni_generate.sh
```

该脚本默认会：

- 从 `OmniSafeBench-MM` 的测试样本目录扫描所有 `test_cases.jsonl`；
- 对每个数据集生成：
  - `trace.jsonl`
  - `responses.jsonl`
- 默认结果输出到：
  - `eval/results/omni_safe_vtool`

脚本中的关键默认参数包括：

```bash
OMNI_ROOT=/mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM
TEST_CASES_ROOT=${OMNI_ROOT}/output_sample/test_cases
MODEL_NAME=GLM-4.6V-Flash
SERVER_BASE_URL=http://127.0.0.1:8000/v1
ABLATION_MODE=self_vlm_tools
PROMPT_VARIANT=safety
RESULT_ROOT=eval/results/omni_safe_vtool
```

如果复现实验时模型服务地址、模型名或结果目录不同，可以通过环境变量覆盖。

### 复现目标二：对已生成结果批量 judge

在仓库根目录运行：

```bash
cd /mnt/disk1/szchen/VLMAlignment/training-v2
RESULT_ROOT=eval/results/omni_safe_vtool \
JUDGE_MODEL=gpt-oss-120b \
JUDGE_PROVIDER=any \
JUDGE_BASE_URL=http://127.0.0.1:8015/v1 \
JUDGE_API_KEY=EMPTY \
bash bash/run_all_omni_judge.sh
```

该脚本会在各数据集目录中补充：

- `*.judged.jsonl`
- `*.judged.summary.json`

### 复现目标三：一键跑完整 pipeline

如果希望从模型服务启动到 judge 全流程自动完成，使用：

```bash
cd /mnt/disk1/szchen/VLMAlignment/training-v2
bash bash/run_all_omni_pipeline.sh
```

该脚本当前默认负责：

1. 启动被测模型服务。
2. 运行 `bash/run_all_omni_generate.sh`。
3. 关闭被测模型服务。
4. 启动 judge 模型服务。
5. 运行 `bash/run_all_omni_judge.sh`。
6. 将日志写入 `logs/omni_pipeline/<RUN_ID>/`。

如果后续要做大规模复现，推荐优先使用这个脚本作为基准入口。

### 复现目标四：复现当前 general benchmark 实验

当前仓库中 `mmbench/mmmu` 的 neutral/safety 两组实验已经被脚本化，直接运行：

```bash
cd /mnt/disk1/szchen/VLMAlignment/training-v2
bash bash/run_all_general_bench.sh
```

这个脚本会执行四组任务：

- `mmbench` + `neutral`
- `mmmu` + `neutral`
- `mmbench` + `safety`
- `mmmu` + `safety`

对应的底层命令都是 `python -m eval.run_omni_safe_vtool`，输出写到：

- `eval/results/omni_safe_vtool/mmbench.trace.jsonl`
- `eval/results/omni_safe_vtool/mmbench.responses.jsonl`
- `eval/results/omni_safe_vtool/mmmu.trace.jsonl`
- `eval/results/omni_safe_vtool/mmmu.responses.jsonl`
- `eval/results/omni_safe_vtool/mmbench_safety.trace.jsonl`
- `eval/results/omni_safe_vtool/mmbench_safety.responses.jsonl`
- `eval/results/omni_safe_vtool/mmmu_safety.trace.jsonl`
- `eval/results/omni_safe_vtool/mmmu_safety.responses.jsonl`

### 复现目标五：生成 benchmark evaluator 汇总结果

在已有 `mmbench/mmmu` 响应文件的前提下，运行：

```bash
cd /mnt/disk1/szchen/VLMAlignment/training-v2
bash bash/eval_general_bench_results.sh
```

该脚本会读取：

- `eval/results/omni_safe_vtool/mmbench.responses.jsonl`
- `eval/results/omni_safe_vtool/mmmu.responses.jsonl`
- `eval/results/omni_safe_vtool/mmbench_safety.responses.jsonl`
- `eval/results/omni_safe_vtool/mmmu_safety.responses.jsonl`

然后将评估结果输出到：

- `/mnt/disk1/szchen/VLMAlignment/training-v2/eval/results/omni_benchmark_eval/neutral/...`
- `/mnt/disk1/szchen/VLMAlignment/training-v2/eval/results/omni_benchmark_eval/safety/...`

这一步是复现当前 benchmark 汇总结果所必需的。

## 数据流与执行链路

从整体上看，当前 safe 链路的数据流如下：

1. 从 OmniSafeBench-MM 读取原始 `test_cases.jsonl`。
2. `eval/run_omni_safe_vtool.py` 构造多模态 prompt。
3. `recipe/safe_vtool/common.py` 注入 system prompt，并按 ablation mode 过滤工具。
4. `verl/tools/safe_vtool_tools.py` 执行工具调用。
5. `eval/run_omni_safe_vtool.py` 写出 `trace.jsonl` 与 `responses.jsonl`。
6. `eval/run_omni_judge.py` 将 `responses.jsonl` 转为 `judged.jsonl` 和 summary。
7. `bash/eval_general_bench_results.sh` 将 benchmark 响应进一步送入 evaluator。
8. `eval/results/omni_benchmark_eval` 中生成最终 benchmark 汇总结果。

这个链路已经可以稳定支撑“生成 -> judge -> evaluator 汇总”三段式实验流程。

## 仓库目前可以做什么

如果把当前实现当成一个可复用模块来看，仓库目前已经可以支持以下工作：

- 评估视觉大模型在安全场景中的结构化响应能力。
- 比较不同 prompt variant 对 benchmark 结果的影响。
- 比较不同 ablation mode 对工具使用和安全表现的影响。
- 分析 tool trace，定位模型在 OCR、grounding、layout parsing 或 policy 使用上的失败点。
- 输出 judge 结果与 benchmark evaluator 汇总结果，便于后续实验汇报。
- 将安全 judge 逻辑接入 reward manager，为未来训练集成做准备。

## 后续还能做什么

在现有基础上，后续最自然的扩展方向包括：

### 1. 将 safe 评测链路进一步训练化

当前 reward manager 已经存在，但缺少完整、安全任务专用、可直接复用的一键训练 recipe。后续可以：

- 增加 safe 专用训练脚本；
- 将 `safe_vtool_agent` 直接接入现有 RL 或 SFT 训练流程；
- 形成从数据转换到训练、评测、汇总的一体化 pipeline。

### 2. 扩展工具能力

当前工具集已经够支撑基础安全视觉任务，但还可以继续扩展：

- 新增更强的 OCR backend；
- 新增更稳健的 grounding backend；
- 加强 layout parsing 的结构建模；
- 增加更细粒度的 policy retrieval / policy grounding 工具。

### 3. 强化打分体系

当前 judge 已经覆盖主要维度，但后续仍可继续细化：

- 更精细的 refusal 分类；
- 更严格的 evidence grounding 检查；
- 更丰富的 tool-use 质量分析指标；
- 更清晰地区分误拒答和漏防御。

### 4. 统一 benchmark 与安全任务评测接口

目前 `omni_safe_vtool` 与 benchmark evaluator 已经打通，但仍然属于两段式流程。后续可以继续封装：

- 一键从生成结果直接产出 benchmark report；
- 自动归档 neutral/safety 多组实验；
- 统一结果命名、版本命名和日志命名。

## 建议的阅读顺序

对于第一次接手这部分代码的开发者，建议按下面顺序阅读：

1. `recipe/safe_vtool/common.py`
   - 先理解模式控制与 prompt 注入。
2. `recipe/safe_vtool/safe_vtool_agent.py`
   - 再理解 agent loop 如何组织工具调用与结果记录。
3. `recipe/safe_vtool/safety_tools_config.yaml`
   - 明确当前工具配置与 backend 默认值。
4. `verl/tools/safe_vtool_tools.py`
   - 理解工具的真实执行逻辑。
5. `judge/safety_judge.py`
   - 理解最终如何打分。
6. `eval/run_omni_safe_vtool.py`
   - 理解完整评测流程如何把数据、工具和模型串起来。
7. `bash/run_all_general_bench.sh` 与 `bash/eval_general_bench_results.sh`
   - 最后复现已有实验。