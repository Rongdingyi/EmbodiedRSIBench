# Embodied RSI Benchmark：OpenETA + RSI 第一轮实验执行任务书

> **版本**：Pilot v0.3（DeepSeek-Flash smoke-first）  
> **日期**：2026-09-15  
> **用途**：交给 coding agent / Codex / OpenCode 直接执行。  
> **目标**：在已经整理完成的 Embodied RSI Core v1.0 数据上，部署并冻结公开 embodied agent **OpenETA Stage-2**，然后在完全相同的 Agent、模型、任务和环境预算下，对多种 RSI / self-evolution 方法进行第一轮可复现测试。

> **v0.3 改动**：为优先跑通系统，本轮 executor 与所有 RSI updater 统一改为 DeepSeek 官方 API `deepseek-flash`；删除本地 Qwen/vLLM 部署要求，改为 DeepSeek API text + vision smoke gate。

---

# 0. 先读：本任务书到底要验证什么

本阶段**不是**要马上跑完 2,200 条 Core，也不是要追求论文最终数字。

本阶段只验证下面这件事情能不能被可靠地做出来：

\[
\boxed{
\text{Same Tasks}
+
\text{Same Public Agent}
+
\text{Same VLM}
+
\text{Different RSI Method}
}
\]

固定项：

- 任务：来自已构建的 Embodied RSI Benchmark Core v1.0；
- Agent：公开的 OpenETA Stage-2；
- Backbone：DeepSeek 官方 API `deepseek-flash`（当前对应 DeepSeek-V4.1-Flash）；
- simulator：各 source benchmark 的原生 simulator/runtime；
- observation/action budget：同一数据集内完全相同；
- evaluator：同一 task 使用同一个 private evaluator；
- 模型权重：全程 frozen；
- probe：全程 frozen，绝对禁止学习。

唯一允许变化的是：

> **episode 之间，RSI 方法如何把过去 experience 转化成持久状态，并在未来 episode 中重新利用。**

第一轮只测试：

1. `none`：无持久自进化；
2. `raw_memory`：原始 episodic memory 基线；
3. `ace_context`：ACE 的 context / playbook evolution；
4. `worldmind`：WorldMind 的 Goal + Process Experience；
5. `embodiskill`：EmbodiSkill 的 skill-aware reflection。

第二阶段再考虑：

- TF-GRPO；
- PRACTICE；
- AHE；
- SHAPER；
- parameter update / RL track。

**不要在第一轮把这些方法全部塞进来。先把统一协议打通。**

---

# 1. 不可违反的实验原则

## 1.1 OpenETA 是公开 Agent，不是本项目自己设计的 Agent

主 Agent 固定为：

- Repository: `https://github.com/OpenMOSS/OpenETA`
- Canonical Pilot Commit:

```text
7d4a0a1522ba8ebbd362bde880bad81d2a98f15e
```

该 commit 对应 OpenETA 2026-09-06 发布的 Stage-2 Agent Harness。

本项目允许：

- 写环境 adapter；
- 写 benchmark runner；
- 写 RSI plugin；
- 写 evaluator bridge；
- 写日志和审计代码。

本项目**不允许**为了提高结果去改：

- OpenETA planner；
- OpenETA planner prompt；
- OpenETA episode loop；
- OpenETA action validation；
- OpenETA context compression；
- OpenETA retry/fallback 行为；
- OpenETA completion logic；
- OpenETA model backend semantics。

论文最终必须能够准确描述为：

> We use the publicly released OpenETA Stage-2 agent harness as the canonical embodied-agent substrate. We do not modify its planning or closed-loop execution policy; we only implement environment adapters and external self-evolution modules.

---

## 1.2 OpenETA 自己的 self-improvement 必须关闭

这是一个**硬性要求**。

当前 OpenETA Stage-2 本身已经实现：

```text
agent/runtime/self_improvement.py
```

其中：

```python
@dataclass(frozen=True, slots=True)
class SelfImprovementConfig:
    enabled: bool = True
```

并且 runtime assembly 会创建：

```python
SelfImprovementReviewer(...)
```

因此：

> 仅仅“不调用 openeta --command iterate”不够。

必须显式关闭 native self-improvement。

Canonical runner 在 runtime assembly 完成后必须执行等价逻辑：

```python
from dataclasses import replace

reviewer = runtime.self_improvement_reviewer
reviewer.config = replace(
    reviewer.config,
    enabled=False,
    auto_apply_reviewed=False,
)
reviewer.auto_applier = None
```

然后 assert：

```python
assert runtime.self_improvement_reviewer.config.enabled is False
assert runtime.self_improvement_reviewer.config.auto_apply_reviewed is False
assert runtime.self_improvement_reviewer.auto_applier is None
```

Canonical experiments 禁止调用：

```text
openeta --command iterate
/approve-skill-update
register_skill
update_skill
native task-playbook promotion
native skill-review promotion
```

若发现 OpenETA 原生 skill/playbook 在 experiment 过程中发生写入：

```text
FINAL_STATUS = FAIL
```

立即停止当前 run。

---

## 1.3 跨 episode 的可变状态只能属于 RSI 方法

每个 episode 使用独立 OpenETA session。

OpenETA session-local memory 可以用于：

- 当前 episode 的 history；
- 当前 episode 的 observation；
- 当前 episode 的 action/result；
- 当前 episode 的临时 working memory。

episode 结束后：

```text
OpenETA session state → 不得作为下一 episode 的隐式 memory
```

跨 episode 唯一可以保留的是：

```text
rsi_state/
```

并且必须由所选 RSI plugin 显式拥有。

例如：

```text
none/
    state = empty

raw_memory/
    episodes.jsonl
    index/

ace_context/
    playbook.txt
    curator_ops.jsonl

worldmind/
    goal_experience/
    process_experience/

embodiskill/
    skill/current.json
    skill/versions/
    reflections/
```

任何跨 episode 状态如果没有明确归属到 RSI state：

```text
FAIL
```

---

## 1.4 Probe 绝对禁止更新

Core v1.0 已经区分：

- Experience: 1500；
- ID Probe: 300；
- Transfer Probe: 300；
- Retention Probe: 100。

规则：

```text
Experience
    可以 update RSI state

ID / Transfer / Retention Probe
    只能读取 RSI state
    禁止 update
```

每次 probe：

1. 对当前 RSI state 做 snapshot；
2. clone snapshot；
3. 以 `update_enabled=False` 运行 probe；
4. probe 完成后计算 clone state hash；
5. 必须和 probe 前 hash 完全相同；
6. 丢弃 clone；
7. 原始 state 继续后续 experience。

如果：

```text
hash_before_probe != hash_after_probe
```

则：

```text
FINAL_STATUS = FAIL
```

一次都不能发生。

---

# 2. 第一轮方法为什么只选这 5 个条件

## 2.1 Condition A：`none`

目的：

> 测 OpenETA + `deepseek-flash` 本身随着任务流运行，在没有任何跨 episode persistent learning 时的表现。

要求：

- 每个 episode 新 session；
- 无 cross-episode memory；
- 无 dynamic skill；
- OpenETA native self-improvement disabled；
- 相同模型、工具、budget。

它是所有 Evolution Gain 的零点。

---

## 2.2 Condition B：`raw_memory`

这是本 benchmark 自己实现的**最低限度 experience reuse baseline**。

它不做：

- LLM reflection；
- skill abstraction；
- playbook editing；
- world-rule induction；
- outcome prediction。

只保存过去 episode 的公开轨迹摘要，并在新任务前 retrieval。

用途：

> 回答“复杂 RSI 是否真的优于简单地把过去 experience 检索回来”。

如果 ACE / WorldMind / EmbodiSkill 连 Raw Memory 都打不过，必须如实报告。

---

## 2.3 Condition C：`ace_context`

官方代码：

- Repository: `https://github.com/ace-agent/ace`
- Pilot Pin:

```text
82709de050e1db6e6ef2f07bcb0393560b94992a
```

ACE 原方法使用：

- Generator；
- Reflector；
- Curator；
- incremental delta updates；
- evolving playbook。

本 benchmark 中：

- OpenETA 是 task executor / physical Generator；
- ACE Reflector 分析 OpenETA trajectory；
- ACE Curator 更新 playbook；
- 下一 episode 把 playbook 作为 persistent context 注入 OpenETA。

注意：

ACE README 当前仍把 “Extending ACE for Tool Calling” 标记为 coming soon，因此本项目**不能宣称“原封不动的 end-to-end ACE 已原生支持 embodied tool calling”**。

方法名建议在代码与论文中写为：

```text
ACE-Context (OpenETA executor adaptation)
```

并明确：

> ACE 的 Reflector / Curator / playbook update mechanism 来自官方实现；执行器由统一的 OpenETA 替代，以保证所有 RSI 方法使用同一 embodied agent。

---

## 2.4 Condition D：`worldmind`

官方代码：

- Repository: `https://github.com/zjunlp/WorldMind`
- Pilot Pin:

```text
712b0fd53b4bd6a1948603f087a1e9f25a5adaea
```

优先使用官方：

```text
Plugin/worldmind_plugin/
```

不要直接修改 WorldMind 自带的 EmbodiedBench runner。

官方 Plugin 已经提供三个独立模块：

```text
ProcessExperienceModule
GoalExperienceModule
ExperienceRetrievalModule
```

因此它是本轮最容易和统一 OpenETA agent 解耦的 embodied RSI 方法。

---

## 2.5 Condition E：`embodiskill`

官方代码：

- Repository: `https://github.com/air-embodied-brain/EmbodiSkill`
- Pilot Pin:

```text
760126030eab1d33ec6a6f30988f0f1fb58df3a7
```

核心代码位于：

```text
agentkit/skill/embodiskill_skill/
    EmbodiSkill.py
    prompt.py
    skill_base.py
```

公开 Quick Start 当前主要是 ALFWorld，因此该方法在我们的统一 benchmark 上需要一个薄 adapter。

**不允许根据论文自己重写一套“类似 EmbodiSkill”的逻辑。**

必须尽量直接调用官方 `EmbodiSkill` 对象和其官方 reflection/manual update 路径。

如果发现为了迁移需要修改官方算法核心超过约 200 行，或必须改变其 reflection/update semantics：

```text
STOP EmbodiSkill integration
```

先完成：

```text
none
raw_memory
ace_context
worldmind
```

并在 `BLOCKERS.md` 记录原因。

不允许静默做一个 pseudo-EmbodiSkill。

---

# 3. 第一轮跑通模型：DeepSeek 官方 `deepseek-flash`

第一轮只使用 DeepSeek 官方 API 模型：

```text
deepseek-flash
```

截至 2026-09-15，DeepSeek 官方文档说明 `deepseek-flash` 指向 **DeepSeek-V4.1-Flash**，并原生支持多模态视觉输入。该 Pilot 的目标是先跑通整个 OpenETA + simulator + RSI + frozen probe 链路，因此暂不部署本地模型服务。

## 3.1 Primary Self-Evolution Track

为了保证真正的 self-evolution：

```text
OpenETA planner = deepseek-flash
ACE reflector = deepseek-flash
ACE curator = deepseek-flash
WorldMind discriminator = deepseek-flash
WorldMind summarizer = deepseek-flash
WorldMind reflector = deepseek-flash
WorldMind extractor/refiner = deepseek-flash
EmbodiSkill updater = deepseek-flash
```

即：

\[
\text{Executor Model} = \text{Updater Model}
\]

这叫：

```text
Primary Self-Evolution Track
```

不要使用 EmbodiSkill README 默认示例中的：

```text
qwen2.5-14b executor + gpt-5.2 updater
```

作为第一轮主结果。

未来可以单独做：

```text
Assisted Evolution Track
```

允许强 teacher/updater，但绝不能混入主表。

---

## 3.2 DeepSeek API 统一配置

创建：

```text
configs/model/deepseek_flash.yaml
```

建议内容：

```yaml
provider: openai_compatible
base_url_env: DEEPSEEK_BASE_URL
api_key_env: DEEPSEEK_API_KEY
model: deepseek-flash
vision: true
# IMPORTANT: OpenETA pinned commit's enable_thinking flag emits Qwen-style
# chat_template_kwargs. Leave it unset / null for DeepSeek official API.
enable_thinking: null
temperature: 0.0   # DeepSeek thinking mode ignores this; kept only because OpenETA config has the field.
top_p: 1.0
max_output_tokens: 8192
request_timeout_s: 180
max_retries: 3
```

注意：DeepSeek `deepseek-flash` 默认开启 thinking mode。官方文档说明 thinking mode 下 `temperature` 不生效，`top_p` 的下限为 0.95。第一轮不要宣称 `temperature=0` 带来 deterministic decoding；这里只要求**所有 condition 使用完全相同的请求配置**。

### OpenETA × DeepSeek 的一个硬兼容规则

Pinned OpenETA commit 中 `OpenAICompatiblePlannerBackendConfig.enable_thinking` 是为 Qwen 风格服务准备的：一旦它不是 `None`，请求体会加入：

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

DeepSeek 官方 API 使用的是：

```json
{"thinking": {"type": "enabled"}}
```

因此本 Pilot **禁止设置 OpenETA 的 `enable_thinking=True/False`**。必须保持 `None` / 未配置，让 OpenETA 不发送 `chat_template_kwargs`，由 DeepSeek 使用默认 thinking mode。

必须新增请求体审计：

```python
assert "chat_template_kwargs" not in outgoing_body
assert requested_model == "deepseek-flash"
```

如果 coding agent 为了关思考而直接改 `external/OpenETA/agent/backends/planner.py`，视为破坏 OpenETA freeze，STOP。

## 3.3 这只是 Smoke/Pilot 模型，不代表论文最终主模型

本轮目标：

```text
先证明 OpenETA + 五套环境 + RSI + snapshot + frozen probe 能完整跑通。
```

所以暂时优先：

- 官方 API 稳定性；
- 原生视觉；
- OpenAI-compatible 接口；
- 低部署成本。

等 G0-G7 全部 PASS 后，再回到最终论文模型矩阵（例如 Qwen3.5-27B / Qwen3.5-9B / 其它架构）做正式比较。

# 4. 建议项目目录

在现有 benchmark 项目下创建：

```text
embodied_rsi/
├── README.md
│
├── external/
│   ├── OpenETA/
│   ├── ace/
│   ├── WorldMind/
│   └── EmbodiSkill/
│
├── benchmark/
│   ├── registry/
│   │   ├── task_registry_normalized.parquet
│   │   ├── selected_core.parquet
│   │   ├── experience.parquet
│   │   ├── id_probe.parquet
│   │   ├── transfer_probe.parquet
│   │   ├── retention_probe.parquet
│   │   ├── id_pairs.parquet
│   │   ├── transfer_pairs.parquet
│   │   └── retention_schedule.json
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── eb_alfred.py
│   │   ├── eb_habitat.py
│   │   ├── spatialworld.py
│   │   ├── asgard.py
│   │   └── tvr.py
│   │
│   ├── openeta_bridge/
│   │   ├── __init__.py
│   │   ├── build_runtime.py
│   │   ├── freeze.py
│   │   ├── tool_registry.py
│   │   ├── episode_runner.py
│   │   ├── context_injection.py
│   │   └── trajectory_export.py
│   │
│   ├── rsi/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── none.py
│   │   ├── raw_memory.py
│   │   ├── ace_context.py
│   │   ├── worldmind.py
│   │   └── embodiskill.py
│   │
│   └── runner/
│       ├── build_pilot.py
│       ├── experience_stream.py
│       ├── frozen_probe.py
│       ├── snapshots.py
│       ├── metrics.py
│       └── audit.py
│
├── configs/
│   ├── agent/
│   │   └── openeta_frozen.yaml
│   ├── model/
│   │   └── deepseek_flash.yaml
│   ├── rsi/
│   │   ├── none.yaml
│   │   ├── raw_memory.yaml
│   │   ├── ace_context.yaml
│   │   ├── worldmind.yaml
│   │   └── embodiskill.yaml
│   └── pilot150.yaml
│
├── manifests/
│   ├── external_sources.json
│   ├── openeta_freeze_manifest.json
│   └── pilot150.json
│
├── scripts/
│   ├── 00_verify_dataset.py
│   ├── 01_clone_pin_external.sh
│   ├── 02_check_deepseek_api.sh
│   ├── 03_verify_openeta_freeze.py
│   ├── 04_validate_adapters.py
│   ├── 05_build_pilot150.py
│   ├── 06_smoke_openeta.py
│   ├── 07_smoke_rsi.py
│   ├── 08_run_pilot150.py
│   ├── 09_analyze_pilot.py
│   └── 10_audit_release.py
│
├── tests/
│   ├── test_no_private_leakage.py
│   ├── test_openeta_frozen.py
│   ├── test_probe_readonly.py
│   ├── test_rsi_state_isolation.py
│   ├── test_snapshot_reload.py
│   └── test_adapter_action_equivalence.py
│
└── outputs/
```

---

# 5. Step 0：确认数据集交付完整

脚本：

```text
scripts/00_verify_dataset.py
```

必须检查：

```python
assert len(raw_registry) == 8046
assert len(selected_core) == 2200
assert len(experience) == 1500
assert len(id_probe) == 300
assert len(transfer_probe) == 300
assert len(retention_probe) == 100
assert len(id_pairs) == 300
assert len(transfer_pairs) == 300
```

必须检查：

```python
assert selected_core.physical_task_id.nunique() == 2200
```

四个 role 的 `physical_task_id` 两两不重叠。

必须检查 source counts 与 Core v1.0 任务书一致。

如果任何一项不一致：

```text
STOP
```

不要继续搭 Agent。

输出：

```text
outputs/preflight/DATASET_AUDIT.json
```

内容至少：

```json
{
  "raw_count": 8046,
  "core_count": 2200,
  "roles": {
    "experience": 1500,
    "id_probe": 300,
    "transfer_probe": 300,
    "retention_probe": 100
  },
  "physical_overlap": 0,
  "status": "PASS"
}
```

---

# 6. Step 1：clone 并固定所有外部方法代码

脚本：

```text
scripts/01_clone_pin_external.sh
```

执行：

```bash
set -euo pipefail

mkdir -p external
cd external

# OpenETA
git clone https://github.com/OpenMOSS/OpenETA.git
cd OpenETA
git checkout 7d4a0a1522ba8ebbd362bde880bad81d2a98f15e
cd ..

# ACE
git clone https://github.com/ace-agent/ace.git
cd ace
git checkout 82709de050e1db6e6ef2f07bcb0393560b94992a
cd ..

# WorldMind
git clone https://github.com/zjunlp/WorldMind.git
cd WorldMind
git checkout 712b0fd53b4bd6a1948603f087a1e9f25a5adaea
cd ..

# EmbodiSkill
git clone https://github.com/air-embodied-brain/EmbodiSkill.git
cd EmbodiSkill
git checkout 760126030eab1d33ec6a6f30988f0f1fb58df3a7
cd ..
```

然后生成：

```text
manifests/external_sources.json
```

格式：

```json
{
  "OpenETA": {
    "repo": "https://github.com/OpenMOSS/OpenETA",
    "commit": "7d4a0a1522ba8ebbd362bde880bad81d2a98f15e"
  },
  "ACE": {
    "repo": "https://github.com/ace-agent/ace",
    "commit": "82709de050e1db6e6ef2f07bcb0393560b94992a"
  },
  "WorldMind": {
    "repo": "https://github.com/zjunlp/WorldMind",
    "commit": "712b0fd53b4bd6a1948603f087a1e9f25a5adaea"
  },
  "EmbodiSkill": {
    "repo": "https://github.com/air-embodied-brain/EmbodiSkill",
    "commit": "760126030eab1d33ec6a6f30988f0f1fb58df3a7"
  }
}
```

不要跟随最新 main 自动更新。

若后续升级 commit：

> 新开 benchmark version，不覆盖 Pilot v0.1。

---

# 7. Step 2：安装 OpenETA，但不要污染 simulator 环境

OpenETA 当前使用 Python 3.10+ 和 `uv`。

建议：

```bash
cd external/OpenETA
uv sync --extra dev
```

然后：

```bash
uv run python -c "import agent; print('OpenETA import OK')"
```

注意：

不同 simulator 很可能存在依赖冲突。

因此不要强行把：

```text
AI2-THOR
Habitat
ProcTHOR
OpenETA
WorldMind
EmbodiSkill
```

全部 pip 到一个 Python 环境。

推荐架构：

```text
OpenETA runtime process
        |
        | typed adapter / RPC / local worker protocol
        v
Simulator worker process
```

各 simulator 使用独立 env。

第一版允许 adapter worker 用 multiprocessing / socket / RPC，不要求全部 MCP 化。

但是对 OpenETA planner 暴露的 action semantics 必须稳定。

---

# 8. Step 3：配置并验证 DeepSeek 官方 `deepseek-flash` API

本 Pilot **不启动本地 vLLM/SGLang**。统一使用 DeepSeek 官方 OpenAI-compatible API。

固定配置：

```bash
export DEEPSEEK_API_KEY="<your-key>"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-flash"
```

**OpenETA 配置中的 `api_base` 必须写 `https://api.deepseek.com`，不要加 `/v1`。** Pinned OpenETA backend 自己会拼接 `/v1/chat/completions`，最终请求 URL 应为：

```text
https://api.deepseek.com/v1/chat/completions
```

同时给 OpenETA / ACE / WorldMind / EmbodiSkill 的 OpenAI-compatible client 统一映射为：

```text
api_key  = $DEEPSEEK_API_KEY
base_url = https://api.deepseek.com
model    = deepseek-flash
```

## 8.1 API identity smoke

先执行一个最小文本请求，必须记录：

```text
requested_model = deepseek-flash
returned_model  = API 实际返回的 model 字段（若有）
base_url        = https://api.deepseek.com
request_time_utc
```

`deepseek-flash` 是官方滚动别名。Pilot 阶段允许使用该别名，但 `provenance.json` 必须记录实际返回的 model/version 信息；论文最终大规模实验前再决定是否需要锁定不可漂移的具体版本。

## 8.2 Text smoke

发送简单文本指令，例如：

```text
Return exactly: DEEPSEEK_FLASH_TEXT_OK
```

必须满足：

```text
HTTP success
response parse success
model response non-empty
```

## 8.3 Vision smoke

从任一已经部署好的 simulator 随机抓一张 RGB 帧，通过 **同一个 `deepseek-flash` endpoint** 发送多模态请求，例如：

```text
Describe the visible scene in one short sentence.
```

必须满足：

```text
image accepted
response non-empty
no unsupported-modality error
```

DeepSeek 官方 Vision API 使用标准 OpenAI-compatible `user.content` block 数组。对本地 simulator PNG，优先用 base64 `data:image/png;base64,...`：

```python
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the visible scene in one short sentence."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}}
  ]
}
```

图片必须出现在 `user` 消息中；不要放进 `system` / `assistant` 消息。

如果 vision request 失败：

```text
STOP
```

不要继续跑 Agent。

## 8.4 OpenETA backend compatibility smoke

普通 DeepSeek API smoke 通过后，**还必须验证 OpenETA 自己的 backend 能调用 DeepSeek**。这是独立 Gate。

使用 OpenETA 的 `OpenAICompatiblePlannerBackend` 构造一个最小 planner request：

```text
1 个 system prompt
1 个 user observation
1 张 RGB image
2~3 个 dummy available tools/actions
```

要求：

```text
final URL == https://api.deepseek.com/v1/chat/completions
model == deepseek-flash
request contains image in user message
request does NOT contain chat_template_kwargs
HTTP success
OpenETA backend extracts non-empty final content
OpenETA planner parser returns a valid PlannerDecision or a clearly classifiable schema-validation retry
```

这里失败时，**先修 provider configuration / request adapter，不允许改 OpenETA planner prompt 或 planner logic。**

## 8.5 `02_check_deepseek_api.sh` 必须做什么

该脚本至少：

1. 检查 `DEEPSEEK_API_KEY` 是否存在；
2. 检查 `DEEPSEEK_BASE_URL=https://api.deepseek.com`；
3. 检查 `DEEPSEEK_MODEL=deepseek-flash`；
4. 跑 text smoke；
5. 跑 vision smoke；
6. 跑 OpenETA backend compatibility smoke；
7. 审计 outgoing request 不含 `chat_template_kwargs`；
8. 把 resolved endpoint/model、thinking 行为和 smoke 结果写入 `outputs/api_smoke/deepseek_flash.json`；
9. 任一步失败则退出码非 0。

**禁止**在第一轮为了兼容某个 RSI 方法，给 updater 换成其它供应商或模型。

# 9. Step 4：冻结 OpenETA

创建：

```text
benchmark/openeta_bridge/freeze.py
```

## 9.1 Critical file hash

至少记录以下 upstream 文件 SHA256：

```text
agent/runtime/planner.py
agent/runtime/episode.py
agent/runtime/runtime.py
agent/runtime/planner_prompts.py
agent/runtime/actions.py
agent/runtime/pipeline.py
agent/backends/planner.py
agent/runtime/self_improvement.py
```

运行前：

```bash
cd external/OpenETA
git status --porcelain
```

必须为空。

然后：

```bash
git diff --exit-code 7d4a0a1522ba8ebbd362bde880bad81d2a98f15e
```

必须无 diff。

## 9.2 禁用 native self-improvement

在：

```text
benchmark/openeta_bridge/build_runtime.py
```

组装完 upstream runtime 后：

```python
from dataclasses import replace

reviewer = runtime.self_improvement_reviewer
reviewer.config = replace(
    reviewer.config,
    enabled=False,
    auto_apply_reviewed=False,
)
reviewer.auto_applier = None
```

并执行 assertions。

## 9.3 禁用与本 benchmark 无关的额外工具

OpenETA 原生包含很多机器人/网络/代码工具。

本 benchmark canonical ToolRegistry **只能包含 source benchmark 正式允许的公开动作能力**。

禁止向模型开放：

```text
web_search
web_fetch
python_exec
SAM3
AnyGrasp
AnyPlace
hidden geometry query
raw simulator metadata query
object coordinate query
private task-state query
```

除非某 source benchmark 原协议明确把等价信息暴露给 Agent。

当前 Core v1.0 默认全部不开放这些额外能力。

## 9.4 固定 planner prompt

记录：

```text
planner.system_prompt SHA256
```

所有方法必须完全相同。

RSI guidance 不允许通过改 system prompt 注入。

只能通过统一的 dynamic context slot 注入。

## 9.5 固定 planner context config

第一轮建议：

```yaml
recent_conversation_action_groups: 4
recent_transition_observations: 3
max_selected_skills: 3
max_skill_content_chars: 8000
auto_compact_enabled: true
```

如果 upstream default 与上述一致，直接使用 upstream default。

若需要改变：

- 所有方法同时改变；
- 写入 resolved config；
- 不因某方法单独放宽。

## 9.6 生成 Freeze Manifest

```text
manifests/openeta_freeze_manifest.json
```

至少：

```json
{
  "repo_commit": "...",
  "critical_file_sha256": {},
  "planner_prompt_sha256": "...",
  "native_self_improvement": false,
  "native_auto_apply": false,
  "native_iterate_allowed": false,
  "cross_session_openeta_memory": false,
  "enabled_tool_policy": "benchmark_source_actions_only",
  "web_tools_enabled": false,
  "python_exec_enabled": false
}
```

---

# 10. Step 5：统一 Environment Adapter 接口

创建：

```text
benchmark/adapters/base.py
```

定义：

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class PublicObservation:
    instruction: str
    images: list[Any]
    text_feedback: str
    public_metadata: dict

@dataclass
class StepOutcome:
    observation: PublicObservation
    action_success: bool | None
    reward_public: float | None
    terminated: bool
    truncated: bool
    public_feedback: str

class BenchmarkEnvAdapter(ABC):
    @abstractmethod
    def reset(self, task_record) -> PublicObservation:
        ...

    @abstractmethod
    def build_tool_specs(self, task_record):
        ...

    @abstractmethod
    def step(self, action_name: str, parameters: dict) -> StepOutcome:
        ...

    @abstractmethod
    def private_evaluate(self):
        ...

    @abstractmethod
    def close(self):
        ...
```

---

# 11. Adapter 的公平性规则

## 11.1 Agent 可以看到

只允许：

- task instruction；
- 第一人称 RGB；
- TVR 中公开 target image；
- 官方允许的 action result / failure text；
- 官方公开的 valid-action 信息（仅当原 benchmark 本身提供）；
- 本 episode 历史；
- RSI 注入的过去 experience/skill/context。

## 11.2 Agent 不可以看到

禁止：

```text
golden_actions
goal predicates
success_conditions
private target pose
private object coordinates
shortest path
oracle subgoals
ground-truth scene graph
simulator hidden state
physical_task_id
ID/Transfer role
pair id
benchmark skill-family label
```

特别注意：

本项目 registry 中有：

```text
skill_family
primitive_set
target_object_classes
```

这些字段可以用于：

- benchmark stratification；
- pair construction；
- analysis。

但如果原 task 不向 Agent 提供这些字段：

> **绝对不要把这些 taxonomy label 注入 prompt。**

否则造成 benchmark-created privileged information leakage。

---

# 12. 五个 source 的 adapter 目标

## 12.1 EB-ALFRED

保持官方：

- scene/reset；
- egocentric RGB；
- high-level embodied action semantics；
- environment feedback。

1 个 OpenETA world-mutating tool call 必须映射到 1 个 source action。

不要把 golden trajectory 传给 OpenETA。

---

## 12.2 EB-Habitat

保持官方 Habitat episode：

- episode scene；
- start state；
- visible RGB；
- official rearrangement action interface。

禁止公开：

```text
goal_preds
start_preds
sampled hidden world state
oracle subgoal sequence
```

除非其中某字段本来就是公开 task input。

---

## 12.3 SpatialWorld

保持统一 source action：

```text
Move
Rotate
Tilt
ChangePosture
Pick
Place
ChangeState
Manipulate
EndTask
```

如果某 task action subset 更小，按 source 原协议暴露。

不要给 golden actions。

---

## 12.4 AsgardBench

保持 AI2-THOR 在线交互。

原 plan.json 中 evaluator 可使用的 goal/setup 信息不要进入 Agent prompt。

执行器只接收：

- instruction；
- RGB；
- 正式 action feedback。

---

## 12.5 TVRBench

TVR 是特例。

公开输入：

```text
current RGB
+
target RGB
```

因此 target image 必须给模型。

但：

```text
target pose
```

只能给 evaluator / simulator reset logic，不能给 Agent。

OpenETA tools 原样对应：

```text
MoveAhead
MoveBack
MoveLeft
MoveRight
RotateLeft
RotateRight
LookUp
LookDown
Stop
```

---

# 13. Step 6：Adapter Leakage Audit

每个 model request 必须保存一个 redacted copy：

```text
public_context_dump.jsonl
```

测试：

```text
tests/test_no_private_leakage.py
```

结构化 assert：

```python
FORBIDDEN_KEYS = {
    "golden_actions",
    "goal_preds",
    "success_conditions",
    "target_pose",
    "physical_task_id",
    "transfer_pair_id",
    "retention_anchor_id",
}
```

注意：TVR 的 `target image` 合法；`target pose` 不合法。

如果任何 private field 到达 model request：

```text
FAIL IMMEDIATELY
```

---

# 14. Step 7：定义统一 RSI Plugin 接口

创建：

```text
benchmark/rsi/base.py
```

建议接口：

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class RSIInjection:
    context_text: str = ""
    dynamic_skills: list = field(default_factory=list)
    provenance_ids: list[str] = field(default_factory=list)
    estimated_tokens: int = 0

class RSIMethod(ABC):
    name: str

    @abstractmethod
    def init_run(self, run_ctx):
        ...

    @abstractmethod
    def before_episode(self, task_public_view) -> RSIInjection:
        ...

    def after_step(self, step_public_record):
        return None

    @abstractmethod
    def after_episode(self, trajectory_public, outcome_public):
        ...

    @abstractmethod
    def snapshot(self, output_dir):
        ...

    @abstractmethod
    def load_snapshot(self, input_dir):
        ...

    @abstractmethod
    def state_hash(self) -> str:
        ...

    def set_update_enabled(self, enabled: bool):
        self.update_enabled = enabled
```

硬规则：

`RSIMethod` 不能持有 simulator object。

它只能读取：

```text
public task
public trajectory
public feedback
success/failure result
```

它不能：

- 调环境；
- 修改环境；
- 修改 tool registry；
- 修改 evaluator；
- 修改 OpenETA core；
- 读取 private metadata。

---

# 15. 统一 RSI Context Injection

RSI 方法必须使用同一个 injection channel。

不要：

```text
ACE 改 system prompt
WorldMind 改 user prompt
EmbodiSkill 改 tool description
```

这样不公平。

创建：

```text
benchmark/openeta_bridge/context_injection.py
```

统一在每个 episode 开始时产生：

```text
[PERSISTENT EXPERIENCE GUIDANCE]
...
[/PERSISTENT EXPERIENCE GUIDANCE]
```

将其作为同一位置的 task-level context / session seed guidance 注入。

要求：

- base planner system prompt 不变；
- tool schema 不变；
- 当前 task instruction 不变；
- RSI guidance 单独可审计；
- 保存 injection 原文和 token count。

---

# 16. RSI Context Budget

第一轮统一：

```yaml
max_rsi_injection_tokens: 8192
```

即：

> 任意方法在单个 episode 开始时注入 OpenETA planner 的持久知识，不得超过 8192 tokens。

如果 method 内部状态更大：

- 可以存；
- 可以 retrieval；
- 只能选择 <=8192 tokens 的内容注入。

禁止 silent truncation。

必须由 method 显式选择内容后，再做 token assert：

```python
assert injection_tokens <= 8192
```

注：8192 是 Pilot 协议，不是论文最终不可修改的常数。若 Pilot 显示明显不合理，Full v1.0 可以统一调整，但必须重新版本化。

---

# 17. Condition A：None 实现

```text
benchmark/rsi/none.py
```

行为：

```python
before_episode(...) -> RSIInjection()
after_step(...) -> None
after_episode(...) -> None
state_hash() -> hash_of_constant_empty_state
```

无文件写入。

它是 zero-persistence baseline。

---

# 18. Condition B：Raw Episodic Memory

```text
benchmark/rsi/raw_memory.py
```

## 18.1 保存内容

Experience episode 完成后保存：

```json
{
  "episode_id": "...",
  "instruction": "...",
  "source_dataset": "...",
  "actions": [
    {
      "action": "...",
      "public_feedback": "..."
    }
  ],
  "success": true,
  "terminated": true,
  "num_steps": 12
}
```

不要保存给未来 Agent 的：

- private goal；
- hidden state；
- golden action；
- pair label；
- skill family label。

## 18.2 Retrieval

第一版推荐 deterministic retrieval：

```text
instruction text embedding / BM25
```

为了减少额外模型调用，优先实现：

```text
BM25 + sentence embedding optional
```

最简单可先：

```text
top_k = 4
```

按当前 task instruction 检索。

不要按 benchmark 提供的 hidden skill-family 直接 oracle retrieve。

## 18.3 Injection

格式：

```text
Previous relevant experiences:

[Episode 1]
Task: ...
Outcome: Success/Failure
Trajectory summary:
- action ... -> feedback ...
...
```

不生成 LLM reflection。

如果 raw trajectory 超 budget：

- deterministic truncate；
- 优先保留 task、outcome、关键最近 action/feedback；
- 记录 truncation。

---

# 19. Condition C：ACE-Context 接 OpenETA

```text
benchmark/rsi/ace_context.py
```

## 19.1 不允许做什么

不要让 ACE Generator 自己重新执行任务。

否则：

```text
OpenETA Agent
+
ACE Generator Agent
```

变成两个不同 executor，失去公平性。

Canonical adaptation：

```text
OpenETA = executor / trajectory generator
ACE = reflector + curator + playbook manager
```

## 19.2 初始 Playbook

使用 ACE 官方空 playbook headings，但其中没有实质知识。

例如：

```text
## STRATEGIES & INSIGHTS

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## OTHERS
```

不要人工预填 embodied knowledge。

## 19.3 before_episode

从当前 ACE playbook 选择可注入内容。

Pilot 先允许整个 playbook，只要：

```text
<=8192 tokens
```

超过后：

- 使用 ACE 自己已有 helpful/harmful counters；
- 或 deterministic relevance filter；
- 不使用 private task taxonomy。

将结果送进统一 `RSIInjection.context_text`。

## 19.4 after_episode

把 OpenETA episode 转成 ACE reflection evidence：

```text
Task instruction
OpenETA action/reasoning trace（若公开记录）
Official visible feedback
Final success/failure
```

不要提供：

```text
golden solution
private evaluator state
```

ACE Reflector：

- 使用官方 Reflector 实现；
- 分析哪些 playbook bullet 有帮助/有害；
- 提取新的 lesson。

ACE Curator：

- 使用官方 Curator；
- 使用官方 delta update semantics；
- 更新 playbook；
- 维护 helpful/harmful counts。

## 19.5 Updater model

第一轮：

```text
reflector_model = deepseek-flash
curator_model = deepseek-flash
```

若 ACE 官方 provider wrapper 不直接支持 DeepSeek 官方 OpenAI-compatible endpoint：

- 写最薄的 OpenAI-compatible client adapter；
- 不改 ACE Reflector/Curator prompt/logic。

## 19.6 日志

必须保存：

```text
playbook_before.txt
reflector_output.json
curator_output.json
playbook_after.txt
curator_ops.jsonl
```

每次 update 都要记录 diff。

---

# 20. Condition D：WorldMind 接 OpenETA

```text
benchmark/rsi/worldmind.py
```

优先直接 import：

```python
from worldmind_plugin import (
    WorldMindConfig,
    ProcessExperienceModule,
    GoalExperienceModule,
    ExperienceRetrievalModule,
    ProcessTrajectoryStep,
    GoalTrajectoryStep,
)
```

## 20.1 before_episode

调用：

```python
result = retrieval_module.retrieve(
    task_instruction=current_instruction,
    enable_refine=True,
)
```

获取：

```python
result["formatted_prompt"]
```

放入统一：

```text
RSIInjection.context_text
```

assert <=8192 tokens。

默认建议：

```yaml
goal_experience_top_k: 3
process_experience_top_k: 5
enable_experience_refine: true
```

Pilot 不单独调参找最好值。

## 20.2 WorldMind 的 predicted_state 问题

官方 ProcessExperience 需要：

```text
observation
action
predicted_state
env_feedback
```

OpenETA 默认 planner 输出的是动作，并不会稳定输出一个可直接用于 WorldMind 的 future-state prediction。

**不要修改 OpenETA planner prompt 让它额外预测状态。**

这会改变 canonical Agent。

正确做法：

在 OpenETA 已经选定 action 之后、env.step 之前，增加一个独立 sidecar prediction call：

输入：

```text
current public observation
chosen action
```

输出：

```text
predicted abstract next state
```

要求：

1. sidecar prediction 发生在 action 已锁定后；
2. sidecar 输出绝不能返回给 OpenETA planner；
3. sidecar 只服务 WorldMind process experience；
4. 使用同一个 `deepseek-flash`；
5. sidecar token 单独计入 RSI cost；
6. 不使用 private environment state。

流程：

```text
OpenETA selects action A_t
        |
        +--> WorldMind sidecar predicts state_hat_{t+1}
        |
        v
env.step(A_t)
        |
        v
public observation / feedback_{t+1}
        |
        v
WorldMind ProcessExperienceModule
```

## 20.3 after_step

构造官方：

```python
ProcessTrajectoryStep(
    observation=public_state_summary_before,
    action=action_text,
    predicted_state=predicted_state,
    env_feedback=public_feedback_after,
)
```

调用：

```python
process_module.process_single_step(...)
```

如果需要图像：

- 使用 WorldMind plugin 官方 `before_image/after_image` 路径；
- 只能使用 Agent 同时可见的 RGB；
- 不传 segmentation/hidden state。

## 20.4 after_episode

若 Experience episode 成功：

```python
goal_module.extract_experience(...)
```

然后：

```python
retrieval_module.reload_experiences()
```

如果失败：

- 不生成 Goal Experience；
- Process Experience 仍可来自 prediction error。

## 20.5 模型

全部 WorldMind component：

```text
deepseek-flash
```

包括：

```text
discriminator
summarizer
reflector
extractor
refiner
```

---

# 21. Condition E：EmbodiSkill 接 OpenETA

```text
benchmark/rsi/embodiskill.py
```

## 21.1 原则

必须复用官方：

```text
agentkit/skill/embodiskill_skill/EmbodiSkill.py
agentkit/skill/embodiskill_skill/prompt.py
agentkit/skill/embodiskill_skill/skill_base.py
```

不要只复制论文思想。

## 21.2 先做 call-path audit

coding agent 第一件事不是写 adapter，而是阅读：

```text
external/EmbodiSkill/tasks/run_epochs.py
external/EmbodiSkill/agentkit/skill/embodiskill_skill/EmbodiSkill.py
external/EmbodiSkill/agentkit/skill/common.py
```

输出：

```text
outputs/preflight/EMBODISKILL_CALL_PATH.md
```

必须写清：

- task 如何初始化；
- StateChain 如何构造；
- trajectory 如何送进 skill module；
- reflection 在哪里触发；
- manual_state 在哪里更新；
- skill retrieval 在哪里发生；
- 哪些字段是 ALFWorld-specific。

在此文档完成前：

> 禁止开始重写 EmbodiSkill。

## 21.3 Adapter 目标

把 OpenETA episode 映射成 EmbodiSkill 官方接受的数据对象。

至少保留：

```text
instruction
agent action
public observation summary
public feedback
success/failure
trajectory order
```

不允许加入 oracle label。

## 21.4 before_episode

从 EmbodiSkill current manual / skill representation 中获取当前 task 的指导内容。

将其放入：

```text
RSIInjection.context_text
```

或者映射为 OpenETA 的 dynamic selected skill guidance，但必须保证：

- 其它方法也有等价 8192-token context budget；
- 不修改 OpenETA built-in skill files；
- dynamic skills 保存在本 run `rsi_state/` 下。

Pilot 为了公平，**优先统一转成 context_text**，不要先利用 OpenETA SkillRegistry 特权通路。

## 21.5 after_episode

调用官方 EmbodiSkill reflection/update 逻辑。

预期产生：

```text
manual_state
execution notes
reflections
manual versions
```

必须保留官方分类思想：

- new knowledge；
- optimization evidence；
- skill defect；
- execution lapse。

不要人工简化成：

```text
失败 -> rewrite skill
```

## 21.6 updater

Primary Self-Evolution：

```text
SKILL_MODEL=deepseek-flash
```

禁止第一轮使用 GPT-5.2 updater。

## 21.7 Stop condition

如果为了支持 OpenETA trajectory，必须：

- 大改官方 reflection prompts；
- 大改 skill update algorithm；
- 删除核心 graph/manual semantics；
- 或核心 upstream 算法修改量 > 约 200 行；

则：

```text
EmbodiSkill integration = BLOCKED
```

记录原因，不造假复现。

---

# 22. 是否测试 OpenETA 自带 Self-Improvement

第一轮主表不需要。

但可以加一个参考条件：

```text
openeta_native_si
```

目的只是回答：

> OpenETA 自己内置的 post-episode skill review，在我们的统一 benchmark 上有没有作用。

若做：

- 与外部 RSI 分开标记；
- 不把它作为 `none`；
- 使用 upstream native mechanism；
- 记录其权限与其它方法不同。

Pilot 优先级：

```text
P2 optional
```

不要阻塞主 5 条线。

---

# 23. 第一轮 Pilot-150，而不是直接 Full-2200

Full Core：

```text
2200 tasks
```

第一轮先构建：

```text
Pilot-150
```

建议组成：

```text
Experience       75
ID Probe         25
Transfer Probe   30
Retention Probe  20
-------------------
Total            150
```

注意这里 150 指 role-level task set，不代表实际总 simulator rollouts；多 checkpoint probe 会重复评测 probe tasks。

---

# 24. Pilot-150 选样规则

脚本：

```text
scripts/05_build_pilot150.py
```

固定 seed：

```text
20260915
```

排序：

```text
SHA256("20260915|pilot150|" + global_task_id)
```

## 24.1 先选 pairs，再补 Experience

不要先独立抽 75 Experience，然后发现 probe 的 anchor 不在里面。

步骤：

1. 从 `id_pairs.parquet` 选 Pilot ID pairs；
2. 从 `transfer_pairs.parquet` 选 Pilot Transfer pairs；
3. 把它们需要的 experience anchors 放进 Pilot Experience；
4. 从 retention schedule 选 retention anchors；
5. 把 retention anchors 对应的 Experience 放入 Pilot Experience；
6. 再 deterministic fill 到 75 Experience。

## 24.2 Transfer target 建议 quota

因为 Core transfer 本身没有 Asgard quota：

```text
TVRBench      10
SpatialWorld  10
EB-ALFRED      5
EB-Habitat     5
----------------
Total         30
```

## 24.3 Experience coverage

尽量保证五个 source 都进入 Experience。

目标约：

```text
15 tasks / source
```

如果 pair constraints 导致偏差，不要强行破坏 pair；记录最终 source distribution。

## 24.4 Assert

Pilot roles 仍然必须：

```text
physical_task_id pairwise disjoint
```

输出：

```text
manifests/pilot150.json
outputs/preflight/PILOT150_AUDIT.json
```

---

# 25. Pilot 的 Experience Stream 顺序

第一版：

- 同一条 deterministic sequence；
- 所有 RSI 方法完全一致；
- 不能为不同方法 shuffle 不同顺序。

建议对 75 个 Experience 按：

```text
SHA256("20260915|experience_stream|" + global_task_id)
```

排序。

未来 full benchmark 应增加多 curriculum / counterbalanced order。

Pilot 不需要。

---

# 26. Snapshot / Probe 时间点

Experience 75 条，保存：

```text
S000
S025
S050
S075
```

其中：

- S000：任何 Experience 之前；
- S075：完成全部 Pilot Experience 后。

最少完整 probe：

```text
S000
S075
```

可选轻量 intermediate probe：

```text
S025
S050
```

如果 simulator 太慢，Pilot 第一轮可以：

- S000：25 ID + 30 Transfer + 20 Retention；
- S075：同样全部；
- S025/S050：只跑固定 10-task mini-probe。

无论哪种方式：

> 所有方法必须完全一致。

---

# 27. Frozen Probe Runner

```text
benchmark/runner/frozen_probe.py
```

伪代码：

```python
def run_frozen_probe(method, snapshot_path, probe_tasks):
    clone = method.clone_from_snapshot(snapshot_path)
    clone.set_update_enabled(False)

    h0 = clone.state_hash()

    results = []
    for task in probe_tasks:
        results.append(run_episode(task, rsi=clone, probe=True))

    h1 = clone.state_hash()
    assert h0 == h1

    return results
```

还要 assert：

```text
no rsi update event during probe
no write under rsi_state during probe
```

---

# 28. OpenETA 每个 episode 的运行流程

Canonical flow：

```text
1. Load task public view
2. RSI.before_episode()
3. Validate injection budget
4. Create a FRESH OpenETA session
5. Seed RSI guidance into the common context slot
6. env.reset(task)
7. OpenETA closed-loop execution
8. Record full public trajectory
9. private evaluator computes task outcome
10. if experience:
       RSI.after_episode(...)
    if probe:
       NO UPDATE
11. close environment
12. close/discard OpenETA session
```

注意：

RSI 的状态活在：

```text
run/rsi_state/
```

不是 OpenETA session memory。

---

# 29. 每一步必须记录什么

统一 trajectory schema：

```json
{
  "episode_id": "...",
  "global_task_id": "...",
  "role": "experience",
  "source_dataset": "...",
  "step_idx": 3,
  "public_instruction": "...",
  "public_observation_refs": ["...png"],
  "planner_request_hash": "...",
  "planner_response": {},
  "chosen_action": {},
  "public_env_feedback": "...",
  "action_success": true,
  "terminated": false,
  "truncated": false,
  "planner_usage": {
    "input_tokens": 0,
    "output_tokens": 0
  },
  "rsi_aux_usage": {}
}
```

private evaluator data：

```text
必须分文件保存
```

例如：

```text
private_eval.json
```

不要混进 planner-visible trajectory serializer。

---

# 30. RSI 更新轨迹也必须记录

这是这个 benchmark 的重要证据。

每次 update 保存：

```json
{
  "update_id": "...",
  "method": "ace_context",
  "source_episode_id": "...",
  "state_hash_before": "...",
  "state_hash_after": "...",
  "input_evidence_refs": [],
  "llm_calls": [],
  "added": [],
  "modified": [],
  "removed": [],
  "update_tokens": 0,
  "update_wall_time_s": 0.0
}
```

最终论文分析时必须能够追踪：

```text
failure
  -> reflection/update
  -> state change
  -> later retrieval/use
  -> behavior change
  -> success/failure
```

否则只有 final SR，解释力太弱。

---

# 31. 模型/采样固定

第一轮建议：

```yaml
planner:
  provider: openai_compatible
  base_url: https://api.deepseek.com
  model: deepseek-flash
  enable_thinking: null   # MUST stay unset for pinned OpenETA; see §3.2
  temperature: 0.0        # ignored by DeepSeek while thinking is enabled
  top_p: 1.0
  max_output_tokens: 8192

rsi_updater:
  provider: openai_compatible
  base_url: https://api.deepseek.com
  model: deepseek-flash
  thinking: default_enabled
  temperature: 0.0        # do not claim determinism from this setting
  top_p: 1.0
  max_output_tokens: 8192
```

如果某 RSI upstream wrapper 无法原样传递 DeepSeek 参数：

- 不修改其核心算法；
- 只写最薄的 DeepSeek OpenAI-compatible client shim；
- 所有方法使用同一 DeepSeek endpoint/model/thinking policy；
- 记录 resolved request parameters。

不要让：

```text
ACE temp=0.7
WorldMind temp=0
EmbodiSkill temp=1
```

这种差异混入主结果。

---

# 32. Environment / Agent Budget 固定

同一 source dataset：

所有 RSI condition 使用同样：

- max environment steps；
- max OpenETA turns；
- max tool calls；
- simulator timeout；
- planner token cap；
- current RGB resolution；
- camera FOV；
- source action schema。

预算从 source benchmark 原协议优先继承。

如果统一 benchmark 必须修改：

- 统一修改；
- 写入 config；
- 不为某个 RSI 特判。

---

# 33. Cost accounting

每个 episode 分开记录：

```text
planner_input_tokens
planner_output_tokens
rsi_injection_tokens
rsi_update_input_tokens
rsi_update_output_tokens
worldmind_prediction_sidecar_tokens
embedding_calls
simulator_steps
wall_clock_seconds
```

不要只报 total tokens。

后面需要算：

\[
\text{Evolution Gain per 1M Updater Tokens}
\]

\[
\text{Evolution Gain per 1k Environment Steps}
\]

因为 TF-GRPO / WorldMind 等方法额外计算量不同。

---

# 34. OpenETA Freeze Test

```text
tests/test_openeta_frozen.py
```

至少：

```python
assert repo_head == PINNED_OPENETA_COMMIT
assert git_worktree_clean
assert critical_hashes_match
assert native_self_improvement_enabled is False
assert auto_applier is None
assert native_iterate_not_used
assert planner_prompt_hash == expected_hash
assert web_tools_disabled
assert python_exec_disabled
```

每个 run 开始和结束各执行一次。

---

# 35. Cross-Episode State Isolation Test

```text
tests/test_rsi_state_isolation.py
```

运行两个 dummy episodes：

- episode 1 写 OpenETA working memory；
- episode 2 新 session。

assert：

```text
episode1 working memory 不出现在 episode2
```

然后给 `raw_memory` 开启 RSI：

assert：

```text
只有 RSIInjection 能把过去 experience 带入 episode2
```

---

# 36. Snapshot Reload Test

```text
tests/test_snapshot_reload.py
```

对每个 RSI：

1. experience 3 episodes；
2. snapshot；
3. 保存 `state_hash_A`；
4. 新进程 load snapshot；
5. 得到 `state_hash_B`；
6. assert A == B；
7. 对同一个 dummy query retrieval；
8. injection content 必须相同。

不通过不能跑 Pilot。

---

# 37. Adapter Smoke Gate

脚本：

```text
scripts/04_validate_adapters.py
```

每个 source 至少抽：

```text
20 tasks
```

检查：

```text
reset success
RGB exists
RGB non-empty
one legal action can execute
new observation returns
termination works
private evaluator runs
close works
```

总计至少：

```text
100 task adapter smoke
```

允许 task 本身失败，不允许 infrastructure crash。

若任一 source：

```text
adapter crash rate > 2%
```

则 STOP。

---

# 38. OpenETA Baseline Smoke Gate

```text
scripts/06_smoke_openeta.py
```

先只跑：

```text
none
```

建议：

```text
5 tasks/source = 25 tasks
```

检查：

- VLM 能看到 RGB；
- OpenETA 能生成合法 tool call；
- action 被环境接受；
- fresh observation 确实返回；
- trajectory 完整；
- private evaluator 工作；
- OpenETA native self-improvement 没有触发；
- 下一 episode 没有隐式跨 session memory。

不要要求很高成功率。

这一 Gate 是 pipeline gate，不是能力 gate。

如果：

```text
>20% action 都是 schema invalid
```

先修 action adapter/schema，再跑 RSI。

不要用 RSI 去“修”坏掉的 agent integration。

---

# 39. RSI Smoke Gate

```text
scripts/07_smoke_rsi.py
```

每个方法：

```text
5 Experience
+ 3 frozen Probe
```

检查：

### none

```text
state unchanged
```

### raw_memory

```text
episode count = 5
retrieval non-empty after first relevant experience
```

### ACE

```text
playbook changes after valid evidence
all changes traceable to curator
```

### WorldMind

```text
goal/process experience files generated
retrieval works
prediction sidecar never affects chosen action
```

### EmbodiSkill

```text
official state/manual files evolve
no upstream algorithm core edited
```

所有方法：

```text
probe state hash before == after
```

否则 FAIL。

---

# 40. 第一轮 Pilot 实验矩阵

主模型：

```text
deepseek-flash
```

seed：

```text
20260915
```

Conditions：

| ID | Agent | RSI | Persistent State | Primary? |
|---|---|---|---|---|
| C0 | OpenETA-Frozen | None | 无 | 是 |
| C1 | OpenETA-Frozen | Raw Episodic Memory | episode store | 是 |
| C2 | OpenETA-Frozen | ACE-Context | playbook | 是 |
| C3 | OpenETA-Frozen | WorldMind | goal/process experience | 是 |
| C4 | OpenETA-Frozen | EmbodiSkill | skill/manual | 是 |

Optional：

| C5 | OpenETA | OpenETA Native SI | native skills/playbooks | 否，参考 |

Pilot 先不跑：

```text
TF-GRPO
PRACTICE
AHE
SHAPER
RL/SFT
```

---

# 41. Pilot Runner

```text
scripts/08_run_pilot150.py
```

每个 condition：

## 41.1 S000

保存空/初始 RSI state。

运行：

```text
ID Probe 25
Transfer Probe 30
Retention Probe 20
```

全部 frozen。

得到：

```text
initial capability
```

## 41.2 Experience 1-25

允许 update。

snapshot：

```text
S025
```

## 41.3 Experience 26-50

snapshot：

```text
S050
```

## 41.4 Experience 51-75

snapshot：

```text
S075
```

## 41.5 S075 final probe

重新跑同一组：

```text
ID 25
Transfer 30
Retention 20
```

updates disabled。

---

# 42. Probe 为什么可以重复同样任务

因为 probe 本身：

- 不更新；
- state clone 后 discard；
- 用来测同一个 checkpoint 前后能力变化。

因此：

```text
S000 vs S075
```

使用同一 probe 是合理的。

但是不能让 probe trajectory 进入 later experience state。

---

# 43. Pilot 指标

第一轮至少计算：

## 43.1 Initial Success Rate

\[
P_0
\]

## 43.2 Final Success Rate

\[
P_{75}
\]

## 43.3 Evolution Gain

\[
\Delta P = P_{75} - P_0
\]

## 43.4 ID Gain

只看 ID Probe。

## 43.5 Transfer Gain

只看 Transfer Probe。

这是本 benchmark 最重要指标之一。

## 43.6 Retention

比较 retention anchors 在不同 checkpoint 的表现。

至少报告：

```text
retention success
forgetting events
```

## 43.7 Source-wise

分别：

```text
EB-ALFRED
EB-Habitat
SpatialWorld
Asgard
TVR
```

## 43.8 Skill-family analysis

只用于 evaluator/offline analysis。

不要把 skill-family 给 Agent。

## 43.9 Agent execution quality

- invalid action rate；
- truncation rate；
- average steps；
- environment error rate。

## 43.10 Evolution cost

- updater tokens；
- total model tokens；
- env steps；
- state size；
- wall time。

---

# 44. 关键区分：Agent Failure vs RSI Failure

结果分析必须区分：

## Agent Integration Failure

例如：

- action schema invalid；
- RGB 没传进去；
- adapter crash；
- simulator reset 错；
- tool call parser 错。

这些不能算 RSI 失败。

## RSI Failure

例如：

- memory retrieved 但无帮助；
- ACE playbook 被错误更新；
- WorldMind 学到错误 causal rule；
- EmbodiSkill 把 execution lapse 错判成 skill defect；
- persistent state 越学越坏；
- transfer negative。

论文真正关心第二类。

---

# 45. 强制记录“经验是否真正被使用”

对每个 probe episode，保存：

```text
retrieved RSI entries
injected text
planner request containing them
planner action sequence
outcome
```

后续需要做 pathway audit：

```text
Experience E_i
  ↓
Update U_j
  ↓
Persistent item K_j
  ↓
Retrieved in probe P_k
  ↓
Appears in planner context
  ↓
Behavioral consequence
  ↓
Outcome
```

如果一个方法 state 长得很大，但从未被 planner retrieval/use：

> 不算有效 self-improvement。

---

# 46. 防止“方法靠更多 token 赢”

Pilot 暂时不强行 compute-match 所有方法，因为首先要验证 viability。

但是：

必须记录所有 token。

如果 C3 WorldMind 比 C0 多用了 10 倍模型调用，结果 +2pp：

不能直接宣称更强。

Pilot 报告里必须有：

```text
raw gain
cost-adjusted gain
```

Full benchmark 再加：

```text
compute-matched TTS control
```

---

# 47. 输出目录规范

每个 condition：

```text
outputs/pilot150/<method>/seed_20260915/
├── config_resolved.yaml
├── provenance.json
├── openeta_freeze_manifest.json
├── external_sources.json
├── run.log
│
├── states/
│   ├── S000/
│   ├── S025/
│   ├── S050/
│   └── S075/
│
├── experience/
│   ├── episode_0001/
│   ├── episode_0002/
│   └── ...
│
├── probes/
│   ├── S000/
│   │   ├── id/
│   │   ├── transfer/
│   │   └── retention/
│   └── S075/
│       ├── id/
│       ├── transfer/
│       └── retention/
│
├── rsi_state/
├── rsi_updates.jsonl
├── token_usage.json
├── metrics.json
├── LEAKAGE_AUDIT.json
├── FREEZE_AUDIT.json
└── FINAL_STATUS.txt
```

---

# 48. provenance.json

至少：

```json
{
  "benchmark_version": "Core-v1.0",
  "pilot_version": "Pilot-150-v0.1",
  "seed": 20260915,
  "model": "deepseek-flash",
  "model_provider": "DeepSeek official API",
  "model_base_url": "https://api.deepseek.com",
  "resolved_model": "record API-returned model/version if available",
  "thinking_policy": "DeepSeek default enabled; OpenETA enable_thinking left unset",
  "openeta_commit": "7d4a0a1522ba8ebbd362bde880bad81d2a98f15e",
  "rsi_method": "worldmind",
  "rsi_repo_commit": "...",
  "dataset_manifest_sha256": "...",
  "pilot_manifest_sha256": "...",
  "planner_prompt_sha256": "...",
  "tool_schema_sha256": "...",
  "start_time": "..."
}
```

---

# 49. Gate 总表

## G0 — Dataset Gate

必须：

```text
8046 raw
2200 Core
1500/300/300/100
no physical overlap
```

否则 STOP。

---

## G1 — OpenETA Freeze Gate

必须：

```text
pinned commit
clean git tree
critical hashes match
native self-improvement disabled
native iterate disabled
planner prompt hash fixed
extra tools disabled
```

否则 STOP。

---

## G2 — Model Gate

必须：

```text
DeepSeek official endpoint reachable
requested model == deepseek-flash
text request PASS
vision request PASS
OpenETA backend compatibility smoke PASS
outgoing request has no chat_template_kwargs
OpenAI-compatible request/response parsing PASS
```

否则 STOP。

---

## G3 — Adapter Gate

至少：

```text
20 tasks/source
100 tasks total
```

要求：

```text
crash rate <= 2%
private leakage = 0
```

否则 STOP。

---

## G4 — OpenETA Baseline Gate

25 task smoke。

要求：

- closed loop 正常；
- RGB 正常；
- action schema 大体正常；
- fresh observation 正常；
- no hidden memory；
- no native SI。

如果 schema invalid action >20%：

STOP，先修 adapter。

---

## G5 — RSI Smoke Gate

每方法：

```text
5 experience + 3 frozen probes
```

要求：

```text
experience phase state can change
probe phase state cannot change
```

---

## G6 — Snapshot Gate

所有方法：

```text
snapshot hash == reload hash
```

---

## G7 — Pilot Gate

要求：

```text
all 5 primary conditions complete
infrastructure crash rate <2%
probe mutation = 0
private leakage = 0
freeze violations = 0
```

注意：

> 不设置 success-rate GO/NO-GO 阈值。

方法可能确实失败，这正是 benchmark 要发现的。

---

# 50. `scripts/09_analyze_pilot.py` 输出

生成：

```text
outputs/pilot150/PILOT_REPORT.md
outputs/pilot150/PILOT_RESULTS.csv
outputs/pilot150/PILOT_RESULTS_BY_SOURCE.csv
outputs/pilot150/PILOT_RESULTS_BY_SKILL.csv
outputs/pilot150/PILOT_COST.csv
```

主表：

| Method | Initial ID | Final ID | ID Gain | Initial Transfer | Final Transfer | Transfer Gain | Retention | Updater Tokens | Env Steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| None | | | | | | | | | |
| Raw Memory | | | | | | | | | |
| ACE-Context | | | | | | | | | |
| WorldMind | | | | | | | | | |
| EmbodiSkill | | | | | | | | | |

另外必须有：

```text
crash rate
invalid action rate
probe mutation count
leakage violation count
```

---

# 51. 不要在 Pilot 阶段做的事情

禁止：

1. 为了让 RSI 看起来有效人工挑任务；
2. 根据结果重新挑 Pilot；
3. 给某方法单独更长 context；
4. 给某方法更强 updater；
5. 让 WorldMind 看 private state；
6. 给 EmbodiSkill 人工初始化 skill；
7. 给 ACE 人工写 playbook；
8. probe 后继续保留 probe experience；
9. 修改 OpenETA planner prompt；
10. 看到结果差就偷偷换 agent；
11. 看到 TVR 差就给 target pose；
12. 把 benchmark 的 skill-family label 提示给模型。

---

# 52. Coding Agent 的执行顺序

必须严格按下列顺序执行。

```text
Phase A
00_verify_dataset.py
    ↓
PASS

Phase B
01_clone_pin_external.sh
    ↓
03_verify_openeta_freeze.py
    ↓
PASS

Phase C
02_check_deepseek_api.sh
    ↓
DeepSeek API identity smoke
text smoke
vision smoke
OpenETA backend compatibility smoke
request-body audit
    ↓
PASS

Phase D
实现 base adapter
实现五个 source adapter
    ↓
04_validate_adapters.py
    ↓
leakage audit
    ↓
PASS

Phase E
实现 OpenETA bridge
    ↓
06_smoke_openeta.py
    ↓
PASS

Phase F
实现 RSI base
none
raw_memory
ACE
WorldMind
EmbodiSkill
    ↓
07_smoke_rsi.py
    ↓
PASS

Phase G
05_build_pilot150.py
    ↓
锁死 pilot manifest
    ↓
08_run_pilot150.py

Phase H
09_analyze_pilot.py
10_audit_release.py
```

禁止跳 Gate。

---

# 53. 每个 Phase 结束必须更新 `CURRENT_STATE.md`

格式：

```markdown
# Current State

## Last completed gate
G3 Adapter Gate

## Status
PASS

## Evidence
- ...

## Known issues
- ...

## Next step
G4 OpenETA Baseline Gate
```

不要只在聊天记录里说“跑通了”。

---

# 54. 失败时怎么处理

## Adapter 失败

修 adapter。

不要换 RSI。

## OpenETA 基线失败

先证明：

- model vision 正常；
- tool schema 正常；
- source action 正常。

不要上 RSI。

## ACE 不工作

检查：

- playbook 是否实际注入；
- Reflector 是否拿到正确 trajectory；
- Curator 是否有真实 delta；
- playbook 是否 collapse。

不要自己写新 ACE prompt 替代官方逻辑。

## WorldMind 不工作

检查：

- prediction sidecar 是否合理；
- Process Experience 是否过多错误规则；
- Goal Experience 是否只来自 success；
- retrieval 是否真的命中。

## EmbodiSkill 不工作

先区分：

```text
integration failure
vs
method failure
```

如果官方代码本身过度 ALFWorld-specific：

写 `BLOCKERS.md`，暂停。

不要伪造一个“兼容版本”。

---

# 55. 第一轮结果怎么看

Pilot 不要求所有 RSI 都提升。

真正值得看的现象包括：

### A. Raw Memory 已经很强

说明：

> 很多“self-improvement”其实只是 retrieval benefit。

### B. ACE ID 提升但 Transfer 不提升

说明：

> playbook 可能过拟合 Experience distribution。

### C. WorldMind Transfer 强

说明：

> causal/world-rule experience 可能比 task-specific procedure 更跨环境。

### D. EmbodiSkill ID 强但 Retention 差

说明：

> skill update 可能存在 destructive overwrite。

### E. 所有方法都不提升

也不是立刻说明 benchmark 失败。

要查：

- executor 是否太弱；
- experience 数量是否太少；
- cross-source semantics 是否真的共享；
- retrieval 是否使用；
- update 是否有行为影响。

---

# 56. 第一轮之后的决策树

## 若至少 2 个 RSI 出现稳定正 Gain

进入：

```text
Full Core v1.0
```

并新增：

- 3 random seeds/curricula；
- compute-matched TTS；
- TF-GRPO；
- PRACTICE（若官方代码可稳定复现）；
- OpenETA-native SI reference；
- Qwen3.5-9B / Qwen3-VL architecture generalization。

## 若只有 ID Gain，没有 Transfer

重点排查：

> Benchmark 是否测到 skill accumulation，却没有测到真正 generalizable self-improvement。

可能需要重新加强：

- transfer pair design；
- cross-source semantic alignment；
- unseen-object / unseen-scene / unseen-composition probes。

## 若 Transfer 正，但 Retention 崩

这是很有价值的结果：

> self-evolution induces negative transfer / forgetting。

Full benchmark 应强化 stability track。

## 若所有方法都负 Gain

先做 pathway audit，不要立刻改任务：

```text
有没有更新？
更新是否正确？
有没有检索？
检索有没有进入 planner？
进入 planner 后有没有改变行为？
改变行为后为什么更差？
```

---

# 57. 最终 Pilot 交付物

coding agent 完成后必须交付：

```text
1. 可运行代码
2. external source pins
3. OpenETA freeze manifest
4. 5 个 source adapters
5. leakage tests
6. 5 个 primary conditions
7. Pilot-150 manifest
8. 全部 episode trajectories
9. RSI update traces
10. frozen probe hashes
11. token/cost accounting
12. PILOT_RESULTS.csv
13. PILOT_REPORT.md
14. CURRENT_STATE.md
15. BLOCKERS.md（若有）
```

---

# 58. `10_audit_release.py` 最终硬性检查

最终脚本必须 assert：

```python
assert dataset_audit["status"] == "PASS"
assert freeze_audit["status"] == "PASS"
assert leakage_violation_count == 0
assert probe_mutation_count == 0
assert snapshot_reload_failures == 0
assert infrastructure_crash_rate < 0.02
assert all_primary_conditions_completed
```

然后才写：

```text
FINAL_STATUS: PASS
```

否则：

```text
FINAL_STATUS: FAIL
```

不得人工改成 PASS。

---

# 59. 给 coding agent 的最后提醒

本项目当前最危险的错误不是模型准确率低，而是**实验比较不干净**。

优先级必须是：

```text
1. 数据无泄漏
2. OpenETA 真冻结
3. RSI state 真隔离
4. Probe 真只读
5. 同模型同预算
6. trajectory 可追溯
7. 最后才看成功率
```

如果为了“让方法跑起来”而：

- 改 OpenETA planner；
- 给某个方法更多 hidden information；
- 用 GPT-5.2 帮 EmbodiSkill，却让 ACE / WorldMind 使用 DeepSeek；
- probe 后继续学习；
- 让不同 condition 使用不同 task order；

那么即使最终 SR 很高，这轮实验也不能进入论文。

第一轮 Pilot 的成功标准不是“RSI 一定提升”，而是：

> **我们能够在一个公开、冻结、统一的 embodied agent 上，以严格隔离的方式运行多种现有 RSI 方法，并可靠测量它们如何随 experience 改变 ID、Transfer、Retention 与 Cost。**

做到这一点，再扩大到 Full Core。

---

# Appendix A. Canonical external repositories

```text
OpenETA
https://github.com/OpenMOSS/OpenETA
pin: 7d4a0a1522ba8ebbd362bde880bad81d2a98f15e

ACE
https://github.com/ace-agent/ace
pin: 82709de050e1db6e6ef2f07bcb0393560b94992a

WorldMind
https://github.com/zjunlp/WorldMind
pin: 712b0fd53b4bd6a1948603f087a1e9f25a5adaea

EmbodiSkill
https://github.com/air-embodied-brain/EmbodiSkill
pin: 760126030eab1d33ec6a6f30988f0f1fb58df3a7
```

---

# Appendix B. OpenETA upstream locations that MUST be audited

```text
external/OpenETA/agent/runtime/planner.py
external/OpenETA/agent/runtime/episode.py
external/OpenETA/agent/runtime/runtime.py
external/OpenETA/agent/runtime/runtime_assembly.py
external/OpenETA/agent/runtime/self_improvement.py
external/OpenETA/agent/runtime/memory.py
external/OpenETA/agent/runtime/planner_prompts.py
external/OpenETA/agent/runtime/actions.py
external/OpenETA/agent/runtime/pipeline.py
external/OpenETA/agent/backends/planner.py
external/OpenETA/adapter/protocol.py
```

---

# Appendix C. WorldMind official plugin locations

```text
external/WorldMind/Plugin/README.md
external/WorldMind/Plugin/worldmind_plugin/core.py
external/WorldMind/Plugin/worldmind_plugin/config.py
external/WorldMind/Plugin/worldmind_plugin/discriminator.py
external/WorldMind/Plugin/worldmind_plugin/reflector.py
external/WorldMind/Plugin/worldmind_plugin/state_summarizer.py
external/WorldMind/Plugin/worldmind_plugin/experience_refiner.py
external/WorldMind/Plugin/worldmind_plugin/knowledge_manager.py
```

---

# Appendix D. EmbodiSkill official core locations

```text
external/EmbodiSkill/tasks/run_epochs.py
external/EmbodiSkill/agentkit/skill/common.py
external/EmbodiSkill/agentkit/skill/embodiskill_skill/EmbodiSkill.py
external/EmbodiSkill/agentkit/skill/embodiskill_skill/prompt.py
external/EmbodiSkill/agentkit/skill/embodiskill_skill/skill_base.py
```

---

# Appendix E. 一句话实验定义

本 Pilot 最终必须可以被概括为：

> **Starting from empty persistent state, we expose the same frozen OpenETA + DeepSeek `deepseek-flash` embodied agent to the same experience stream, vary only the self-evolution mechanism, and evaluate frozen checkpoints on held-out ID, cross-source transfer, and retention probes.**

这句话如果因为代码实现而无法成立，说明实验设计已经被破坏，必须先修复再继续。
