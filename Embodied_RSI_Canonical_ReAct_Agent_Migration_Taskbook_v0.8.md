# Embodied RSI Benchmark：Canonical Multimodal ReAct Agent 迁移与 RSI 集成执行任务书

> **版本**：v0.8-draft  
> **日期**：2026-09-17  
> **目标仓库**：`Rongdingyi/EmbodiedRSIBench`  
> **迁移起点 commit**：`56ad3aaba4fe9c0d2cc11e6903f08bde1fab5543`  
> **用途**：直接交给 coding agent / Codex / OpenCode 执行。  
> **任务性质**：代码重构 + canonical agent 替换 + 现有 RSI 重新接入。  
> **核心要求**：把 OpenETA 从 canonical agent 降级为 legacy/reference track，新增并冻结一个 **Canonical Multimodal ReAct Planner–Executor Agent**；同一个 agent 架构下只切换 RSI 方法。

---

# 0. 先读这一页：coding agent 不允许自行解释任务

## 0.1 本次迁移只回答一个问题

新的 benchmark 要严格实现：

```text
Same Tasks
+ Same Canonical Embodied Agent
+ Same Backbone Model
+ Same Prompt Skeleton
+ Same Observation Interface
+ Same Action Interface
+ Same Working Memory
+ Same Recovery / Replanning
+ Same Episode Budget
+ Different Cross-Episode RSI Method
```

即：

```text
固定 Agent
固定模型
固定环境
固定预算
固定当前 episode 内的工作记忆与 recovery

唯一变化：RSI 如何把过去 Experience 转成跨 episode 的持久状态，并在未来 episode 使用。
```

**不要把“换 agent”误解为“取消 agent”。**

目标不是裸 MLLM：

```text
RGB -> MLLM -> action
```

目标是完整的、冻结的 closed-loop embodied agent：

```text
Task + RGB + env feedback + current-episode history + RSI context
                         |
                         v
               Multimodal Planner
                         |
                    reasoning
                         |
               EXACTLY ONE ACTION
                         |
                         v
                     Executor
                         |
                         v
                    Simulator
                         |
          new RGB + public feedback
                         |
                         +--------> replan
```

论文/代码统一称：

> **Canonical Multimodal ReAct Planner–Executor Agent**

架构参考 EmbodiedBench 的 closed-loop `VLMPlanner`，但为了跨五个环境公平、并支持 WorldMind step-level prediction，canonical 协议规定：

> **one planner decision -> exactly one real environment action**

不允许一次生成并连续执行 `[a1, a2, a3, ...]`。

---

## 0.2 本次绝对禁止做的事情

Coding agent 如果做下面任何一条，任务视为失败：

1. **禁止删除旧 OpenETA 代码和旧结果。**
2. **禁止覆盖 `outputs/pilot60/` 的 OpenETA historical results。**
3. **禁止让新的 canonical runner import `benchmark.openeta_bridge.*`。**
4. **禁止重新实现一套伪 ACE / 伪 WorldMind / 伪 EmbodiSkill。**
5. **禁止为了某个 RSI 单独修改 canonical agent system prompt。**
6. **禁止给 WorldMind 更多 action、给 ACE 不同 history、给 RawMemory 不同 recovery。**
7. **禁止让 RSI 读取 private evaluator、goal predicates、golden actions、hidden simulator state。**
8. **禁止 probe 更新任何跨 episode state。**
9. **禁止用随机 action 修复 JSON/action parsing 错误。**
10. **禁止 agent 自己声明 benchmark success。成功必须来自 source official environment/evaluator。**
11. **禁止在完成迁移后自动启动完整 Pilot-60。**
12. **禁止顺手换 backbone model。** 本次先继续使用当前固定模型配置；换模型属于单独 protocol revision。
13. **禁止用 EmbodiSkill 的 `TeamSolver`、ground-truth fallback agent 作为 canonical agent。**
14. **禁止把 OpenETA native tools/skills 重新偷偷带进新 agent。**
15. **禁止把 model-generated reasoning 作为 RSI 的额外 privileged input。** RSI 继续只看公开 task/action/feedback/outcome。

---

# 1. 为什么要做这次迁移

## 1.1 当前 OpenETA Pilot-60 已完成，但出现严重 floor effect

当前 Pilot-60 报告显示：

- `none` 初始 ID 仅约 10%，最终 0%；
- `ace_context` ID/Transfer 基本全 0；
- `raw_memory` ID/Transfer 基本全 0；
- `worldmind` 最终 ID 只有单题级改善；
- Transfer 几乎全方法全 checkpoint 为 0；
- probe state invariants 是正常的。

这说明工程链路大体可运行，但 operating point 太低，难以测 RSI gain。

## 1.2 当前 OpenETA canonical stack 与完整 OpenETA 也不是同一能力栈

当前仓库的 `benchmark/openeta_bridge/build_runtime.py` 明确：

- 只暴露 benchmark source actions；
- `skills=SkillRegistry()`，即没有 OpenETA built-in skills；
- 禁止 `sam3 / anygrasp / code_policy / observe / sense ...`；
- native self-improvement disabled；
- Pilot-60 的 OpenETA turn budget 之前仅为 8。

因此当前系统更准确地说是：

```text
OpenETA Stage-2 planner/episode harness
+ benchmark-native action space
+ stripped native skills/tools
+ DeepSeek planner
```

不适合作为长期 canonical substrate。

## 1.3 文献侧的设计依据

本次 canonical agent 取多个 non-parametric RSI 工作的共同交集：

- ACE：ReAct-style agent + evolving context/playbook；
- WorldMind：官方直接包裹 EmbodiedBench `VLMPlanner`；
- EmbodiSkill：有完整 Agent/Orchestrator，但 skill/manual 是独立 persistent layer；
- RawMemory：天然是 agent-agnostic persistent memory baseline。

所以 canonical substrate 应该是：

> **冻结的多模态 planner–executor closed loop + episode-local working memory + feedback-conditioned replanning**

而不是一个包含大量自身学习机制的重型 framework。

---

# 2. 版本、外部代码与 provenance 必须冻结

## 2.1 当前仓库迁移起点

必须从：

```text
Rongdingyi/EmbodiedRSIBench
56ad3aaba4fe9c0d2cc11e6903f08bde1fab5543
```

建立新 branch：

```text
refactor/canonical-react-agent
```

不要直接在 main 上边写边试。

## 2.2 EmbodiedBench 参考 pin

新增参考 source：

```text
repo: https://github.com/EmbodiedBench/EmbodiedBench
commit: 9be4e980e9cd6bcb38373cd4aab7c32724bdd401
reference file: embodiedbench/planner/vlm_planner.py
```

**注意：**

- canonical runtime 不要求直接 import EmbodiedBench package；
- 这是 architecture/reference pin；
- 我们跨五个 source 有自己的 adapter，因此应实现统一 agent package；
- 如果复制 upstream 代码片段，必须检查并保留 upstream license/attribution；
- 不要永远依赖 `master`。

## 2.3 继续保留现有 RSI pin

当前 `manifests/external_sources.json` 中这些 pin 不动：

```text
ACE:
82709de050e1db6e6ef2f07bcb0393560b94992a

WorldMind:
712b0fd53b4bd6a1948603f087a1e9f25a5adaea

EmbodiSkill:
760126030eab1d33ec6a6f30988f0f1fb58df3a7

OpenETA historical reference:
7d4a0a1522ba8ebbd362bde880bad81d2a98f15e
```

修改 `manifests/external_sources.json`，新增：

```json
"EmbodiedBench": {
  "repo": "https://github.com/EmbodiedBench/EmbodiedBench",
  "commit": "9be4e980e9cd6bcb38373cd4aab7c32724bdd401",
  "purpose": "canonical agent architecture reference only"
}
```

不要删除 OpenETA entry，因为旧实验要可复现。

---

# 3. 哪些现有代码必须保留，哪些代码降级为 legacy

## 3.1 必须保留，不要大改

以下属于 benchmark asset，不是 OpenETA asset：

```text
embodied_rsi/benchmark/adapters/
embodied_rsi/benchmark/registry/
embodied_rsi/benchmark/runner/accounting.py
embodied_rsi/benchmark/runner/gates.py
embodied_rsi/benchmark/rsi/
embodied_rsi/manifests/
embodied_rsi/configs/       # 新增配置，不要覆盖旧协议
embodied_rsi/outputs/       # 历史结果不可删
```

尤其：

```text
benchmark/adapters/base.py
```

已有接口：

```python
PublicObservation(
    instruction,
    images,
    text_feedback,
    public_metadata,
)

StepOutcome(
    observation,
    action_success,
    reward_public,
    terminated,
    truncated,
    public_feedback,
)
```

这个接口正好适合新 agent，不要另造第二套 simulator protocol。

## 3.2 保留但降级为 legacy/reference

整个目录保留：

```text
benchmark/openeta_bridge/
```

不要删除，不要重命名，以免历史代码/结果 provenance 断裂。

但是新 canonical 路径不得 import 它。

在目录增加：

```text
benchmark/openeta_bridge/LEGACY.md
```

内容写清：

```text
This package reproduces the historical OpenETA substrate used by Pilot-60 v0.7.
It is no longer the canonical agent for the main RSI benchmark after protocol v0.8.
Do not import this package from canonical_react_agent code paths.
```

## 3.3 旧 scripts 不删除

保留：

```text
03_verify_openeta_freeze.py
06_smoke_openeta.py
08_run_pilot60.py
```

把它们视为 historical track。

新建新 scripts，不要覆盖旧脚本：

```text
03b_verify_canonical_agent.py
06b_smoke_canonical_agent.py
07b_smoke_rsi_canonical.py
08b_run_pilot60_canonical.py
10b_audit_canonical_release.py
```

`09_analyze_pilot.py` 可以改造成可接受 `--root` / `--protocol`，但必须保持能读取旧 Pilot-60。

---

# 4. 目标目录结构

最终至少应形成：

```text
embodied_rsi/
├── benchmark/
│   ├── adapters/                       # 保持
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── types.py
│   │   ├── prompt.py
│   │   ├── vision.py
│   │   ├── backend.py
│   │   ├── action_schema.py
│   │   ├── working_memory.py
│   │   ├── react_agent.py
│   │   ├── episode_runner.py
│   │   └── freeze.py
│   ├── openeta_bridge/                 # historical only
│   ├── registry/
│   ├── rsi/
│   │   ├── base.py
│   │   ├── context.py                  # 从 openeta_bridge 解耦
│   │   ├── none.py
│   │   ├── raw_memory.py
│   │   ├── ace_context.py
│   │   ├── worldmind.py
│   │   └── embodiskill.py
│   └── runner/
│       ├── accounting.py
│       └── gates.py
├── configs/
│   ├── agent/
│   │   └── canonical_react.yaml
│   └── pilot60_canonical.yaml
├── manifests/
│   ├── pilot60.json                    # 原 task set 继续用
│   ├── canonical_agent_reference.json
│   └── external_sources.json
├── scripts/
│   ├── 03b_verify_canonical_agent.py
│   ├── 06b_smoke_canonical_agent.py
│   ├── 07b_smoke_rsi_canonical.py
│   ├── 08b_run_pilot60_canonical.py
│   ├── 09_analyze_pilot.py
│   └── 10b_audit_canonical_release.py
└── tests/
    ├── test_agent_one_action_per_turn.py
    ├── test_agent_no_cross_episode_state.py
    ├── test_agent_prompt_contract.py
    ├── test_agent_action_validation.py
    ├── test_agent_feedback_replanning.py
    ├── test_agent_budget_semantics.py
    ├── test_agent_dynamic_tools.py
    ├── test_agent_request_audit.py
    ├── test_agent_private_leakage.py
    ├── test_worldmind_step_timing_canonical.py
    ├── test_rsi_context_decoupled.py
    └── ... existing tests ...
```

---

# 5. Phase A：先把 RSI 从 OpenETA 解耦

这是第一项代码任务。不要先写 agent。

## A1. 新建 `benchmark/rsi/context.py`

把当前：

```text
benchmark/openeta_bridge/context_injection.py
```

中的通用 RSI 内容移动/复制到：

```text
benchmark/rsi/context.py
```

至少包含：

```python
OPEN_TAG = "[PERSISTENT EXPERIENCE GUIDANCE]"
CLOSE_TAG = "[/PERSISTENT EXPERIENCE GUIDANCE]"
MAX_RSI_INJECTION_TOKENS = 8192

@dataclass
class RSIInjection:
    context_text: str = ""
    provenance_ids: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    truncated: bool = False


def count_tokens(text: str) -> int:
    ...


def wrap_guidance(text: str) -> str:
    ...


def make_injection(...):
    ...


def injection_hash(...):
    ...


def log_injection(...):
    ...
```

### A1.1 重要修改

现在 `count_tokens()` 会尝试 import OpenETA tokenizer：

```python
from agent.runtime.token_counting import estimate_text_tokens
```

新文件禁止依赖 OpenETA。

使用：

```python
def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)
```

这里只用于 shared budget estimate；在 provenance 里写：

```text
rsi_token_counter = tiktoken-cl100k-or-char4-estimate
```

不要假装它是 DeepSeek exact tokenizer。

## A2. 修改所有 RSI imports

必须搜索整个 repo：

```bash
grep -R "benchmark.openeta_bridge.context_injection" -n embodied_rsi
```

至少修改：

```text
benchmark/rsi/base.py
benchmark/rsi/none.py
benchmark/rsi/raw_memory.py
benchmark/rsi/ace_context.py
benchmark/rsi/worldmind.py
benchmark/rsi/embodiskill.py
```

全部改为：

```python
from benchmark.rsi.context import ...
```

## A3. 修改 `RSIMethod` 文档

当前 `base.py` 里有：

```text
cannot touch ... OpenETA core
```

改成更通用：

```text
RSI method cannot touch the canonical agent internals, simulator, action schema,
private evaluator, or benchmark task selector.
```

`predict_sidecar()` docstring 中：

```text
Runs after OpenETA has locked the action
```

改成：

```text
Runs after the canonical agent has selected and validated the action,
and before the environment executes it.
```

## A4. backward compatibility

旧 `benchmark/openeta_bridge/context_injection.py` 不删除。

改成 compatibility shim：

```python
from benchmark.rsi.context import *
```

这样旧 OpenETA track 仍可复现。

## A5. Phase A 验收

执行：

```bash
grep -R "openeta_bridge.context_injection" -n embodied_rsi/benchmark/rsi
```

必须没有结果。

现有 snapshot/read-only tests 必须继续 PASS。

---

# 6. Phase B：定义 canonical agent 数据类型

## B1. 新建 `benchmark/agent/types.py`

写：

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentDecision:
    action_name: str
    parameters: dict
    thought: str = ""
    raw_response: str = ""
    planner_calls: int = 1
    validation_errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkingMemoryStep:
    step_idx: int
    action_name: str
    parameters: dict
    action_success: bool | None
    public_feedback: str
    terminated: bool
    truncated: bool


@dataclass
class AgentEpisodeState:
    instruction: str
    steps: list[WorkingMemoryStep] = field(default_factory=list)


@dataclass
class CanonicalEpisodeResult:
    task: dict
    public_trajectory: list[dict]
    private_evaluation: dict
    outcome: dict
    env_steps: int
    planner_calls: int
    wall_time_s: float
    accounting: dict
    leakage_violations: list[str]
    status: str
    error: str | None = None
```

**不要把 simulator/private fields 放进这些 dataclass。**

---

# 7. Phase C：定义 Agent 配置

## C1. 新建 `configs/agent/canonical_react.yaml`

第一版固定：

```yaml
agent_name: canonical_multimodal_react_v1
architecture: multimodal_react_planner_executor
reference: embodiedbench_vlmplanner
reference_commit: 9be4e980e9cd6bcb38373cd4aab7c32724bdd401

planner:
  temperature: 0.0
  max_output_tokens: 2048
  max_validation_retries: 2
  max_images: 2
  json_mode: prompt_only

observation:
  current_images_only: true
  include_public_feedback: true
  include_public_metadata: false
  include_target_image_when_source_exposes_it: true

working_memory:
  include_action_history: true
  include_feedback_history: true
  include_reasoning_history: false
  max_history_tokens: 12000
  truncation: drop_oldest_steps_deterministically
  model_generated_summary: false

execution:
  one_environment_action_per_turn: true
  random_fallback_action: false
  trust_agent_success_claim: false

rsi:
  max_injection_tokens: 8192
  single_shared_injection_slot: true

recovery:
  deterministic_json_extract: true
  planner_retry_on_invalid_json: true
  planner_retry_on_invalid_action: true
  stuck_warning_after_identical_actions: 3
  auto_choose_recovery_action: false
```

## C2. 模型配置

本次迁移不要同时换模型。

backend 继续读取：

```text
DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL
DEEPSEEK_MODEL
```

并保证：

```text
same DEEPSEEK_MODEL for canonical planner and RSI updater modules
```

如果以后要换模型，新建 protocol revision，不要在同一次 migration 偷换。

---

# 8. Phase D：写固定 Agent system prompt

## D1. 新建 `benchmark/agent/prompt.py`

必须把 prompt 写成代码常量，不能 scattered string concatenation。

建议内容直接使用以下版本，不要自行发挥：

```python
CANONICAL_SYSTEM_PROMPT = r"""
You are the planner of a closed-loop embodied agent.

Your job is to complete the user's physical task by repeatedly observing the
current environment, reasoning about the next useful step, and selecting exactly
ONE action from the provided legal action set.

Hard rules:
1. Select exactly one environment action per turn.
2. Use only action names and parameters that appear in the legal action schema.
3. Never invent hidden object ids, simulator state, success predicates, or goal state.
4. Use the current visual observation, public environment feedback, and current-episode
   action history to recover from failed actions and replan.
5. Persistent cross-episode guidance is advisory experience, not ground truth.
6. Do not claim benchmark success yourself. The environment decides termination and the
   private evaluator decides task success.
7. If the previous action failed, use the public feedback to choose a different useful
   next action when appropriate.
8. Return JSON only. Do not return markdown fences.

Required JSON schema:
{
  "thought": "brief task-relevant reasoning for the next action",
  "action": {
    "name": "one legal action name",
    "parameters": {}
  }
}
""".strip()
```

## D2. user prompt 必须有固定槽位

写函数：

```python
def build_turn_prompt(
    *,
    instruction: str,
    tool_specs: list[dict],
    working_memory_text: str,
    public_feedback: str,
    rsi_injection: RSIInjection,
    stuck_warning: str = "",
) -> str:
    ...
```

顺序固定：

```text
## Task
{instruction}

## Persistent cross-episode guidance
{RSI slot}

## Current public environment feedback
{feedback}

## Current-episode action history
{history}

## Legal actions
{schema}

## Optional recovery note
{stuck warning}

Select exactly one next action and return JSON only.
```

### D2.1 RSI slot 的公平规则

所有方法必须走同一槽位。

`none` 时不要删除整个 section；写：

```text
[PERSISTENT EXPERIENCE GUIDANCE]
(none)
[/PERSISTENT EXPERIENCE GUIDANCE]
```

其他方法使用已有 `RSIInjection.context_text`。

因此 prompt skeleton 完全相同。

## D3. 不允许方法专属 system prompt

禁止：

```python
if method == "worldmind":
    system_prompt += ...
```

禁止：

```python
if method == "ace_context":
    recovery = ...
```

唯一方法差异是 `RSIInjection.context_text` 以及方法自己的 updater sidecar。

---

# 9. Phase E：视觉输入编码

## E1. 新建 `benchmark/agent/vision.py`

职责只有：

```text
numpy/PIL RGB -> data:image/png;base64,...
```

接口：

```python
def image_to_data_url(image) -> str:
    ...


def observation_image_parts(observation, max_images: int = 2) -> list[dict]:
    ...
```

规则：

1. 只处理 `PublicObservation.images`；
2. 最多 2 张；
3. TVR 的 current + target 可同时进；
4. 其他 source 通常只当前 RGB；
5. 不从 `private_eval_metadata` 找图片；
6. 不把 artifact path 里的 task id 暴露给模型；
7. request dump 继续由 accountant 把 data URL redacted。

---

# 10. Phase F：Agent backend

## F1. 新建 `benchmark/agent/backend.py`

不要复用 OpenETA backend class。

实现一个小而透明的 backend：

```python
class CanonicalVLMBackend:
    def __init__(self, config, accountant=None):
        ...

    def complete(
        self,
        *,
        system_prompt: str,
        user_text: str,
        images: list,
        role: str = "canonical_planner",
    ) -> dict:
        ...
```

## F2. request body

OpenAI-compatible message 形式：

```python
messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": [
            *image_parts,
            {"type": "text", "text": user_text},
        ],
    },
]
```

body 至少：

```python
{
    "model": MODEL,
    "messages": messages,
    "temperature": 0.0,
    "max_tokens": 2048,
}
```

第一版不要依赖 provider-specific function calling。

原因：canonical action schema 已经由 prompt + validator 控制，避免不同 provider tool-call semantics 成为 confound。

## F3. accounting

每次真实 planner request 必须调用：

```python
accountant.record_planner_exchange(..., role="canonical_planner")
```

继续生成：

```text
public_context_dump.jsonl
```

每个 model request 都必须可审计。

## F4. error handling

backend 层只处理：

- HTTP/API errors；
- retry/backoff；
- token usage extraction。

不要在 backend 层选择 action。

---

# 11. Phase G：Action schema、JSON parse 与 validation

## G1. 新建 `benchmark/agent/action_schema.py`

接口：

```python
def format_tool_specs_for_prompt(tool_specs: list[dict]) -> str:
    ...


def parse_agent_json(raw: str) -> tuple[dict | None, list[str]]:
    ...


def validate_agent_action(
    payload: dict,
    tool_specs: list[dict],
) -> tuple[str | None, dict | None, list[str]]:
    ...
```

## G2. JSON repair 只能 deterministic

允许：

- strip markdown fences；
- 从文本中提取第一个 balanced JSON object；
- 去首尾 whitespace。

不要写：

```python
if parse_fail:
    choose_random_action()
```

不要让 regex 猜 action id。

## G3. validation

必须验证：

1. `action.name` 在当前 episode `tool_specs`；
2. `parameters` 是 dict；
3. 使用 tool spec 的 JSON schema 验证 required/type/enum；
4. 不允许额外的 private keys；
5. 一次 response 只能有一个 action。

可以使用 `jsonschema`。

## G4. invalid response retry

每一 environment action 最多：

```text
initial planner call + 2 validation retries
```

retry 时，把错误以固定文本追加：

```text
Your previous response was invalid for the following protocol reason:
- ...
Return a corrected JSON object with exactly one legal action.
```

这属于 canonical agent recovery，所有 RSI 一样。

如果三次都失败：

```text
status = FAIL
failure_reason = planner_protocol_failure
```

不要执行随机 action。

---

# 12. Phase H：Current-episode working memory

## H1. 新建 `benchmark/agent/working_memory.py`

Agent memory 只能是 episode-local。

接口：

```python
class EpisodicWorkingMemory:
    def __init__(self, max_history_tokens: int = 12000):
        self.steps = []

    def append(self, step: WorkingMemoryStep) -> None:
        ...

    def render(self) -> str:
        ...

    def clear(self) -> None:
        ...
```

## H2. memory 内容

每步只存：

```text
step index
chosen action name
public parameters
public action_success
public feedback
terminated/truncated
```

不要存：

```text
private evaluator result
hidden state
success predicate
model chain-of-thought
RSI internal state
```

## H3. token cap

当 history 太长：

- deterministic drop oldest steps；
- 保留最近 steps；
- 不调用额外 LLM summary；
- 记录 `history_steps_dropped`。

这样不同 RSI 不会因为总结调用不同而产生额外 confound。

## H4. 跨 episode 必须为空

每次 `run_episode()` 新建一个 `EpisodicWorkingMemory`。

禁止 global singleton。

禁止写磁盘并在下一 episode 自动 reload。

---

# 13. Phase I：Canonical ReAct Agent 本体

## I1. 新建 `benchmark/agent/react_agent.py`

建议接口：

```python
class CanonicalMultimodalReActAgent:
    def __init__(self, *, backend, config, accountant=None):
        self.backend = backend
        self.config = config
        self.accountant = accountant
        self.memory = EpisodicWorkingMemory(...)
        self.tool_specs = []
        self.instruction = ""
        self.rsi_injection = None

    def start_episode(
        self,
        *,
        instruction: str,
        tool_specs: list[dict],
        rsi_injection: RSIInjection,
    ) -> None:
        ...

    def decide(self, observation: PublicObservation) -> AgentDecision:
        ...

    def record_transition(self, ...):
        ...

    def close_episode(self) -> None:
        self.memory.clear()
```

## I2. `start_episode()`

必须：

```python
self.memory.clear()
self.instruction = instruction
self.tool_specs = deep_copy(tool_specs)
self.rsi_injection = rsi_injection
```

不允许保留前一集任何 planner messages/history。

## I3. `decide()` 流程

严格：

```text
1. render episodic memory
2. build fixed user prompt
3. attach current public images
4. model call
5. deterministic JSON parse
6. validate one action
7. invalid -> fixed retry prompt
8. return AgentDecision
```

**`decide()` 绝对不能调用 environment.step。**

## I4. stuck warning

如果连续 3 次：

```text
same action name + same parameters
```

且反馈没有明显变化，下一 turn prompt 增加：

```text
Recovery note: the previous action pattern has repeated. Re-evaluate the current
observation and choose a different legal action if the repeated action is not
making progress.
```

Agent 仍自己决策。

禁止自动替它选 recovery action。

---

# 14. Phase J：新的 canonical episode runner

## J1. 新建 `benchmark/agent/episode_runner.py`

不要复制 OpenETA episode runner。

这是 benchmark 自己的 canonical agent runner。

## J2. 关键原则

```text
1 planner decision = 1 real adapter.step()
```

一次 model response 不允许执行多个 source actions。

## J3. reset / dynamic tools 顺序

非常重要：

```python
observation = adapter.reset(task)
tool_specs = adapter.build_tool_specs(task)
```

**必须 reset 在前。**

因为 EB-Habitat 的 70 skills 是 live episode schema。

## J4. runner 伪代码

必须按这个时序实现：

```python
def run_episode(task, adapter, rsi, *, role, config, output_dir=None):
    t0 = time.time()
    accountant = Accountant(episode_dir=output_dir)

    if rsi is not None:
        rsi.run_ctx["accountant"] = accountant

    # 1. reset real source environment
    observation = adapter.reset(task)

    # 2. obtain live legal actions AFTER reset
    tool_specs = adapter.build_tool_specs(task)
    assert tool_specs

    # 3. get RSI cross-episode guidance
    public_task_view = {
        "global_task_id": task["global_task_id"],
        "instruction": observation.instruction or task.get("instruction"),
        "source_dataset": task["source_dataset"],
        "scene_id_canonical": task.get("scene_id_canonical"),
        "difficulty_canonical": task.get("difficulty_canonical"),
    }
    injection = rsi.before_episode(public_task_view) if rsi else make_injection("")
    accountant.record_injection(injection.estimated_tokens)

    # 4. fresh canonical agent session
    agent = CanonicalMultimodalReActAgent(...)
    agent.start_episode(
        instruction=observation.instruction,
        tool_specs=tool_specs,
        rsi_injection=injection,
    )

    trajectory = []
    error = None

    # 5. closed loop
    for env_step in range(max_agent_actions_for_source):
        state_before = public_view(observation)

        decision = agent.decide(observation)

        # WorldMind sidecar: AFTER action selection, BEFORE env step
        prediction = None
        if rsi is not None:
            prediction = rsi.predict_sidecar(
                public_observation=state_before,
                action={
                    "name": decision.action_name,
                    "parameters": decision.parameters,
                },
                accountant=accountant,
            )

        outcome = adapter.step(decision.action_name, decision.parameters)
        accountant.record_env_step()

        state_after = public_view(outcome.observation)

        # step-level RSI hook only sees public data
        if rsi is not None:
            rsi.after_step({
                "task_instruction": observation.instruction,
                "role": role,
                "action": {
                    "name": decision.action_name,
                    "parameters": decision.parameters,
                },
                "predicted_state": prediction,
                "state_before": state_before,
                "state_after": state_after,
                "public_feedback": outcome.public_feedback,
                "action_success": outcome.action_success,
                "terminated": outcome.terminated,
                "truncated": outcome.truncated,
            })

        trajectory.append(public_step(...))
        agent.record_transition(...)
        observation = outcome.observation

        if outcome.terminated or outcome.truncated:
            break

    # 6. private evaluator AFTER loop
    private_eval = adapter.private_evaluate()

    # 7. Experience only: RSI persistent update
    if role == "experience" and rsi is not None:
        rsi.after_episode(public_episode(...), public_outcome(...))
        accountant.record_rsi_update(...)

    # 8. leakage / dumps / close
    ...
```

## J5. error semantics

区分：

```text
planner_protocol_failure
simulator_infrastructure_error
rsi_update_error
max_agent_actions
source_terminated
source_truncated
```

不要把全部异常揉成一个字符串。

## J6. success semantics

`AgentDecision.thought` 里即使写：

```text
Task complete.
```

也不能算成功。

只认：

```python
private_eval.get("success")
```

如果 source 有官方 Stop/EndTask，它只是一个合法 action；最终 success 仍由 evaluator 决定。

---

# 15. Phase K：重新定义 budget

**绝对不要把 OpenETA 的 `openeta_turns: 8` 带到新 agent。**

因为新 canonical agent 已规定：

```text
1 turn = exactly 1 source environment action
```

所以直接把现有 source `max_env_steps` 作为 `max_agent_actions`。

新建：

```text
configs/pilot60_canonical.yaml
```

建议：

```yaml
seed: 20260915

roles:
  experience: 30
  id: 10
  transfer: 10
  retention: 10

total: 60
probe_checkpoints: [S000, S030]
probe_update_enabled: false

agent:
  config: configs/agent/canonical_react.yaml

budgets:
  episode_timeout_s: 1200
  planner_max_output_tokens: 2048
  planner_validation_retries: 2
  max_rsi_injection_tokens: 8192

  max_agent_actions:
    EB-ALFRED: 30
    EB-Habitat: 60
    SpatialWorld: 60
    AsgardBench: 60
    TVRBench: 100
```

### K1. 为什么不再有 `openeta_tool_calls`

新 agent 没有 OpenETA internal tool pipeline。

真正可解释的指标就是：

```text
planner_calls
valid environment actions
simulator_steps
```

### K2. validation retry 不算 environment action

但必须算：

```text
planner_calls
tokens
cost
```

因此 budget/report 同时输出：

```text
env_steps
planner_calls
planner_validation_retries
```

---

# 16. Phase L：Public observation allowlist

新 agent 必须继续保留当前 benchmark 的最重要安全规则。

## L1. planner 可见

仅：

```text
instruction
PublicObservation.images
PublicObservation.text_feedback
action history (public)
legal action schema
RSI context (public-history-derived)
```

## L2. planner 不可见

禁止：

```text
physical_task_id
global_task_id
source_task_id
private_eval_metadata
golden_actions
goal_preds
success_conditions
target_pose
transfer_pair_id
retention_anchor_id
hidden object states
hidden object ids (unless source official public action schema explicitly exposes them)
private evaluator measures
```

## L3. `public_metadata`

默认不要直接 dump 给 model。

只有显式 allowlist 才可：

```text
image_roles
```

其他字段只作为 diagnostic。

---

# 17. Phase M：RSI 公共接口保持不变

现有 `RSIMethod` 的思想是对的，尽量不破坏：

```python
init_run()
before_episode()
predict_sidecar()
after_step()
after_episode()
snapshot()
load_snapshot()
clone_from_snapshot()
state_hash()
set_update_enabled()
```

Probe clone 的：

```text
new instance
+ temp snapshot copy
+ update_enabled=False
```

必须原样保留。

---

# 18. Phase N：C0 `none`

## N1. 不新增任何逻辑

`NoneRSI.before_episode()`：

```python
return make_injection("")
```

`after_episode()`：no-op。

## N2. 关键 invariant

30 Experience 后：

```text
state_hash unchanged
```

Agent 自己也不得出现跨 episode state。

---

# 19. Phase O：C1 `raw_memory`

保留现有方法。

## O1. 数据源

只能保存：

```text
instruction
actions
public feedback
action_success
public outcome success/terminated
```

不要保存 planner hidden reasoning 当作 extra supervision。

## O2. before_episode

按现有方法 retrieve top-k，然后：

```python
return make_injection(retrieved_text, provenance_ids=...)
```

## O3. Agent 不知道“这是 RawMemory”

Agent 只看到统一：

```text
[PERSISTENT EXPERIENCE GUIDANCE]
...
[/PERSISTENT EXPERIENCE GUIDANCE]
```

---

# 20. Phase P：C2 ACE-Context

## P1. 保留 official ACE updater

继续使用 pinned：

```text
ACE Reflector
ACE Curator
playbook update machinery
```

不要重新写 prompt/algorithm。

## P2. 修改命名

把 docstring：

```text
ACE-Context (OpenETA executor adaptation)
```

改为：

```text
ACE-Context (Canonical ReAct executor adaptation)
```

## P3. execution path

```text
Canonical agent public trajectory
        |
        v
ACE Reflector
        |
        v
ACE Curator
        |
        v
playbook.txt
        |
next episode before_episode()
        |
        v
shared RSI injection slot
```

## P4. 不允许 ACE 改 canonical agent

ACE 不得：

- 替换 system prompt；
- 改 action schema；
- 改 recovery；
- 改 working-memory policy；
- 自动控制 simulator。

## P5. updater usage/accounting

继续保留当前 audited client proxy。

确保每个 ACE updater request：

```text
role=ace_updater
```

出现在 `public_context_dump.jsonl`。

---

# 21. Phase Q：C3 WorldMind

这是最需要严格时序的 RSI。

## Q1. 保留当前 official plugin 路径

继续使用：

```text
ProcessExperienceModule
GoalExperienceModule
ExperienceRetrievalModule
```

## Q2. 修改描述

把：

```text
with OpenETA executor
```

改为：

```text
with Canonical Multimodal ReAct executor
```

## Q3. step timing 必须严格

正确：

```text
agent sees state_before
       |
agent selects ONE action
       |
WorldMind predict_sidecar(action, state_before)
       |
NO REAL FEEDBACK YET
       |
adapter.step(action)
       |
state_after + public feedback
       |
WorldMind process_single_step(...)
```

禁止：

```text
env.step -> prediction
```

## Q4. probes

`update_enabled=False` 时：

- 可以 read retrieved WorldMind experiences；
- 不做 process/goal persistent update；
- sidecar 若只用于学习且不回流 planner，则可以继续 disabled；
- probe 前后 state hash 必须相同。

## Q5. current official semantics 必须保留

`process_single_step()` 要继续：

```text
step.observation = state_after
state_before = explicit kwarg
predicted_state = prediction
```

不要回退到旧 bug。

## Q6. WorldMind 不允许单独增加 planner 能力

它得到的 canonical planner 与 `none` 一样。

只允许：

```text
retrieved persistent guidance
+ official WorldMind learning sidecar
```

---

# 22. Phase R：C4 EmbodiSkill 激活路线

当前主 `benchmark/rsi/embodiskill.py` 仍然：

```text
BLOCKED=True
POC_READY_UNVERIFIED
```

当前已有：

```text
benchmark/adapters/workers/embodiskill_worker.py
```

它已经调用 official：

```text
init_task_context
move_skill_state
save_task_context
reflect_episode
revise_manual
get_active_manual_text
```

本次不要把 EmbodiSkill AgentKit 的 `TeamSolver` 当 canonical agent。

我们只接 official skill/manual evolution module。

## R1. 先修 worker 的硬编码路径

当前有类似：

```python
PROJECT = Path("/data/users/.../EmbodiedRSIBench/embodied_rsi")
```

改为从 `__file__` 推导：

```python
PROJECT = Path(__file__).resolve().parents[3]
```

实际层级 coding agent 必须打印/测试确认，不要盲抄数字。

## R2. 去掉 OpenETA 命名

例如：

```text
namespace="embodiskill_openeta"
task_name="embodied_openeta"
```

改为：

```text
namespace="embodiskill_canonical_react"
task_name="embodied_rsi_canonical"
```

这只是 metadata，不改算法。

## R3. 给 worker 增加 usage/audit 返回

当前 worker 的 `DeepSeekLLM` 没把 usage 返回主进程。

修改：

```python
self.usage = {prompt_tokens, completion_tokens, calls}
self.request_records = []
```

每次 LLM request 记录：

```text
model
messages
max_tokens
```

不要记录 API key。

`after_episode` response 增加：

```json
{
  "reflection": {},
  "revision": {},
  "manual_version": 3,
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "calls": 2
  },
  "request_records": []
}
```

主进程 wrapper 把这些喂给 `Accountant`。

## R4. 实现真正的 `EmbodiSkillRSI`

不能继续只返回空 guidance。

建议写一个 isolated worker client：

```python
class EmbodiSkillRSI(RSIMethod):
    name = "embodiskill"

    def init_run(self, run_ctx):
        super().init_run(run_ctx)
        self._start_worker()

    def _start_worker(self):
        # start external EmbodiSkill python env
        # pass EMBODISKILL_STATE_ROOT=self.state_dir
        ...

    def _stop_worker(self):
        ...

    def _reload_state(self):
        self._stop_worker()
        self._start_worker()

    def before_episode(self, task_public_view):
        result = self.worker.call("before_episode", {...})
        return make_injection(result["guidance"], provenance_ids=[...])

    def after_episode(self, trajectory_public, outcome_public):
        if not self._guard_update():
            return
        result = self.worker.call("after_episode", {
            "trajectory": trajectory_public,
            "outcome": outcome_public,
        })
        self._last_update_usage = result.get("usage")

    def snapshot(self, output_dir):
        self.copy_tree(self.state_dir, output_dir)

    def state_hash(self):
        return self.hash_dir(self.state_dir)
```

## R5. snapshot clone 要特别测

因为 base clone 顺序是：

```text
clone.init_run(temp)
load_snapshot(snapshot)
_reload_state()
```

`EmbodiSkillRSI._reload_state()` 必须 restart worker，让它从复制后的 temp state 读取。

否则 probe clone 会用空 state。

## R6. 什么时候才能把 BLOCKED=False

只有全部满足：

```text
- worker dependency env 实际启动成功
- before_episode 能读取 manual
- 1 个 experience 后 manual/state 有可验证变化
- snapshot -> reload hash 一致
- probe update_disabled 后 hash 不变
- official API path 没被替换
- no hidden/private input
- request accounting 完整
```

才改：

```python
BLOCKED = False
STATUS = "READY"
```

如果做不到，继续 BLOCKED。

**禁止为了“必须有五个方法”而假装 PASS。**

---

# 23. Phase S：新的 agent freeze

## S1. 新建 `benchmark/agent/freeze.py`

不要再检查 OpenETA commit。

检查 canonical agent 自己：

```text
benchmark/agent/prompt.py
benchmark/agent/backend.py
benchmark/agent/action_schema.py
benchmark/agent/working_memory.py
benchmark/agent/react_agent.py
benchmark/agent/episode_runner.py
configs/agent/canonical_react.yaml
```

输出：

```json
{
  "agent_name": "canonical_multimodal_react_v1",
  "reference": "EmbodiedBench/VLMPlanner",
  "reference_commit": "9be4...",
  "system_prompt_sha256": "...",
  "agent_config_sha256": "...",
  "critical_file_sha256": {},
  "one_action_per_turn": true,
  "cross_episode_agent_memory": false,
  "rsi_only_persistent_state": true
}
```

## S2. script

新增：

```text
scripts/03b_verify_canonical_agent.py
```

只做 deterministic checks，不调用 API。

---

# 24. Phase T：新的 model/API smoke

旧 G2 可保留作为 DeepSeek endpoint evidence，但新 canonical agent 还需要验证自己的 wire format。

不要做大测试。

`03b/06b` 之间加一个非常小的 canonical planner API smoke 或让 `06b` 第一题覆盖。

要求：

```text
- current image 真进入 request
- 返回 JSON
- action 能通过 schema validator
- accountant 收到 planner usage
- public_context_dump 没 private leak
```

一条即可。

---

# 25. Phase U：新的 cheap capability calibration（唯一的新付费 preflight）

这是为了防止再烧 360 episode 才发现 baseline 0%。

**不是新增无穷测试链。只允许这一轮 capability calibration。**

## U1. 新建 calibration manifest

从 Experience pool 中选：

```text
每 source 2 条
总共 10 条
```

要求：

- 不属于 Pilot-60 ID/Transfer/Retention probes；
- 最好也不属于 Pilot-60 30 条 Experience；
- 固定 seed；
- 保存 `manifests/agent_calibration10.json`。

## U2. 只跑 `none`

不跑 ACE/WorldMind/RawMemory。

目的只有：

> canonical agent 本身是否处在可评测 operating point？

## U3. acceptance

必须生成：

```text
outputs/canonical_calibration/CALIBRATION_REPORT.json
```

报告：

```text
success by source
valid action ratio
planner protocol failures
invalid action retries
avg env steps
avg planner calls
stop reasons
cost/tokens
```

### U3.1 硬工程 gate

```text
infra error == 0
private leakage == 0
planner protocol failure == 0
valid parsed action ratio >= 95%
```

### U3.2 capability warning/stop rule

如果：

```text
overall success <= 10%
```

则：

```text
STOP
DO NOT RUN PILOT-60
```

先诊断 model/agent/budget。

理想 operating point：

```text
roughly 20%-60% success
```

这不是论文统计，是防止 floor/ceiling。

**Coding agent 不得在 calibration 后自动接着跑 Pilot。**

---

# 26. Phase V：新的 RSI smoke

新建：

```text
scripts/07b_smoke_rsi_canonical.py
```

只有 calibration 已经证明 canonical agent 可用后才跑。

每个方法最小：

```text
2 Experience
1 ID probe
```

这里只验证：

```text
none: state unchanged
raw_memory: stores + retrieves
ACE: playbook changes
WorldMind: sidecar + component calls + state changes
EmbodiSkill: if READY, manual changes
probe: snapshot read-only
```

不要拿 smoke success rate 做方法比较。

---

# 27. Phase W：正式 Pilot-60 canonical runner

## W1. 新建脚本

```text
scripts/08b_run_pilot60_canonical.py
```

直接复用：

```text
manifests/pilot60.json
```

这样 task set 与旧 OpenETA Pilot-60 一致，可以做 historical substrate comparison。

## W2. 输出目录必须新建

```text
outputs/pilot60_canonical/
    none/
    raw_memory/
    ace_context/
    worldmind/
    embodiskill/   # only if READY
```

绝不能写：

```text
outputs/pilot60/
```

## W3. fail-closed empty state

沿用现有保护：

```text
canonical run output exists -> abort
```

除非显式 `--overwrite`，并且 overwrite 前必须 archive。

## W4. checkpoint

继续：

```text
S000
30 Experience
S030
```

Probe roles：

```text
ID 10
Transfer 10
Retention 10
```

## W5. same task ordering

所有方法：

```text
same experience stream order
same probe task ids
same probe ordering
```

---

# 28. Phase X：分析脚本

## X1. `09_analyze_pilot.py` 参数化

增加：

```bash
--root outputs/pilot60_canonical
--protocol configs/pilot60_canonical.yaml
```

默认可指向新 canonical，但必须支持：

```bash
--root outputs/pilot60
```

读取 historical OpenETA run。

## X2. 新增 report 字段

除现有：

```text
Init ID
Final ID
ID Gain
Transfer Gain
Retention
paired transitions
```

再加：

```text
planner_calls
planner_tokens
env_steps
invalid_response_retries
invalid_action_retries
avg_actions_per_success
stop_reason distribution
RSI updater tokens
WorldMind sidecar tokens
```

## X3. 按 source

保留：

```text
PILOT_RESULTS_BY_SOURCE.csv
```

这是这次 floor diagnosis 非常重要的输出。

---

# 29. Phase Y：release audit

新建：

```text
scripts/10b_audit_canonical_release.py
```

检查：

```text
canonical agent freeze PASS
adapter gate PASS
calibration engineering invariants PASS
RSI smoke PASS/BLOCKED
Pilot counts correct
probe mutation == 0
private leakage == 0
state snapshot reload == correct
all completed methods use same model
all completed methods use same prompt hash
all completed methods use same agent config hash
all completed methods use same task manifest hash
```

不要再要求：

```text
OPENETA_FREEZE == PASS
```

那属于 historical track。

---

# 30. Provenance 必须写完整

每个正式 method root 的 `provenance.json` 至少：

```json
{
  "benchmark_repo_commit": "...",
  "benchmark_version": "Core-v1.0",
  "protocol": "Pilot-60 Canonical ReAct v0.8",
  "pilot_manifest_sha256": "...",
  "seed": 20260915,

  "canonical_agent_name": "canonical_multimodal_react_v1",
  "canonical_agent_config_sha256": "...",
  "canonical_prompt_sha256": "...",
  "agent_reference_repo": "EmbodiedBench/EmbodiedBench",
  "agent_reference_commit": "9be4e980e9cd6bcb38373cd4aab7c32724bdd401",

  "model": "resolved env model",
  "model_base_url": "resolved base url",
  "temperature": 0.0,
  "planner_max_output_tokens": 2048,

  "rsi_method": "worldmind",
  "rsi_upstream_repo": "...",
  "rsi_upstream_commit": "...",

  "one_action_per_turn": true,
  "cross_episode_agent_memory": false,
  "probe_update_enabled": false,

  "start_time_utc": "..."
}
```

不要把 API key 写进去。

---

# 31. Accountant 继续复用，但增加三个计数

当前 `benchmark/runner/accounting.py` 很有价值，保留。

增加：

```text
planner_validation_retries
planner_invalid_json_count
planner_invalid_action_count
```

可在 `to_dict()` 输出：

```json
{
  "planner_calls": 12,
  "planner_input_tokens": 12345,
  "planner_output_tokens": 1000,
  "planner_invalid_json_count": 1,
  "planner_invalid_action_count": 0,
  "planner_validation_retries": 1,
  "simulator_steps": 11
}
```

注意：

```text
planner_calls >= simulator_steps
```

如果没有 validation retry，则相等。

---

# 32. Private leakage audit 继续保留

所有请求，包括：

```text
canonical_planner
ace_updater
worldmind_sidecar
worldmind_component
embodiskill_updater
```

都必须走 request audit。

如果任何 request 出现 private sentinel：

```text
episode FAIL
release FAIL
```

---

# 33. Public trajectory 的统一 schema

不要让不同 RSI 收到不同 trajectory。

统一：

```python
{
    "episode_id": ...,          # RSI state/log use；不要进 planner prompt
    "instruction": ...,
    "source_dataset": ...,
    "actions": [
        {
            "action": "Pick",
            "parameters": {...},
            "public_feedback": "...",
            "action_success": True,
        }
    ],
    "success": True,
    "terminated": True,
    "num_steps": 7,
}
```

**不要给 RSI：**

```text
planner thought
raw model response
private evaluator measures
hidden state
```

这样比较的是 experience reuse，不是把 CoT 当额外训练数据。

---

# 34. 五个 source 的特殊注意事项

## 34.1 EB-Habitat

- 必须 `reset()` 后拿动态 70-skill schema；
- 保留现有 pickle-authoritative physical signature join；
- 不回退 instruction/id fallback；
- 70 actions 全部合法暴露给 agent；
- one decision 只选一个 skill。

## 34.2 EB-ALFRED

- 保留 official high-level action abstraction；
- environment feedback 必须进入下一 planner turn；
- DISPLAY/renderer infra issue 与 agent logic 分离；
- 不把 private task success predicate放 prompt。

## 34.3 SpatialWorld

- 保留现有 `Move/Rotate/Tilt/ChangePosture/Pick/Place/ChangeState/Manipulate/EndTask`；
- `EndTask` 是合法 source action；
- agent 选择 EndTask 不等于 success，仍 private evaluate。

## 34.4 TVRBench

- 当前 view + target view 均可进入当前 turn；
- 保留 image role；
- 不把 target private identifier 暴露；
- Stop/terminal action 按 source schema。

## 34.5 AsgardBench

- 不再需要 OpenETA `tool_batch`；
- agent 每轮一个 source action；
- 参数 schema 严格 validation。

---

# 35. 必须写的 deterministic unit tests

全部使用 fake backend + fake adapter，不要调用真实 API。

## Test 1 — one action per turn

`test_agent_one_action_per_turn.py`

构造 fake model 返回：

```json
{"thought":"x","action":{"name":"Move","parameters":{"direction":"ahead"}}}
```

assert：

```text
1 planner valid decision -> exactly 1 adapter.step
```

## Test 2 — reject multi-action

模型返回：

```json
{"actions":[...,...]}
```

必须 invalid，不允许执行第一条凑合。

## Test 3 — current episode memory reset

连续跑两个 fake episode。

第二个 episode prompt 不能出现第一个 episode action/feedback。

## Test 4 — RSI slot identical skeleton

`none` 和 `raw_memory`：

除 persistent guidance 内容外，prompt section/order/hash-template 必须相同。

## Test 5 — feedback replanning

第 1 步 fake env 返回 failure feedback。

第 2 次 model request 必须包含该 public feedback。

## Test 6 — invalid JSON recovery

第一次 invalid JSON，第二次 valid。

assert：

```text
planner_calls=2
env_steps=1
validation_retries=1
```

## Test 7 — invalid action name recovery

第一次 action 不在 schema，不能触碰 env。

## Test 8 — no random fallback

连续三次 invalid planner response，assert：

```text
0 env steps
planner_protocol_failure
```

## Test 9 — dynamic tool schema

fake adapter 在 reset 前 schema empty，reset 后 schema non-empty。

runner 必须 reset first。

## Test 10 — private observation leakage

public metadata 里故意放：

```text
private_eval_metadata
target_pose
physical_task_id
```

assert planner request 不含。

## Test 11 — WorldMind timing

用 spy RSI：

```text
predict_sidecar timestamp < adapter.step timestamp < after_step timestamp
```

并 assert after_step `state_after` 是真实 post-action state。

## Test 12 — probe read-only

继续跑现有 snapshot tests。

## Test 13 — agent no cross-episode state

不只 RSI hash；还检查 agent object 每 episode 是 fresh instance 或 memory clear。

## Test 14 — request accounting

一 valid step：

```text
planner_calls=1
simulator_steps=1
request_dumps role=canonical_planner
```

## Test 15 — source termination

adapter returns `terminated=True` 后，runner 不得再 call planner。

## Test 16 — max action budget

fake env never done，达到 max_agent_actions 后 deterministic truncate。

---

# 36. 原有 tests 如何处理

继续保留并适配：

```text
test_adapter_action_equivalence.py
test_gate_status.py
test_no_private_leakage.py
test_observation_allowlist.py
test_probe_readonly.py
test_rsi_state_isolation.py
test_snapshot_reload.py
```

`test_openeta_frozen.py` 不删除，但只属于 legacy suite。

新 test suite 可以分 marker：

```text
legacy_openeta
canonical_agent
rsi_core
```

---

# 37. 不要把测试搞成无止境：明确三层验收

这次迁移只需要三层。

## Level 1：free deterministic tests

全部 fake backend。

要求全 PASS。

## Level 2：10-episode `none` capability calibration

唯一的付费 bring-up。

如果 floor，STOP。

## Level 3：RSI smoke + 正式 Pilot

只有 Level 2 可用才进入。

**禁止新增 Canary-32、Mini-Mini-Pilot、第二套 smoke 等无限层级。**

---

# 38. Git commit 顺序：coding agent 必须按这个顺序提交

不要一个 5000 行巨 commit。

## Commit M1

```text
chore: pin EmbodiedBench canonical-agent reference
```

内容：

- external_sources manifest；
- canonical_agent_reference manifest；
- legacy OpenETA note。

## Commit M2

```text
refactor: decouple RSI context types from OpenETA
```

内容：

- `benchmark/rsi/context.py`；
- all RSI imports；
- compatibility shim；
- tests。

## Commit M3

```text
feat: canonical multimodal ReAct agent core
```

内容：

```text
agent/types.py
agent/config.py
agent/prompt.py
agent/vision.py
agent/backend.py
agent/action_schema.py
agent/working_memory.py
agent/react_agent.py
```

## Commit M4

```text
feat: canonical one-action episode runner
```

内容：

- new episode runner；
- accountant integration；
- leakage；
- deterministic tests。

## Commit M5

```text
refactor: attach none raw-memory ACE WorldMind to canonical agent
```

内容：

- method docstrings；
- runner hooks；
- WorldMind timing tests。

## Commit M6

```text
feat: activate EmbodiSkill thin RSI adapter when verified
```

如果 env 未验证：

```text
chore: harden EmbodiSkill adapter and keep C4 blocked
```

绝不能谎写 activate。

## Commit M7

```text
feat: canonical agent configs gates and Pilot-60 runner
```

内容：

- configs；
- scripts 03b/06b/07b/08b/10b；
- analysis 参数化。

## Commit M8

```text
test: canonical agent deterministic acceptance suite
```

把所有 fake tests 跑通。

**到 M8 停下来。不要自动跑付费 calibration。**

---

# 39. Coding agent 每个阶段必须输出什么

每个 commit 完成后写：

```text
MIGRATION_STATUS.md
```

格式：

```markdown
# Migration Status

## Current commit
...

## Completed phases
- [x] M1
- [x] M2
- [ ] M3
...

## Tests run
- command
- PASS/FAIL

## Known blockers
...

## Files changed
...

## Expensive actions NOT run
- no real API Pilot executed
```

避免 coding agent 做完一堆事但无法追踪。

---

# 40. 新 canonical agent 的正式不变量

最终所有正式条件必须满足：

```text
C0 none
C1 raw_memory
C2 ace_context
C3 worldmind
C4 embodiskill (if READY)
```

每一个方法：

```text
same canonical agent code hash
same system prompt hash
same agent config hash
same backbone model
same temperature
same max output tokens
same action schema for same task
same working-memory policy
same validation retries
same stuck warning rule
same source budget
same task ordering
same probes
```

唯一允许不同：

```text
RSI state
RSI updater calls
RSI retrieval output
WorldMind learning sidecar
```

---

# 41. Agent 与 RSI 的所有权边界

## Agent owns

```text
current observation consumption
current-episode memory
reasoning/planning
one-action selection
JSON/action validation
replanning
fixed recovery
```

## RSI owns

```text
cross-episode memory
playbook
world/process experience
skill/manual
retrieval
reflection/update
```

## Environment owns

```text
legal action execution
public observation
public feedback
termination/truncation
private success evaluation
```

这三个边界不要混。

---

# 42. 论文方法命名建议

主 agent：

```text
Canonical Multimodal ReAct Planner–Executor Agent
```

描述：

> We use a fixed multimodal ReAct-style planner–executor architecture adapted
> from the closed-loop VLM planning paradigm of EmbodiedBench. The agent observes
> the current visual state and public environment feedback, maintains only
> episode-local working memory, selects exactly one legal environment action per
> reasoning turn, and replans after each transition. Cross-episode persistent
> state is exclusively owned by the evaluated RSI method.

方法命名：

```text
None
Raw Memory
ACE-Context (Canonical Agent adaptation)
WorldMind (Canonical Agent adaptation)
EmbodiSkill-Manual (Canonical Agent adaptation)  # only if verified
```

不要声称：

```text
original end-to-end ACE
original end-to-end EmbodiSkill agent
```

因为 executor 已统一。

---

# 43. 历史 OpenETA 结果如何处理

不要删。

标记：

```text
Historical Substrate Study: OpenETA Stage-2 adaptation
```

它可以成为论文 appendix / negative result：

```text
OpenETA-based substrate produced severe floor effects under the benchmark-native
source-action adaptation, motivating the canonical planner-executor redesign.
```

但不要和 canonical agent RSI leaderboard 混表后直接比较“谁好”。

---

# 44. 本任务书完成定义（Definition of Done）

Coding agent 只有达到下面全部条件，才能说“代码迁移完成”。

## Architecture

- [ ] `benchmark/agent/` 已建立；
- [ ] canonical runtime 不 import OpenETA；
- [ ] one planner turn == one env action；
- [ ] current-episode memory exists and clears every episode；
- [ ] feedback-conditioned replanning exists；
- [ ] random fallback does not exist；
- [ ] success comes from private evaluator, not model text。

## RSI

- [ ] RSI context moved to `benchmark/rsi/context.py`；
- [ ] none works；
- [ ] raw_memory works；
- [ ] ACE official updater preserved；
- [ ] WorldMind official process/goal/retrieval preserved；
- [ ] WorldMind prediction timing correct；
- [ ] EmbodiSkill either truly READY or honestly BLOCKED；
- [ ] probe clone remains read-only。

## Safety / fairness

- [ ] same prompt hash across methods；
- [ ] same agent config hash across methods；
- [ ] same model across methods；
- [ ] private leakage zero in tests；
- [ ] no cross-episode agent state；
- [ ] only RSI can persist state。

## Logging

- [ ] every planner request audited；
- [ ] ACE/WorldMind/EmbodiSkill updater requests audited；
- [ ] token accounting categories complete；
- [ ] action validation retry counts logged；
- [ ] stop reason logged；
- [ ] provenance complete。

## Tests

- [ ] all deterministic canonical-agent tests PASS；
- [ ] existing adapter tests PASS；
- [ ] snapshot/read-only tests PASS；
- [ ] no paid full Pilot auto-run。

---

# 45. 最后给 coding agent 的简单执行指令

如果你不知道下一步做什么，就严格按下面顺序：

```text
1. 创建 branch refactor/canonical-react-agent
2. 添加 EmbodiedBench reference pin
3. 把 RSI context 从 openeta_bridge 搬到 benchmark/rsi/context.py
4. 确保所有 RSI 不再依赖 openeta_bridge
5. 新建 benchmark/agent/types.py
6. 新建 config.py + canonical_react.yaml
7. 新建固定 prompt.py
8. 新建 vision.py
9. 新建 backend.py
10. 新建 action_schema.py
11. 新建 working_memory.py
12. 新建 react_agent.py
13. 写 deterministic tests
14. 新建 canonical episode_runner.py
15. 把 none/raw_memory/ACE/WorldMind 接进去
16. 测 WorldMind action-before-step / state-after semantics
17. 加固并尝试激活 EmbodiSkill worker；失败就保持 BLOCKED
18. 新建 agent freeze + scripts 03b/06b/07b/08b/10b
19. 参数化 09_analyze_pilot.py
20. 跑免费 unit tests
21. 写 MIGRATION_STATUS.md
22. STOP
23. 等人工确认后，才允许跑 10-episode none calibration
24. calibration 若 <=10% success，STOP，不准跑 Pilot
25. calibration operating point 可用，再人工决定正式 Pilot-60
```

---

# Appendix A：当前仓库已知迁移点清单

基于起点 commit `56ad3aa...`，coding agent 至少要检查这些旧耦合：

```text
benchmark/rsi/base.py
  imports benchmark.openeta_bridge.context_injection

benchmark/rsi/none.py
  imports benchmark.openeta_bridge.context_injection

benchmark/rsi/raw_memory.py
  imports benchmark.openeta_bridge.context_injection

benchmark/rsi/ace_context.py
  imports benchmark.openeta_bridge.context_injection

benchmark/rsi/worldmind.py
  imports benchmark.openeta_bridge.context_injection
  docstring mentions OpenETA

benchmark/rsi/embodiskill.py
  imports benchmark.openeta_bridge.context_injection
  currently BLOCKED=True

scripts/06_smoke_openeta.py
  historical only

scripts/07_smoke_rsi.py
  imports OpenETA runner; create canonical sibling

scripts/08_run_pilot60.py
  imports OpenETA runner; historical only

benchmark/openeta_bridge/episode_runner.py
  historical only

benchmark/openeta_bridge/build_runtime.py
  historical only
```

不要只改文件名，必须 grep 全 repo 确认 canonical path 没有 OpenETA import。

---

# Appendix B：建议的 grep / audit commands

```bash
# canonical RSI must not depend on OpenETA
grep -R "benchmark.openeta_bridge" -n embodied_rsi/benchmark/rsi

# canonical agent must not depend on OpenETA
grep -R "OpenETA\|openeta_bridge\|agent.runtime" -n embodied_rsi/benchmark/agent

# find private-field leakage risks
grep -R "private_eval_metadata\|golden_actions\|success_conditions\|target_pose" \
  -n embodied_rsi/benchmark/agent embodied_rsi/benchmark/rsi

# find accidental random fallbacks
grep -R "random.choice\|np.random\|random action" -n embodied_rsi/benchmark/agent

# find multi-action execution accidentally introduced
grep -R "tool_batch\|executable_plan\|for .*action.* in" -n embodied_rsi/benchmark/agent
```

上述 grep 命中不一定全部是 bug，但 coding agent 必须逐条解释。

---

# Appendix C：Source basis / reproducibility references

```text
EmbodiedRSIBench migration base:
https://github.com/Rongdingyi/EmbodiedRSIBench
56ad3aaba4fe9c0d2cc11e6903f08bde1fab5543

EmbodiedBench reference:
https://github.com/EmbodiedBench/EmbodiedBench
9be4e980e9cd6bcb38373cd4aab7c32724bdd401
embodiedbench/planner/vlm_planner.py

ACE:
https://github.com/ace-agent/ace
82709de050e1db6e6ef2f07bcb0393560b94992a

WorldMind:
https://github.com/zjunlp/WorldMind
712b0fd53b4bd6a1948603f087a1e9f25a5adaea

EmbodiSkill:
https://github.com/air-embodied-brain/EmbodiSkill
760126030eab1d33ec6a6f30988f0f1fb58df3a7

Historical OpenETA:
https://github.com/OpenMOSS/OpenETA
7d4a0a1522ba8ebbd362bde880bad81d2a98f15e
```

---

# Appendix D：一句话判断 coding agent 有没有做偏

完成后，问下面这个问题：

> 如果把 `rsi_method` 从 `none` 改成 `worldmind`，除了 RSI 自己的 persistent state / retrieval / updater / learning-sidecar 以外，canonical agent 的 planner、prompt、working memory、action validation、recovery、budget、environment interface 有没有任何变化？

正确答案必须是：

> **没有。**

只要答案不是“没有”，就还没达到这个 benchmark 的实验控制要求。
