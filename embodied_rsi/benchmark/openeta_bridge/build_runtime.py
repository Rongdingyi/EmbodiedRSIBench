"""Bridge runtime: pinned OpenETA runtime configured for benchmark source actions.

The planner prompt, planner loop, pipeline, action interfaces and backends are
the pinned upstream implementations. This module only:
  * registers the benchmark's public source actions as tools,
  * points the planner backend at DeepSeek `deepseek-flash`,
  * disables OpenETA's native self-improvement,
  * removes every tool the source benchmarks do not expose.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OPENETA = PROJECT / "external" / "OpenETA"
if str(OPENETA) not in sys.path:
    sys.path.insert(0, str(OPENETA))

from dataclasses import replace  # noqa: E402

from agent.backends.planner import (  # noqa: E402
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
)
from agent.runtime.planner import PlannerContextConfig, ToolCallingPlanner  # noqa: E402
from agent.runtime.runtime import OpenEtaAgentRuntime  # noqa: E402
from agent.runtime.skills import SkillRegistry  # noqa: E402
from agent.tools.registry import ToolEffect, ToolRegistry, ToolSpec  # noqa: E402

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
PLANNER_CONTEXT = PlannerContextConfig(
    recent_conversation_action_groups=4,
    recent_transition_observations=3,
    max_selected_skills=3,
    max_skill_content_chars=8000,
    auto_compact_enabled=True,
)

# audit sink: every outgoing planner request body lands here (per episode)
REQUEST_AUDIT: list[dict] = []


def _make_audited_transport(accountant=None):
    """Transport that records every model request (P1-6/P1-4)."""
    import time as _time

    from agent.backends.planner import _post_json

    def transport(url, body, headers, timeout_s):
        REQUEST_AUDIT.append({"url": url, "body": body})
        started = _time.time()
        response = None
        try:
            response = _post_json(url, body, headers, timeout_s)
            return response
        finally:
            if accountant is not None:
                accountant.record_planner_exchange(url=url, body=body, response=response,
                                                   wall_s=_time.time() - started)

    return transport


def deepseek_backend(*, max_tokens: int = 8192, accountant=None) -> OpenAICompatiblePlannerBackend:
    config = OpenAICompatiblePlannerBackendConfig(
        provider="openai-compatible",
        model=DEEPSEEK_MODEL,
        api_base=DEEPSEEK_BASE_URL,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        timeout_s=180,
        max_attempts=3,
        retry_backoff_s=0.5,
        temperature=0.0,
        max_tokens=max_tokens,
        use_json_response_format=False,
        enable_thinking=None,          # MUST stay None (guide section 3.2)
        enable_vision=True,
        max_vision_images=2,
    )
    missing = config.missing_fields()
    if missing:
        raise RuntimeError(f"DeepSeek planner config incomplete: {missing}")
    return OpenAICompatiblePlannerBackend(config, transport=_make_audited_transport(accountant))


def _noop_handler(context):  # tool calls become pending EnvActions for the host
    return None


class BenchmarkRuntime(OpenEtaAgentRuntime):
    """Pinned runtime with a benchmark-only tool registry.

    The pinned constructor binds handlers for OpenETA's native tool set
    (memory tools, python_exec, SAM3/AnyGrasp helpers). Canonical benchmark runs
    expose only source-benchmark actions, so those bindings are skipped. The
    planner, pipeline, interfaces, backends and episode loop remain upstream.
    """

    def _bind_memory_tool_handlers(self) -> None:  # noqa: D401 - deliberate no-op
        return None


def build_benchmark_tools(tool_specs: list[dict]) -> ToolRegistry:
    """ToolRegistry containing exactly the source benchmark actions."""
    registry = ToolRegistry()
    for spec in tool_specs:
        name = spec["name"]
        effect = ToolEffect.PLANNING if name in {"done", "stop"} else ToolEffect.WORLD_MUTATING
        registry.register(
            ToolSpec(
                name=name,
                description=spec.get("description", ""),
                category="benchmark_source_action",
                parameters=spec.get("parameters") or {"type": "object", "properties": {}},
                effect=effect,
            ),
            handler=_noop_handler,
        )
    return registry


def build_runtime(tool_specs: list[dict], *, guidance: str = "",
                  guidance_provenance: list[str] | None = None,
                  planner_tokens: int = 8192,
                  accountant=None) -> OpenEtaAgentRuntime:
    """Assemble one frozen runtime for one episode (fresh session per episode)."""
    from benchmark.openeta_bridge.freeze import (
        assert_native_self_improvement_disabled,
        disable_native_self_improvement,
        planner_prompt,
    )

    tools = build_benchmark_tools(tool_specs)
    system_prompt, _hash = planner_prompt()
    planner = ToolCallingPlanner(
        backend=deepseek_backend(max_tokens=planner_tokens, accountant=accountant),
        system_prompt=system_prompt,
        context_config=PLANNER_CONTEXT,
        max_validation_retries=2,
    )
    startup_facts = {}
    if guidance:
        startup_facts["persistent_experience_guidance"] = {
            "text": guidance,
            "provenance_ids": list(guidance_provenance or []),
            "estimated_tokens": max(1, len(guidance) // 4),
        }
    runtime = BenchmarkRuntime(
        planner=planner,
        tools=tools,
        skills=SkillRegistry(),          # no built-in skills in canonical runs
        rollout_enabled=False,           # trajectories are exported by the runner
        startup_facts=startup_facts,
    )
    disable_native_self_improvement(runtime)
    assert_native_self_improvement_disabled(runtime)
    return runtime


def forbidden_tool_names() -> set[str]:
    """Tools the guide forbids opening to the model (section 9.3)."""
    return {
        "web_search", "web_fetch", "python_exec", "sam3", "anygrasp", "anyplace",
        "graspgenx", "molmopoint", "depth_prior", "observe", "sense", "code_policy",
    }


def assert_tool_policy(registry: ToolRegistry) -> None:
    names = {tool.name for tool in registry.list()}
    bad = names & forbidden_tool_names()
    if bad:
        raise RuntimeError(f"forbidden tools exposed to the planner: {sorted(bad)}")


def runtime_descriptor(runtime: OpenEtaAgentRuntime) -> dict:
    reviewer = runtime.self_improvement_reviewer
    return {
        "planner": type(runtime.planner).__name__,
        "backend": type(getattr(runtime.planner, "backend", None)).__name__,
        "backend_model": getattr(getattr(runtime.planner, "backend", None), "config", None) and
                         runtime.planner.backend.config.model,
        "enable_thinking": getattr(getattr(runtime.planner, "backend", None), "config", None) and
                           runtime.planner.backend.config.enable_thinking,
        "tools": sorted(tool.name for tool in runtime.tools.list()),
        "skills": sorted(skill.name for skill in runtime.skills.list()),
        "native_self_improvement_enabled": bool(reviewer.config.enabled),
        "native_auto_apply_reviewed": bool(reviewer.config.auto_apply_reviewed),
        "auto_applier": reviewer.auto_applier is not None,
    }
