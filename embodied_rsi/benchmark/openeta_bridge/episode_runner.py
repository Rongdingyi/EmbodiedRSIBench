"""Canonical episode runner (P0-1): upstream OpenETA loop + benchmark environment.

Flow (guide section 28 + review amendment):

    RSI.before_episode -> fresh OpenETA runtime
      -> upstream OpenEtaEpisodeRunner
         -> BenchmarkEpisodeEnvironment -> simulator worker
      -> private evaluator -> RSI.after_episode (experience only)

The runner is the pinned upstream `OpenEtaEpisodeRunner`; this module only wires
it, records the public trajectory, runs the private evaluator and closes the
episode. Nothing here re-implements completion/tool-budget/token-budget/timeout
semantics.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OPENETA = PROJECT / "external" / "OpenETA"
for _p in (str(PROJECT), str(OPENETA)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.adapters.base import BenchmarkEnvAdapter  # noqa: E402
from benchmark.openeta_bridge.benchmark_environment import (  # noqa: E402
    BenchmarkEpisodeEnvironment,
)
from benchmark.openeta_bridge.context_injection import (  # noqa: E402
    RSIInjection,
    log_injection,
    make_injection,
)
from benchmark.runner.accounting import Accountant  # noqa: E402

DEFAULTS = {
    "max_turns": 8,
    "max_tool_calls": 40,
    "timeout_s": 900.0,
    "max_total_tokens": 2_000_000,
    "recovery_turns_per_branch": 0,
    "max_recovery_turns": 0,
}


@dataclass
class EpisodeResult:
    task: dict
    public_trajectory: list[dict]
    private_evaluation: dict
    outcome: dict
    turns: int
    env_steps: int
    wall_time_s: float
    runtime_descriptor: dict
    accounting: dict = field(default_factory=dict)
    leakage_violations: list[str] = field(default_factory=list)
    error: str | None = None
    status: str = "PASS"


def run_episode(task: dict, adapter: BenchmarkEnvAdapter, rsi, *,
                role: str, config: dict | None = None,
                output_dir: Path | None = None,
                task_tools: list[dict] | None = None) -> EpisodeResult:
    from agent.runtime.episode import OpenEtaEpisodeRunner  # noqa: PLC0415
    from benchmark.openeta_bridge.build_runtime import (  # noqa: PLC0415
        build_runtime,
        runtime_descriptor,
    )

    cfg = {**DEFAULTS, **(config or {})}
    t0 = time.time()
    result_dir = output_dir
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
    accountant = Accountant(episode_dir=result_dir)
    artifact_dir = (result_dir / "artifacts") if result_dir else None
    if rsi is not None:
        # per-episode accountant, visible to RSI methods (ACE audit trail, P1-10)
        rsi.run_ctx["accountant"] = accountant

    # 2. RSI.before_episode + shared-budget validation
    public_view = {
        "global_task_id": task["global_task_id"],
        "instruction": task.get("instruction"),
        "source_dataset": task["source_dataset"],
        "scene_id_canonical": task.get("scene_id_canonical"),
        "difficulty_canonical": task.get("difficulty_canonical"),
    }
    injection: RSIInjection = (rsi.before_episode(public_view) if rsi is not None
                               else make_injection(""))
    accountant.record_injection(injection.estimated_tokens)
    if result_dir is not None:
        log_injection(injection, result_dir / "rsi_injection.jsonl",
                      episode_id=task["global_task_id"],
                      method=getattr(rsi, "name", "none"))
        (result_dir / "rsi_injection.jsonl.raw").write_text(injection.context_text)

    # 3/4/5. environment prepare -> live tool schema -> fresh runtime
    env = BenchmarkEpisodeEnvironment(
        adapter, task, rsi=rsi, accountant=accountant,
        env_step_budget=cfg.get("max_env_steps"),
        artifact_dir=artifact_dir,
    )
    error: str | None = None
    runner = None
    descriptor: dict = {}
    public_trajectory: list[dict] = []
    private_eval: dict = {}
    upstream_summary: dict = {}
    try:
        first_observation = env.prepare()
        # the environment's public instruction is authoritative (TVR has no
        # household instruction in the registry and synthesizes one)
        runner_task = (first_observation.task or task.get("instruction") or
                       task.get("global_task_id") or "")
        specs = task_tools or env.tool_specs()
        runtime = build_runtime(specs, guidance=injection.context_text,
                                guidance_provenance=injection.provenance_ids,
                                accountant=accountant,
                                planner_tokens=int(cfg.get("planner_max_output_tokens", 8192)))
        descriptor = runtime_descriptor(runtime)
        runner = OpenEtaEpisodeRunner(runtime=runtime, environment=env)
        upstream = runner.run(
            task=runner_task,
            max_turns=int(cfg["max_turns"]),
            max_tool_calls=int(cfg["max_tool_calls"]),
            timeout_s=float(cfg["timeout_s"]),
            max_total_tokens=int(cfg["max_total_tokens"]),
            recovery_turns_per_branch=int(cfg["recovery_turns_per_branch"]),
            max_recovery_turns=int(cfg["max_recovery_turns"]),
            # internal benchmark identifiers (task id / role / dataset) stay in
            # our logs; they must never enter the agent's memory or prompt.
            metadata={},
        )
        public_trajectory = _public_trajectory(upstream)
        upstream_summary = {
            "terminated": upstream.terminated,
            "truncated": upstream.truncated,
            "steps": len(upstream.steps),
            "session_id": upstream.session_id,
            "stop_reason": runner.stop_reason,
            "failure_reason": runner.failure_reason,
            "tool_calls": runner.tool_call_count,
            "runner_tokens": runner.total_tokens,
        }
        private_eval = adapter.private_evaluate()
    except Exception as exc:  # noqa: BLE001 — audits infrastructure failures honestly
        error = f"{type(exc).__name__}: {exc}"
        try:
            private_eval = adapter.private_evaluate()
        except Exception:
            private_eval = {"success": None, "error": "evaluator unavailable"}
    finally:
        try:
            env.close()
        except Exception:
            pass

    turns = len(public_trajectory)
    outcome = {
        "success": private_eval.get("success"),
        "terminated": bool(upstream_summary.get("terminated")),
        "truncated": bool(upstream_summary.get("truncated")),
        "num_steps": turns,
        "env_steps": env.env_step_count(),
        "status": "FAIL" if error else "PASS",
        "infrastructure_error": error,
        "upstream": upstream_summary,
    }

    # RSI update (experience only; probes run with update disabled by caller)
    if rsi is not None and role == "experience":
        try:
            rsi.after_episode(_public_episode(task, public_trajectory, outcome),
                              _public_outcome(outcome))
            usage = rsi.get_last_update_usage() if hasattr(rsi, "get_last_update_usage") else None
            wall_s = rsi.get_last_update_wall_s() if hasattr(rsi, "get_last_update_wall_s") else 0.0
            calls = int((usage or {}).get("calls") or 0)
            accountant.record_rsi_update(usage=usage, wall_s=float(wall_s or 0.0), calls=calls)
        except Exception as exc:  # noqa: BLE001
            outcome["rsi_update_error"] = f"{type(exc).__name__}: {exc}"
            outcome["status"] = "FAIL"

    if env.rsi_step_errors:
        outcome["rsi_update_error"] = "; ".join(env.rsi_step_errors[:3])
        outcome["status"] = "FAIL"

    # leakage audit over every model request made in this episode (P0-10/P1-10)
    private_refs = _private_reference_values(task)
    env.private_reference_values = private_refs
    violations = accountant.leakage_violations(private_refs)
    accountant.write_dump()

    result = EpisodeResult(
        task=task,
        public_trajectory=public_trajectory,
        private_evaluation=private_eval,
        outcome=outcome,
        turns=turns,
        env_steps=env.env_step_count(),
        wall_time_s=time.time() - t0,
        runtime_descriptor=descriptor,
        accounting=accountant.to_dict(),
        leakage_violations=violations,
        error=error,
        status="FAIL" if (error or violations or outcome.get("rsi_update_error")) else "PASS",
    )
    if result_dir is not None:
        (result_dir / "outcome.json").write_text(
            json.dumps(result.outcome, ensure_ascii=False, indent=2) + "\n")
        (result_dir / "token_usage.json").write_text(
            json.dumps(result.accounting, ensure_ascii=False, indent=2) + "\n")
        (result_dir / "public_trajectory.json").write_text(
            json.dumps(result.public_trajectory, ensure_ascii=False, indent=2) + "\n")
        (result_dir / "private_eval.json").write_text(
            json.dumps(result.private_evaluation, ensure_ascii=False, indent=2) + "\n")
        (result_dir / "LEAKAGE_AUDIT.json").write_text(
            json.dumps({"violations": violations, "requests": len(accountant.request_dumps),
                        "status": "PASS" if not violations else "FAIL"},
                       ensure_ascii=False, indent=2) + "\n")
    return result


# ------------------------------------------------------------------ helpers
def _private_reference_values(task: dict) -> list[str]:
    """High-precision leak sentinel values.

    Only full identifiers and the canonical serialisation of the private metadata
    are used: bare numbers (e.g. a pose component like ``0.0``) legitimately occur
    in agent-visible text and would produce false positives.
    """
    import re

    values: list[str] = []
    for key in ("physical_task_id", "source_task_id", "global_task_id"):
        value = task.get(key)
        if not isinstance(value, str) or not value:
            continue
        # only distinctive identifiers are sentinels: EB-Habitat episode ids are
        # short integers (e.g. "35") that trivially occur in ordinary text
        if len(value) < 6 or value.isdigit():
            continue
        values.append(value)
    private = task.get("private_eval_metadata") or {}
    if private:
        values.append(json.dumps(private, sort_keys=True, separators=(",", ":")))

    # identifier-like only: a plain dictionary word that also occurs in the
    # public instruction (e.g. "Microwave") must NOT be treated as private.
    identifier_like = re.compile(r"^[A-Za-z0-9_\-:.|]*[0-9_:|.\-][A-Za-z0-9_\-:.|]{6,}$")

    def walk(node):
        if isinstance(node, str):
            if identifier_like.match(node):
                values.append(node)
        elif isinstance(node, dict):
            for k, v in node.items():
                if k not in ("outcome",):
                    walk(v)
        elif isinstance(node, list):
            for item in node[:50]:
                walk(item)

    walk(private)
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique[:60]


def _public_trajectory(upstream) -> list[dict]:
    from agent.runtime.episode import action_token_usage  # noqa: PLC0415

    rows = []
    for step in upstream.steps:
        action = step.action
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request", {}) if isinstance(command, dict) else {}
        name = str(request.get("name") or "")
        parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
        info = step.step_result.info or {}
        rows.append({
            "step_idx": step.turn_index,
            "chosen_action": {"action_type": getattr(action, "action_type", ""),
                              "name": name, "parameters": parameters},
            "public_env_feedback": info.get("public_feedback") or info.get("termination_reason") or "",
            "action_success": info.get("action_success"),
            "terminated": bool(step.step_result.terminated),
            "truncated": bool(step.step_result.truncated),
            "planner_usage": _planner_usage(action),

            "rsi_aux_usage": {},
        })
    return rows


def _planner_usage(action) -> dict:
    """Token usage for one planner action, including the upstream charged total."""
    from agent.runtime.episode import action_token_usage  # noqa: PLC0415

    command = action.command if isinstance(action.command, dict) else {}
    metadata = command.get("metadata") if isinstance(command, dict) else {}
    planner_metadata = metadata.get("planner_metadata") if isinstance(metadata, dict) else {}
    backend_usage = planner_metadata.get("backend_usage") if isinstance(planner_metadata, dict) else {}
    if not isinstance(backend_usage, dict):
        backend_details = planner_metadata.get("backend_details") if isinstance(planner_metadata, dict) else {}
        backend_usage = backend_details.get("usage") if isinstance(backend_details, dict) else {}
    usage = backend_usage if isinstance(backend_usage, dict) else {}
    charged, sources = action_token_usage(action)
    return {
        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "charged_tokens": int(charged),
        "sources": dict(sources),
    }


def _public_episode(task: dict, trajectory: list[dict], outcome: dict) -> dict:
    return {
        "episode_id": task["global_task_id"],
        "instruction": task.get("instruction"),
        "source_dataset": task["source_dataset"],
        "actions": [
            {"action": s["chosen_action"].get("name"),
             "parameters": s["chosen_action"].get("parameters"),
             "public_feedback": s.get("public_env_feedback", ""),
             "action_success": s.get("action_success")}
            for s in trajectory
        ],
        "success": outcome.get("success"),
        "terminated": outcome.get("terminated"),
        "num_steps": outcome.get("num_steps"),
    }


def _public_outcome(outcome: dict) -> dict:
    return {
        "success": outcome.get("success"),
        "terminated": outcome.get("terminated"),
        "truncated": outcome.get("truncated"),
        "num_steps": outcome.get("num_steps"),
        "env_steps": outcome.get("env_steps"),
    }
