"""Canonical one-action episode runner (taskbook v0.8, Phase J).

`1 planner decision = 1 real adapter.step()`. The canonical runner does not
import the historical OpenETA bridge; ordering, accounting, leakage auditing
and RSI hook timing are owned here.

Stop reasons (J5): planner_protocol_failure, simulator_infrastructure_error,
rsi_update_error, max_agent_actions, source_terminated, source_truncated,
episode_timeout. Success always comes from the source private evaluator.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from benchmark.agent.backend import PlannerProtocolError
from benchmark.agent.config import ProtocolConfig
from benchmark.agent.react_agent import CanonicalMultimodalReActAgent
from benchmark.agent.types import CanonicalEpisodeResult
from benchmark.rsi.context import make_injection

PUBLIC_METADATA_ALLOWLIST = {"image_roles"}


# --------------------------------------------------------------------- helpers
def _public_view_of(public) -> dict:
    if public is None:
        return {}
    return {
        "instruction": public.instruction,
        "text_feedback": public.text_feedback,
        "public_metadata": {k: v for k, v in (public.public_metadata or {}).items()
                            if k in PUBLIC_METADATA_ALLOWLIST},
    }


def _private_reference_values(task: dict) -> list[str]:
    """High-precision leak sentinel values (same policy as the historical track)."""
    values: list[str] = []
    for key in ("physical_task_id", "source_task_id", "global_task_id"):
        value = task.get(key)
        if not isinstance(value, str) or not value:
            continue
        if len(value) < 6 or value.isdigit():
            continue
        values.append(value)
    private = task.get("private_eval_metadata") or {}
    if private:
        values.append(json.dumps(private, sort_keys=True, separators=(",", ":")))
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
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique[:60]


def _public_step(*, step_idx: int, action_name: str, parameters: dict,
                 public_feedback: str, action_success, terminated: bool,
                 truncated: bool) -> dict:
    return {
        "step_idx": step_idx,
        "chosen_action": {"action_type": "tool_call", "name": action_name,
                          "parameters": parameters},
        "public_env_feedback": public_feedback,
        "action_success": action_success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
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


# ---------------------------------------------------------------------- runner
def run_episode(task: dict, adapter, rsi, *, role: str, config: ProtocolConfig,
                output_dir: Path | None = None, agent=None, accountant=None):
    """One canonical episode: reset -> loop(decide/step) -> private eval."""
    from benchmark.agent.backend import CanonicalVLMBackend
    from benchmark.runner.accounting import Accountant

    t0 = time.time()
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    accountant = accountant or Accountant(episode_dir=Path(output_dir) if output_dir else None)
    if rsi is not None:
        rsi.run_ctx["accountant"] = accountant
    agent = agent or CanonicalMultimodalReActAgent(
        backend=CanonicalVLMBackend(config.agent.planner, accountant=accountant),
        config=config.agent,
        accountant=accountant,
    )
    # the runner owns the accountant: wire injected agents/backends into it so
    # validation-retry and token counters never silently disappear
    if getattr(agent, "accountant", None) is None:
        agent.accountant = accountant
    backend = getattr(agent, "backend", None)
    if backend is not None and getattr(backend, "accountant", None) is None:
        backend.accountant = accountant

    observation = adapter.reset(task)
    tool_specs = adapter.build_tool_specs(task) or []
    assert tool_specs, "no legal actions exposed after reset"

    public_task_view = {
        "global_task_id": task.get("global_task_id"),
        "instruction": observation.instruction or task.get("instruction"),
        "source_dataset": task.get("source_dataset"),
        "scene_id_canonical": task.get("scene_id_canonical"),
        "difficulty_canonical": task.get("difficulty_canonical"),
    }
    injection = rsi.before_episode(public_task_view) if rsi is not None else make_injection("")
    accountant.record_injection(injection.estimated_tokens)
    if output_dir is not None and rsi is not None:
        from benchmark.rsi.context import log_injection

        log_injection(injection, Path(output_dir) / "rsi_injection.jsonl",
                      episode_id=str(task.get("global_task_id")), method=rsi.name)

    agent.start_episode(instruction=observation.instruction,
                        tool_specs=tool_specs, rsi_injection=injection)

    max_actions = int(config.max_agent_actions.get(task["source_dataset"], 60))
    episode_timeout_s = float(config.episode_timeout_s)
    trajectory: list[dict] = []
    error: str | None = None
    stop_reason = ""
    rsi_step_errors: list[str] = []

    while True:
        if len(trajectory) >= max_actions:
            stop_reason = "max_agent_actions"
            break
        if time.time() - t0 > episode_timeout_s:
            stop_reason = "episode_timeout"
            break

        state_before = _public_view_of(observation)
        try:
            decision = agent.decide(observation)
        except PlannerProtocolError as exc:
            error = f"PlannerProtocolError: {exc}"
            stop_reason = "planner_protocol_failure"
            break

        prediction = None
        if rsi is not None:
            prediction = rsi.predict_sidecar(
                public_observation=state_before,
                action={"name": decision.action_name, "parameters": decision.parameters},
                accountant=accountant,
            )

        try:
            outcome = adapter.step(decision.action_name, decision.parameters)
        except Exception as exc:  # noqa: BLE001 - simulator failures are infra errors
            error = f"{type(exc).__name__}: {exc}"
            stop_reason = "simulator_infrastructure_error"
            break
        accountant.record_env_step()

        state_after = _public_view_of(outcome.observation)
        if rsi is not None:
            try:
                rsi.after_step({
                    "task_instruction": observation.instruction,
                    "role": role,
                    "action": {"name": decision.action_name, "parameters": decision.parameters},
                    "predicted_state": prediction,
                    "state_before": state_before,
                    "state_after": state_after,
                    "public_feedback": outcome.public_feedback,
                    "action_success": outcome.action_success,
                    "terminated": outcome.terminated,
                    "truncated": outcome.truncated,
                })
            except Exception as exc:  # noqa: BLE001 - surfaces as rsi_update_error
                rsi_step_errors.append(f"after_step: {type(exc).__name__}: {exc}")

        trajectory.append(_public_step(
            step_idx=len(trajectory) + 1,
            action_name=decision.action_name,
            parameters=decision.parameters,
            public_feedback=outcome.public_feedback or "",
            action_success=outcome.action_success,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
        ))
        agent.record_transition(action_name=decision.action_name,
                                parameters=decision.parameters, outcome=outcome)
        observation = outcome.observation

        if outcome.terminated:
            stop_reason = stop_reason or "source_terminated"
            break
        if outcome.truncated:
            stop_reason = stop_reason or "source_truncated"
            break

    try:
        agent.close_episode()
    except Exception:  # noqa: BLE001
        pass

    private_eval: dict = {"success": None, "error": "evaluator unavailable"}
    try:
        private_eval = adapter.private_evaluate() or private_eval
    except Exception as exc:  # noqa: BLE001
        private_eval = {"success": None, "error": f"evaluator error: {exc}"}

    outcome = {
        "success": private_eval.get("success"),
        "terminated": stop_reason == "source_terminated",
        "truncated": stop_reason in {"source_truncated", "max_agent_actions", "episode_timeout"},
        "num_steps": len(trajectory),
        "env_steps": accountant.simulator_steps,
        "status": "FAIL" if error else "PASS",
        "infrastructure_error": error,
        "stop_reason": stop_reason,
    }

    rsi_update_error: str | None = None
    if rsi is not None and role == "experience":
        if error:
            # infrastructure-failed episodes are not valid learning signals
            outcome["rsi_update_skipped"] = "infrastructure_error"
        else:
            try:
                rsi.after_episode(_public_episode(task, trajectory, outcome),
                                  _public_outcome(outcome))
                usage = rsi.get_last_update_usage() if hasattr(rsi, "get_last_update_usage") else None
                wall_s = rsi.get_last_update_wall_s() if hasattr(rsi, "get_last_update_wall_s") else 0.0
                calls = int((usage or {}).get("calls") or 0)
                accountant.record_rsi_update(usage=usage, wall_s=float(wall_s or 0.0), calls=calls)
            except Exception as exc:  # noqa: BLE001
                rsi_update_error = f"{type(exc).__name__}: {exc}"
    if rsi_step_errors:
        rsi_update_error = "; ".join(rsi_step_errors[:3])
    if rsi_update_error:
        outcome["rsi_update_error"] = rsi_update_error
        outcome["status"] = "FAIL"
        stop_reason = stop_reason or "rsi_update_error"

    private_refs = _private_reference_values(task)
    violations = accountant.leakage_violations(private_refs)
    accountant.write_dump()
    try:
        adapter.close()
    except Exception:  # noqa: BLE001
        pass

    status = "FAIL" if (error or violations or rsi_update_error) else "PASS"
    result = CanonicalEpisodeResult(
        task=task,
        public_trajectory=trajectory,
        private_evaluation=private_eval,
        outcome=outcome,
        env_steps=accountant.simulator_steps,
        planner_calls=accountant.planner.calls,
        wall_time_s=time.time() - t0,
        accounting=accountant.to_dict(),
        leakage_violations=violations,
        status=status,
        error=error,
        stop_reason=stop_reason or "source_terminated",
        rsi_update_error=rsi_update_error,
    )
    if output_dir is not None:
        out = Path(output_dir)
        (out / "outcome.json").write_text(json.dumps(outcome, indent=2, default=str) + "\n")
        (out / "public_trajectory.json").write_text(
            json.dumps(trajectory, indent=2, ensure_ascii=False, default=str) + "\n")
        (out / "private_eval.json").write_text(json.dumps(private_eval, indent=2, default=str) + "\n")
        (out / "token_usage.json").write_text(json.dumps(accountant.to_dict(), indent=2) + "\n")
        (out / "LEAKAGE_AUDIT.json").write_text(json.dumps(
            {"violations": violations, "requests": len(accountant.request_dumps),
             "status": "FAIL" if violations else "PASS"}, indent=2) + "\n")
    return result
