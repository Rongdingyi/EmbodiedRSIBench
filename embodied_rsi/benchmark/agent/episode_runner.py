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
import shutil
import tempfile
import time
from pathlib import Path

from benchmark.agent.backend import CanonicalVLMBackend, PlannerProtocolError
from benchmark.agent.config import ProtocolConfig
from benchmark.agent.react_agent import CanonicalMultimodalReActAgent
from benchmark.agent.types import CanonicalEpisodeResult
from benchmark.rsi.context import make_injection
from benchmark.runner.accounting import Accountant

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
    """One canonical episode: reset -> loop(decide/step) -> private eval.

    Robustness contract (review v0.9):
      * the whole adapter lifecycle is guarded; `adapter.close()` always runs;
      * reset / tool-spec failures return a canonical result classified as
        `simulator_infrastructure_error` instead of raising;
      * experience episodes run inside an RSI transaction: a pre-episode
        snapshot is taken, any `after_step`/`after_episode` exception stops
        learning immediately, restores the snapshot and fails the episode with
        `stop_reason=rsi_update_error` (no half-written RSI state survives).
    """
    t0 = time.time()
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    accountant = accountant or Accountant(episode_dir=Path(output_dir) if output_dir else None)
    if rsi is not None:
        rsi.run_ctx["accountant"] = accountant
    if agent is None:
        agent = CanonicalMultimodalReActAgent(
            backend=CanonicalVLMBackend(config.agent.planner, accountant=accountant),
            config=config.agent,
            accountant=accountant,
        )
    if getattr(agent, "accountant", None) is None:
        agent.accountant = accountant
    backend = getattr(agent, "backend", None)
    if backend is not None and getattr(backend, "accountant", None) is None:
        backend.accountant = accountant

    from benchmark.agent.config import validate_agent_config

    validate_agent_config(config.agent)
    max_actions = int(config.max_agent_actions.get(task["source_dataset"], 60))
    episode_timeout_s = float(config.episode_timeout_s)
    max_injection_tokens = int(getattr(config.agent.rsi, "max_injection_tokens", 8192) or 8192)

    trajectory: list[dict] = []
    error: str | None = None
    stop_reason = ""
    rsi_update_error: str | None = None
    rsi_step_errors: list[str] = []
    private_eval: dict = {"success": None, "error": "evaluator unavailable"}
    tx_dir: Path | None = None
    tx_ok = False
    rollback_needed = False
    outcome: dict = {}

    try:
        # ---- adapter reset + dynamic legal actions (infrastructure-classified)
        observation = None
        tool_specs: list[dict] = []
        try:
            observation = adapter.reset(task)
            tool_specs = adapter.build_tool_specs(task) or []
            if not tool_specs:
                raise RuntimeError("adapter exposed no legal actions after reset")
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            stop_reason = "simulator_infrastructure_error"

        if error is None:
            # ---- shared RSI guidance slot
            public_task_view = {
                "global_task_id": task.get("global_task_id"),
                "instruction": observation.instruction or task.get("instruction"),
                "source_dataset": task.get("source_dataset"),
                "scene_id_canonical": task.get("scene_id_canonical"),
                "difficulty_canonical": task.get("difficulty_canonical"),
            }
            if rsi is not None:
                try:
                    injection = rsi.before_episode(public_task_view)
                except Exception as exc:  # noqa: BLE001
                    rsi_update_error = f"before_episode: {type(exc).__name__}: {exc}"
                    injection = make_injection("")
            else:
                injection = make_injection("")
            if injection.estimated_tokens > max_injection_tokens:
                rsi_update_error = (f"{rsi_update_error + '; ' if rsi_update_error else ''}"
                                    f"injection {injection.estimated_tokens} tokens exceeds "
                                    f"budget {max_injection_tokens}")
                injection = make_injection("")
            accountant.record_injection(injection.estimated_tokens)
            if output_dir is not None and rsi is not None:
                from benchmark.rsi.context import log_injection

                log_injection(injection, Path(output_dir) / "rsi_injection.jsonl",
                              episode_id=str(task.get("global_task_id")), method=rsi.name)

            agent.start_episode(instruction=observation.instruction,
                                tool_specs=tool_specs, rsi_injection=injection)

            # ---- RSI transaction: pre-episode snapshot for safe rollback
            if rsi is not None:
                try:
                    tx_dir = Path(tempfile.mkdtemp(prefix=f"rsi_tx_{rsi.name}_"))
                    rsi.snapshot(tx_dir)
                    tx_ok = True
                except Exception as exc:  # noqa: BLE001
                    tx_ok = False
                    rsi_update_error = (f"{rsi_update_error + '; ' if rsi_update_error else ''}"
                                        f"rsi_transaction_snapshot: {type(exc).__name__}: {exc}")

            # ---- closed loop: 1 planner decision = 1 adapter.step
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
                    try:
                        prediction = rsi.predict_sidecar(
                            public_observation=state_before,
                            action={"name": decision.action_name,
                                    "parameters": decision.parameters},
                            accountant=accountant,
                        )
                    except Exception as exc:  # noqa: BLE001
                        rsi_step_errors.append(f"predict_sidecar: {type(exc).__name__}: {exc}")
                        prediction = None

                try:
                    step_outcome = adapter.step(decision.action_name, decision.parameters)
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    stop_reason = "simulator_infrastructure_error"
                    break
                accountant.record_env_step()

                state_after = _public_view_of(step_outcome.observation)
                if rsi is not None:
                    try:
                        rsi.after_step({
                            "task_instruction": observation.instruction,
                            "role": role,
                            "action": {"name": decision.action_name,
                                       "parameters": decision.parameters},
                            "predicted_state": prediction,
                            "state_before": state_before,
                            "state_after": state_after,
                            "public_feedback": step_outcome.public_feedback,
                            "action_success": step_outcome.action_success,
                            "terminated": step_outcome.terminated,
                            "truncated": step_outcome.truncated,
                        })
                    except Exception as exc:  # noqa: BLE001
                        rsi_step_errors.append(f"after_step: {type(exc).__name__}: {exc}")
                        rollback_needed = True
                        break

                trajectory.append(_public_step(
                    step_idx=len(trajectory) + 1,
                    action_name=decision.action_name,
                    parameters=decision.parameters,
                    public_feedback=step_outcome.public_feedback or "",
                    action_success=step_outcome.action_success,
                    terminated=step_outcome.terminated,
                    truncated=step_outcome.truncated,
                ))
                agent.record_transition(action_name=decision.action_name,
                                        parameters=decision.parameters, outcome=step_outcome)
                observation = step_outcome.observation

                if step_outcome.terminated:
                    stop_reason = stop_reason or "source_terminated"
                    break
                if step_outcome.truncated:
                    stop_reason = stop_reason or "source_truncated"
                    break

            # ---- resolve the RSI transaction BEFORE the evaluator sees more state
            if rsi is not None:
                if rollback_needed or rsi_update_error or error:
                    if tx_ok:
                        try:
                            rsi.load_snapshot(tx_dir)
                        except Exception as exc:  # noqa: BLE001
                            rsi_update_error = (
                                f"{rsi_update_error + '; ' if rsi_update_error else ''}"
                                f"rollback_failed: {type(exc).__name__}: {exc}")
                    if error is not None:
                        outcome["rsi_update_skipped"] = "infrastructure_error"
                elif role == "experience":
                    public_episode = _public_episode(task, trajectory, {
                        "success": None, "terminated": stop_reason == "source_terminated",
                        "truncated": False, "num_steps": len(trajectory),
                    })
                    public_outcome = {"success": None,
                                      "terminated": stop_reason == "source_terminated",
                                      "truncated": False, "num_steps": len(trajectory),
                                      "env_steps": accountant.simulator_steps}
                    try:
                        rsi.after_episode(public_episode, public_outcome)
                        usage = (rsi.get_last_update_usage()
                                 if hasattr(rsi, "get_last_update_usage") else None)
                        wall_s = (rsi.get_last_update_wall_s()
                                  if hasattr(rsi, "get_last_update_wall_s") else 0.0)
                        calls = int((usage or {}).get("calls") or 0)
                        accountant.record_rsi_update(usage=usage,
                                                     wall_s=float(wall_s or 0.0), calls=calls)
                    except Exception as exc:  # noqa: BLE001
                        rsi_update_error = f"after_episode: {type(exc).__name__}: {exc}"
                        if tx_ok:
                            try:
                                rsi.load_snapshot(tx_dir)
                            except Exception as exc2:  # noqa: BLE001
                                rsi_update_error += (f"; rollback_failed: "
                                                     f"{type(exc2).__name__}: {exc2}")
            if rsi_step_errors:
                joined = "; ".join(rsi_step_errors[:3])
                rsi_update_error = f"{rsi_update_error}; {joined}" if rsi_update_error else joined

        # ---- private evaluator (guarded, adapter still open)
        if observation is None:
            private_eval = {"success": None,
                            "error": "evaluator unavailable (reset failed)"}
        else:
            try:
                private_eval = adapter.private_evaluate() or private_eval
            except Exception as exc:  # noqa: BLE001
                private_eval = {"success": None, "error": f"evaluator error: {exc}"}
    except Exception as exc:  # noqa: BLE001 - never lose the canonical result shape
        error = error or f"internal: {type(exc).__name__}: {exc}"
        stop_reason = stop_reason or "internal_error"
    finally:
        try:
            adapter.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            agent.close_episode()
        except Exception:  # noqa: BLE001
            pass
        if tx_dir is not None:
            shutil.rmtree(tx_dir, ignore_errors=True)

    success = private_eval.get("success")
    outcome.update({
        "success": success,
        "terminated": stop_reason == "source_terminated",
        "truncated": stop_reason in {"source_truncated", "max_agent_actions",
                                     "episode_timeout", "internal_error"},
        "num_steps": len(trajectory),
        "env_steps": accountant.simulator_steps,
        "status": "FAIL" if (error or rsi_update_error) else "PASS",
        "infrastructure_error": error,
        "stop_reason": stop_reason or "source_terminated",
    })
    if rsi_update_error:
        outcome["rsi_update_error"] = rsi_update_error
        # an episode whose RSI update failed is not a valid transition, no matter
        # how the environment terminated
        outcome["stop_reason"] = "rsi_update_error"

    private_refs = _private_reference_values(task)
    violations = accountant.leakage_violations(private_refs)
    accountant.write_dump()

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
        stop_reason=outcome["stop_reason"],
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
