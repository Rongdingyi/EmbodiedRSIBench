"""Canonical episode runner (guide section 28) + trajectory export."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OPENETA = PROJECT / "external" / "OpenETA"
for _p in (str(PROJECT), str(OPENETA)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapter.protocol import CameraFrame, EnvObservation, RobotState  # noqa: E402

from benchmark.adapters.base import BenchmarkEnvAdapter  # noqa: E402
from benchmark.openeta_bridge.build_runtime import build_runtime, runtime_descriptor  # noqa: E402
from benchmark.openeta_bridge.context_injection import RSIInjection, log_injection, make_injection  # noqa: E402

MAX_TURNS_DEFAULT = 30


ROLE_MAP = {"current_view": "scene_primary", "target_view": "scene_secondary"}


def to_env_observation(public, *, step_idx: int, artifact_dir: Path | None = None) -> EnvObservation:
    """Build an upstream EnvObservation and materialise RGB frames as artifacts.

    The pinned planner only forwards images to the backend through
    `metadata["image_artifacts"]` paths, so every camera frame is written to
    disk here and declared with kind/frame_id/role/width/height.
    """
    from PIL import Image  # noqa: PLC0415

    cameras = []
    artifacts = []
    roles = public.public_metadata.get("image_roles") or ["current_view"] * len(public.images)
    for index, (role, frame) in enumerate(zip(roles, public.images)):
        cameras.append(CameraFrame(frame_id=role, rgb=frame.tolist(),
                                   role=ROLE_MAP.get(role, "scene_secondary")))
        path = None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            path = artifact_dir / f"step{step_idx:03d}_{index}_{role}.png"
            Image.fromarray(frame).save(path)
        artifacts.append({
            "kind": "rgb",
            "frame_id": role,
            "role": ROLE_MAP.get(role, "scene_secondary"),
            "path": str(path) if path else "",
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "format": "png",
            "index": index,
        })
    metadata = {"step_idx": step_idx, **public.public_metadata,
                "image_artifacts": [a for a in artifacts if a["path"]]}
    return EnvObservation(
        task=public.instruction,
        cameras=cameras,
        robot=RobotState(),
        objects=[],
        metadata=metadata,
    )


@dataclass
class EpisodeResult:
    task: dict
    public_trajectory: list[dict]
    private_evaluation: dict
    outcome: dict
    turns: int
    wall_time_s: float
    runtime_descriptor: dict
    error: str | None = None
    audit_requests: int = 0
    field: dict = field(default_factory=dict)


def run_episode(task: dict, adapter: BenchmarkEnvAdapter, rsi, *,
                role: str, output_dir: Path | None = None,
                max_turns: int = MAX_TURNS_DEFAULT, task_tools: list[dict] | None = None,
                exploration: bool = False) -> EpisodeResult:
    """One canonical episode: fresh session, frozen probe rules applied by caller."""
    from benchmark.openeta_bridge import build_runtime as br

    t0 = time.time()
    task_tools = task_tools or adapter.build_tool_specs(task)

    # 2. RSI.before_episode + budget validation (also for `none`, which is empty)
    public_view = {
        "global_task_id": task["global_task_id"],
        "instruction": task.get("instruction"),
        "source_dataset": task["source_dataset"],
        "scene_id_canonical": task.get("scene_id_canonical"),
        "difficulty_canonical": task.get("difficulty_canonical"),
    }
    injection: RSIInjection = rsi.before_episode(public_view) if rsi is not None else make_injection("")

    # 3/4. fresh OpenETA session with the guidance in the shared context slot
    br.REQUEST_AUDIT.clear()
    runtime = build_runtime(task_tools, guidance=injection.context_text,
                            guidance_provenance=injection.provenance_ids)
    descriptor = runtime_descriptor(runtime)

    trajectory: list[dict] = []
    error = None
    private_eval: dict = {}
    turns = 0
    try:
        # 6. env.reset
        observation = adapter.reset(task)
        runtime.start_session(task=observation.instruction,
                              metadata={"global_task_id": task["global_task_id"], "role": role})
        artifact_dir = (output_dir / "artifacts") if output_dir else (
            Path(tempfile.mkdtemp(prefix="openeta_artifacts_")))
        env_obs = to_env_observation(observation, step_idx=0, artifact_dir=artifact_dir)

        terminated = truncated = False
        while turns < max_turns and not (terminated or truncated):
            turns += 1
            action = runtime.act(env_obs)
            action_type = getattr(action, "action_type", "")
            command = getattr(action, "command", {}) or {}
            request = command.get("request", {}) if isinstance(command, dict) else {}
            name = str(request.get("name", ""))
            parameters = request.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}

            step_record = {
                "step_idx": turns,
                "planner_request_hash": "",
                "chosen_action": {"name": name, "parameters": parameters,
                                  "action_type": action_type},
                "public_env_feedback": "",
                "action_success": None,
                "terminated": False,
                "truncated": False,
            }
            if br.REQUEST_AUDIT:
                step_record["planner_usage"] = br.REQUEST_AUDIT[-1]["body"].get("messages") and {}
                step_record["planner_request_hash"] = _audit_hash(br.REQUEST_AUDIT[-1])

            if action_type != "tool_call":
                # planner response (e.g. task_complete / talk / ask_human)
                feedback = f"planner response: {name or 'response'}"
                step_record["public_env_feedback"] = feedback
                trajectory.append(step_record)
                runtime.update_memory({"type": "step_result", "public_feedback": feedback,
                                       "terminated": False, "truncated": False})
                if name in {"task_complete", "done", "stop"}:
                    terminated = True
                elif exploration and turns >= max_turns:
                    truncated = True
                env_obs = to_env_observation(observation, step_idx=turns, artifact_dir=artifact_dir)
                continue

            if name not in {t["name"] for t in task_tools}:
                feedback = f"invalid action {name!r} (not in this task's action space)"
                step_record["public_env_feedback"] = feedback
                step_record["action_success"] = False
                step_record["invalid_action"] = True
                trajectory.append(step_record)
                runtime.update_memory({"type": "step_result", "public_feedback": feedback,
                                       "terminated": False, "truncated": False})
                env_obs = to_env_observation(observation, step_idx=turns, artifact_dir=artifact_dir)
                continue

            outcome = adapter.step(name, parameters)
            observation = outcome.observation
            step_record["public_env_feedback"] = outcome.public_feedback
            step_record["action_success"] = outcome.action_success
            step_record["terminated"] = outcome.terminated
            step_record["truncated"] = outcome.truncated
            trajectory.append(step_record)
            runtime.update_memory({
                "type": "step_result",
                "public_feedback": outcome.public_feedback,
                "terminated": outcome.terminated,
                "truncated": outcome.truncated,
                "action_success": outcome.action_success,
            })
            terminated = outcome.terminated
            truncated = outcome.truncated
            env_obs = to_env_observation(observation, step_idx=turns, artifact_dir=artifact_dir)

        if not terminated and not truncated:
            truncated = True
        private_eval = adapter.private_evaluate()
    except Exception as exc:  # noqa: BLE001 — infrastructure failures are audited
        error = f"{type(exc).__name__}: {exc}"
        try:
            private_eval = adapter.private_evaluate()
        except Exception:
            private_eval = {"success": None, "error": "evaluator unavailable"}
    finally:
        try:
            adapter.close()
        except Exception:
            pass

    outcome = {
        "success": private_eval.get("success"),
        "terminated": bool(locals().get("terminated")),
        "truncated": bool(locals().get("truncated")),
        "num_steps": turns,
        "infrastructure_error": error,
    }

    # 10. experience update (probe callers disable updates via set_update_enabled)
    if rsi is not None and role == "experience":
        try:
            rsi.after_episode(_public_episode(task, trajectory, outcome),
                              _public_outcome(outcome))
        except Exception as exc:  # noqa: BLE001
            outcome["rsi_update_error"] = f"{type(exc).__name__}: {exc}"

    return EpisodeResult(
        task=task,
        public_trajectory=trajectory,
        private_evaluation=private_eval,
        outcome=outcome,
        turns=turns,
        wall_time_s=time.time() - t0,
        runtime_descriptor=descriptor,
        error=error,
        audit_requests=len(br.REQUEST_AUDIT),
    )


def _audit_hash(audit: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(audit.get("body", {}), sort_keys=True).encode()).hexdigest()[:24]


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
    }
