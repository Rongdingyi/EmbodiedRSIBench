"""Benchmark episode environment for the upstream OpenETA episode runner (P0-1).

This is the ONLY place where the benchmark touches the simulator boundary.
Upstream `OpenEtaEpisodeRunner` owns the closed loop (session start, turn
budgets, tool/token budgets, timeout, environment receipts, recovery, terminal
responses); this environment translates an `EnvAction` tool call into a source
benchmark step and returns an `EnvObservation`.

WorldMind's prediction sidecar runs here: after OpenETA has locked the action
and before `adapter.step()` — i.e. before any real feedback exists (P0-2).
RSI `after_step` hooks also run here, still before the runner commits the turn.
"""
from __future__ import annotations

from pathlib import Path

from adapter.protocol import (
    CameraFrame,
    EnvAction,
    EnvObservation,
    JsonDict,
    RobotState,
    StepResult,
)

from benchmark.adapters.base import BenchmarkEnvAdapter

# model-visible observation allowlist (P1-3): instruction, RGB, official feedback
PUBLIC_METADATA_ALLOWLIST = {"image_roles"}

# The upstream planner attaches only primary cameras (scene_primary /
# wrist_primary). The guide requires the TVR target image to reach the model
# (sections 11.1/1375), so the target view must be a second primary camera;
# evidence_id/frame_id ("target_view") keeps it distinguishable.
ROLE_MAP = {"current_view": "scene_primary", "target_view": "scene_primary"}


class BenchmarkEpisodeEnvironment:
    """EpisodeEnvironment implementation over a `BenchmarkEnvAdapter`."""

    def __init__(self, adapter: BenchmarkEnvAdapter, task_record: dict, *,
                 rsi=None, accountant=None, env_step_budget: int | None = None,
                 artifact_dir: Path | None = None, public_object_list: bool = False):
        self.adapter = adapter
        self.task = task_record
        self.rsi = rsi
        self.accountant = accountant
        self.env_step_budget = env_step_budget
        self.artifact_dir = artifact_dir
        self.public_object_list = public_object_list
        self.env_steps = 0
        self._prepared: EnvObservation | None = None
        self._last_public = None
        self._closed = False
        self.private_reference_values: list[str] = []
        self.rsi_step_errors: list[str] = []

    # ---------------------------------------------------- two-phase preparation
    def prepare(self) -> EnvObservation:
        """Real simulator reset performed before the runtime/session exists.

        Needed so the tool schema reflects the live episode (EB-Habitat skills
        are per-episode). The subsequent `reset()` from the upstream runner
        reuses this observation instead of resetting the simulator again.
        """
        observation = self.adapter.reset(self.task)
        self._last_public = observation
        self._prepared = self._to_env_observation(observation, step_idx=0)
        return self._prepared

    def tool_specs(self) -> list[dict]:
        return self.adapter.build_tool_specs(self.task)

    # ----------------------------------------------------------- EpisodeEnvironment
    def reset(self, *, task: str, metadata: JsonDict | None = None) -> EnvObservation:
        if self._prepared is not None:
            observation = self._prepared
            self._prepared = None
            return observation
        observation = self.adapter.reset(self.task)
        self._last_public = observation
        return self._to_env_observation(observation, step_idx=0)

    def step(self, action: EnvAction) -> StepResult:
        request = (action.command or {}).get("request", {})
        name = str(request.get("name") or "")
        parameters = request.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        calls = parameters.get("calls")
        if name in {"batch", "tool_batch"} or isinstance(calls, list):
            batch = [call for call in (calls or []) if isinstance(call, dict)]
            if batch:
                return self._step_batch(batch)
        return self._step_once(name, parameters)

    def _step_batch(self, calls: list[dict]) -> StepResult:
        """Execute a planner `tool_batch` as sequential environment steps.

        Official semantic (agent/tools/sim_mcp.py): a batch is one runtime act
        and terminal evidence inside it is absorbing, so execution stops at the
        first terminating sub-call and the last executed observation is
        returned for the turn.
        """
        result: StepResult | None = None
        for call in calls:
            name = str(call.get("name") or "")
            parameters = call.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {"value": parameters}
            result = self._step_once(name, parameters)
            if result.terminated or result.truncated:
                break
        if result is None:
            raise ValueError("empty tool_batch action")
        return result

    def _step_once(self, name: str, parameters: dict) -> StepResult:
        # ---- WorldMind sidecar: action locked, feedback not yet observed
        state_before = self._public_view()
        sidecar = None
        if self.rsi is not None and hasattr(self.rsi, "predict_sidecar"):
            sidecar = self.rsi.predict_sidecar(
                public_observation=state_before,
                action={"name": name, "parameters": parameters},
                accountant=self.accountant,
            )

        outcome = self.adapter.step(name, parameters)
        self.env_steps += 1
        if self.accountant is not None:
            self.accountant.record_env_step()

        # ---- RSI step hook: real instruction + before/after public states (P0-4)
        if self.rsi is not None:
            state_after = self._public_view_of(outcome.observation)
            record = {
                "task_instruction": self._last_public.instruction if self._last_public else "",
                "role": self.task.get("role"),
                "action": {"name": name, "parameters": parameters},
                "predicted_state": sidecar,
                "state_before": state_before,
                "state_after": state_after,
                "public_feedback": outcome.public_feedback,
                "action_success": outcome.action_success,
                "terminated": outcome.terminated,
                "truncated": outcome.truncated,
            }
            try:
                self.rsi.after_step(record)
            except Exception as exc:  # noqa: BLE001 - surfaces as rsi_update_error
                self.rsi_step_errors.append(
                    f"after_step: {type(exc).__name__}: {exc}")

        terminated = bool(outcome.terminated)
        truncated = bool(outcome.truncated)
        termination_reason = ""
        if self.env_step_budget is not None and self.env_steps >= self.env_step_budget \
                and not terminated:
            truncated = True
            termination_reason = "env_step_budget"
        self._last_public = outcome.observation
        observation = self._to_env_observation(outcome.observation, step_idx=self.env_steps)
        reward = float(outcome.reward_public) if outcome.reward_public is not None else 0.0
        info: JsonDict = {
            "environment_receipt_trusted": True,
            "env_step": self.env_steps,
            "action_success": outcome.action_success,
            "public_feedback": outcome.public_feedback,
            "truncation_source": "environment" if truncated else "",
            "truncation_reason": termination_reason,
        }
        return StepResult(observation=observation, reward=reward, terminated=terminated,
                          truncated=truncated, info=info)

    def close(self) -> dict:
        if not self._closed:
            self._closed = True
            try:
                self.adapter.close()
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
        return {"ok": True}

    # ------------------------------------------------------------------ helpers
    def env_step_count(self) -> int:
        return self.env_steps

    def _public_view(self) -> dict:
        return self._public_view_of(self._last_public)

    @staticmethod
    def _public_view_of(public) -> dict:
        if public is None:
            return {}
        return {"instruction": public.instruction,
                "text_feedback": public.text_feedback,
                "public_metadata": {k: v for k, v in public.public_metadata.items()
                                    if k in PUBLIC_METADATA_ALLOWLIST}}

    def _to_env_observation(self, public, *, step_idx: int) -> EnvObservation:
        from PIL import Image  # noqa: PLC0415

        roles = public.public_metadata.get("image_roles") or \
            ["current_view"] * len(public.images)
        cameras, artifacts = [], []
        for index, (role, frame) in enumerate(zip(roles, public.images)):
            cameras.append(CameraFrame(frame_id=role, rgb=frame.tolist(),
                                       role=ROLE_MAP.get(role, "scene_secondary")))
            path = None
            if self.artifact_dir is not None:
                self.artifact_dir.mkdir(parents=True, exist_ok=True)
                path = self.artifact_dir / f"step{step_idx:03d}_{index}_{role}.png"
                Image.fromarray(frame).save(path)
            artifacts.append({
                "kind": "rgb", "frame_id": role,
                "role": ROLE_MAP.get(role, "scene_secondary"),
                "path": str(path) if path else "",
                "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                "format": "png", "index": index,
            })
        metadata = {
            "step_idx": step_idx,
            "image_artifacts": [a for a in artifacts if a["path"]],
        }
        # strict allowlist: only explicitly public metadata keys pass through
        for key in PUBLIC_METADATA_ALLOWLIST:
            if key in public.public_metadata:
                metadata[key] = public.public_metadata[key]
        if self.public_object_list and "visible_object_types" in public.public_metadata:
            metadata["visible_object_types"] = public.public_metadata["visible_object_types"]
        return EnvObservation(task=public.instruction, cameras=cameras,
                              robot=RobotState(), objects=[], metadata=metadata)
