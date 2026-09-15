"""Condition D: WorldMind (Goal + Process Experience) with OpenETA executor.

Official `Plugin/worldmind_plugin` modules:
  ProcessExperienceModule / GoalExperienceModule / ExperienceRetrievalModule.

Prediction timing (P0-2, guide section 20.2): the prediction sidecar runs via
`predict_sidecar()` AFTER OpenETA locks the action and BEFORE the environment
executes it. `after_step()` then feeds the real public feedback into
`process_single_step()` so the module compares predicted vs actual state. The
sidecar output never returns to the planner.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from benchmark.openeta_bridge.context_injection import RSIInjection, count_tokens, make_injection
from benchmark.rsi.base import RSIMethod

PROJECT = Path(__file__).resolve().parents[2]
PLUGIN = PROJECT / "external" / "WorldMind" / "Plugin"
if str(PLUGIN) not in sys.path:
    sys.path.insert(0, str(PLUGIN))

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
MAX_INJECTION_TOKENS = 8192


def _sidecar_predict(observation_text: str, action_text: str) -> dict:
    """Independent DeepSeek call predicting the abstract next state."""
    prompt = (
        "You are a world-model helper. Given the current observation summary and the action "
        "the robot is about to take, predict the abstract next state in one short sentence.\n"
        f"Observation: {observation_text[:600]}\nAction: {action_text}\n"
        "Reply with the predicted next state only."
    )
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }
    request = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', '')}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode())
    message = payload["choices"][0]["message"]
    text = (message.get("content") or message.get("reasoning_content") or "").strip()
    return {"text": text, "usage": payload.get("usage", {}), "wall_s": time.time() - t0}


class WorldMindRSI(RSIMethod):
    name = "worldmind"

    def __init__(self) -> None:
        super().__init__()
        self.config = None
        self.process_module = None
        self.goal_module = None
        self.retrieval_module = None
        self._pending_prediction: str | None = None
        self._pending_action: dict | None = None
        self._sidecar_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self._sidecar_wall_s = 0.0
        self._goal_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    # ------------------------------------------------------------------ setup
    def _build_modules(self) -> None:
        from worldmind_plugin.config import WorldMindConfig
        from worldmind_plugin.core import (
            ExperienceRetrievalModule,
            GoalExperienceModule,
            ProcessExperienceModule,
        )

        self.config = WorldMindConfig(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            api_base=BASE_URL,
            is_multimodal=False,
            discriminator_model=MODEL,
            reflector_model=MODEL,
            summarizer_model=MODEL,
            extractor_model=MODEL,
            refiner_model=MODEL,
            enable_experience_refine=True,
            goal_experience_top_k=3,
            process_experience_top_k=5,
            save_path=str(self.state_dir),
            detailed_output=False,
        )
        self.process_module = ProcessExperienceModule(self.config)
        self.goal_module = GoalExperienceModule(self.config)
        self.retrieval_module = ExperienceRetrievalModule(self.config)

    def init_run(self, run_ctx: dict) -> None:
        super().init_run(run_ctx)
        self._build_modules()

    def _reload_state(self) -> None:
        self._build_modules()

    # ------------------------------------------------------------------ hooks
    def before_episode(self, task_public_view: dict) -> RSIInjection:
        self._pending_prediction = None
        self._pending_action = None
        try:
            result = self.retrieval_module.retrieve(
                task_instruction=task_public_view.get("instruction") or "",
                enable_refine=True,
            )
        except Exception:  # noqa: BLE001 — retrieval failure must not stop the episode
            return make_injection("")
        text = (result or {}).get("formatted_prompt") or ""
        if not text.strip():
            return make_injection("")
        if count_tokens(text) > MAX_INJECTION_TOKENS:
            text = text[: MAX_INJECTION_TOKENS * 3]
        return make_injection(text, provenance_ids=(result or {}).get("experience_ids") or [])

    def predict_sidecar(self, *, public_observation: dict, action: dict,
                        accountant=None) -> str | None:
        """Called by the environment after the action is locked (no feedback yet)."""
        if not self._guard_update():
            return None
        observation_text = (
            f"instruction: {public_observation.get('instruction', '')}; "
            f"last feedback: {public_observation.get('text_feedback', '')}"
        )
        action_text = f"{action.get('name')} {json.dumps(action.get('parameters') or {})}".strip()
        try:
            result = _sidecar_predict(observation_text, action_text)
        except Exception as exc:  # noqa: BLE001
            self._log({"sidecar_error": f"{type(exc).__name__}: {exc}"})
            return None
        usage = result.get("usage") or {}
        self._sidecar_usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self._sidecar_usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        self._sidecar_usage["calls"] += 1
        self._sidecar_wall_s += float(result.get("wall_s") or 0.0)
        if accountant is not None:
            accountant.record_sidecar(usage=usage, wall_s=float(result.get("wall_s") or 0.0))
        self._pending_prediction = result.get("text") or ""
        self._pending_action = dict(action)
        return self._pending_prediction

    def after_step(self, step_public_record: dict) -> None:
        if not self._guard_update() or self._pending_prediction is None:
            self._pending_prediction = None
            return
        from worldmind_plugin.core import ProcessTrajectoryStep

        prediction = self._pending_prediction
        action = self._pending_action or {}
        self._pending_prediction = None
        self._pending_action = None
        action_text = f"{action.get('name')} {json.dumps(action.get('parameters') or {})}".strip()
        observation_text = f"instruction: {step_public_record.get('task', '')}"
        feedback = step_public_record.get("public_feedback") or ""
        try:
            self.process_module.process_single_step(
                task_instruction=step_public_record.get("task") or "",
                step=ProcessTrajectoryStep(
                    observation=observation_text,
                    action=action_text,
                    predicted_state=prediction,
                    env_feedback=feedback,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._log({"process_step_error": f"{type(exc).__name__}: {exc}"})

    def after_episode(self, trajectory_public: dict, outcome_public: dict) -> None:
        if not self._guard_update():
            return
        from worldmind_plugin.core import GoalTrajectoryStep

        if outcome_public.get("success"):
            trajectory = [
                GoalTrajectoryStep(action=str(item.get("action")),
                                   env_feedback=item.get("public_feedback") or "")
                for item in (trajectory_public.get("actions") or [])
            ]
            try:
                self.goal_module.extract_experience(
                    task_instruction=trajectory_public.get("instruction") or "",
                    trajectory=trajectory,
                    success=True,
                )
            except Exception as exc:  # noqa: BLE001
                self._log({"goal_extract_error": f"{type(exc).__name__}: {exc}"})
        try:
            self.retrieval_module.reload_experiences()
        except Exception as exc:  # noqa: BLE001
            self._log({"reload_error": f"{type(exc).__name__}: {exc}"})

    def last_update_usage(self) -> dict:
        return {
            "prompt_tokens": self._sidecar_usage["prompt_tokens"] + self._goal_usage["prompt_tokens"],
            "completion_tokens": self._sidecar_usage["completion_tokens"] + self._goal_usage["completion_tokens"],
            "calls": self._sidecar_usage["calls"] + self._goal_usage["calls"],
        }

    def _log(self, record: dict) -> None:
        if self.state_dir is None:
            return
        with open(self.state_dir / "errors.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --------------------------------------------------------------- snapshot
    def snapshot(self, output_dir: Path) -> None:
        self.copy_tree(self.state_dir, output_dir)

    def state_hash(self) -> str:
        assert self.state_dir is not None
        return self.hash_dir(self.state_dir)
