"""Condition D: WorldMind (Goal + Process Experience) with OpenETA executor.

Uses the official `Plugin/worldmind_plugin` modules:
  ProcessExperienceModule / GoalExperienceModule / ExperienceRetrievalModule.
The predicted-state sidecar is an independent DeepSeek call that happens after
OpenETA has locked its action; its output never returns to the planner
(guide section 20.2).
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


def _sidecar_predict(observation_text: str, action_text: str, image_path: str | None) -> dict:
    """DeepSeek call that predicts the abstract next state (RSI-cost counted)."""
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
        self.current_task: dict | None = None
        self.pending: dict | None = None

    # ------------------------------------------------------------------ setup
    def init_run(self, run_ctx: dict) -> None:
        super().init_run(run_ctx)
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

    # ------------------------------------------------------------------ hooks
    def before_episode(self, task_public_view: dict) -> RSIInjection:
        self.current_task = task_public_view
        self.pending = None
        try:
            result = self.retrieval_module.retrieve(
                task_instruction=task_public_view.get("instruction") or "",
                enable_refine=True,
            )
        except Exception as exc:  # noqa: BLE001 — retrieval failures must not stop the episode
            return make_injection("")
        text = (result or {}).get("formatted_prompt") or ""
        if not text.strip():
            return make_injection("")
        if count_tokens(text) > MAX_INJECTION_TOKENS:
            text = text[: MAX_INJECTION_TOKENS * 3]
        return make_injection(text, provenance_ids=(result or {}).get("experience_ids") or [])

    def after_step(self, step_public_record: dict) -> None:
        self.pending = dict(step_public_record)

    def after_episode(self, trajectory_public: dict, outcome_public: dict) -> None:
        if not self._guard_update():
            return
        from worldmind_plugin.core import GoalTrajectoryStep, ProcessTrajectoryStep

        actions = trajectory_public.get("actions") or []
        # ---- process experience: prediction-error driven, step by step
        for i, item in enumerate(actions):
            action_text = f"{item.get('action')} {json.dumps(item.get('parameters') or {})}".strip()
            feedback = item.get("public_feedback") or ""
            observation_text = f"step {i}, previous feedback: {feedback[:200]}"
            prediction = _sidecar_predict(observation_text, action_text, None)
            try:
                self.process_module.process_single_step(
                    task_instruction=trajectory_public.get("instruction") or "",
                    step=ProcessTrajectoryStep(
                        observation=observation_text,
                        action=action_text,
                        predicted_state=prediction["text"],
                        env_feedback=feedback,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                self._log({"process_step_error": f"{type(exc).__name__}: {exc}", "step": i})
        # ---- goal experience: only from successful episodes (guide 20.4)
        if outcome_public.get("success"):
            trajectory = [
                GoalTrajectoryStep(action=str(item.get("action")),
                                   env_feedback=item.get("public_feedback") or "")
                for item in actions
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

    def _log(self, record: dict) -> None:
        assert self.state_dir is not None
        with open(self.state_dir / "errors.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --------------------------------------------------------------- snapshot
    def snapshot(self, output_dir: Path) -> None:
        self.copy_tree(self.state_dir, output_dir)

    def load_snapshot(self, input_dir: Path) -> None:
        self.state_dir = Path(input_dir)
        self.init_run({"state_root": str(input_dir), **self.run_ctx})

    def state_hash(self) -> str:
        assert self.state_dir is not None
        return self.hash_dir(self.state_dir)
