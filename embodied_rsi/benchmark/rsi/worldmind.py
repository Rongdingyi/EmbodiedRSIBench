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
from benchmark.utils import chat_completions_url, normalize_base_url

PROJECT = Path(__file__).resolve().parents[2]
PLUGIN = PROJECT / "external" / "WorldMind" / "Plugin"
if str(PLUGIN) not in sys.path:
    sys.path.insert(0, str(PLUGIN))

BASE_URL = normalize_base_url(os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
MAX_INJECTION_TOKENS = 8192

# Instrumentation hook: the official modules construct their own LLMClients, so
# we intercept LLMClient._call_api once (identical request, plus accounting).
_ACTIVE_OWNER: "WorldMindRSI | None" = None
_CLIENT_INSTRUMENTED = False
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def _ensure_worldmind_client_instrumented() -> None:
    global _CLIENT_INSTRUMENTED
    if _CLIENT_INSTRUMENTED:
        return
    from worldmind_plugin.llm_client import LLMClient

    original = LLMClient._call_api

    def instrumented(self, messages):
        owner = _ACTIVE_OWNER
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - mirror upstream error surface
            if getattr(self, "logger", None) is not None:
                self.logger.error(f"LLM API call failed: {exc}")
            raise
        text = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        if owner is not None:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            owner._component_usage["prompt_tokens"] += prompt
            owner._component_usage["completion_tokens"] += completion
            owner._component_usage["calls"] += 1
            accountant = (owner.run_ctx or {}).get("accountant")
            if accountant is not None:
                accountant.record_request_dump(
                    url=chat_completions_url(BASE_URL),
                    body={"model": self.model_name, "messages": list(messages),
                          "max_tokens": self.max_tokens},
                    role="worldmind_component",
                )
            # accounting authority is the runner: it records
            # get_last_update_usage() exactly once per episode.
        return text

    LLMClient._call_api = instrumented
    LLMClient._openeta_original_call_api = original  # keep a handle for audits
    _CLIENT_INSTRUMENTED = True


def _state_text(state) -> str:
    """Compact public state summary used for predicted-vs-actual comparison."""
    if not isinstance(state, dict):
        return ""
    parts = []
    if state.get("text_feedback"):
        parts.append(str(state["text_feedback"]))
    for key in ("visible_object_types", "image_roles"):
        value = state.get(key)
        if value:
            parts.append(f"{key}={value}")
    return "; ".join(parts)[:400]


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
        chat_completions_url(BASE_URL),
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
    return {"text": text, "usage": payload.get("usage", {}),
            "wall_s": time.time() - t0, "request_body": body}


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
        self._sidecar_usage = dict(ZERO_USAGE)
        self._sidecar_wall_s = 0.0
        self._component_usage = dict(ZERO_USAGE)
        self.strict_updates = True
        self.update_errors: list[str] = []
        self.prediction_errors: list[dict] = []

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _ensure_worldmind_importable() -> None:
        """Load the official plugin submodules without its broken __init__.

        Upstream packaging bug at the pinned commit 712b0fd:
        `worldmind_plugin/__init__.py` imports `parse_llm_output` while
        `utils.py` defines `parse_agent_output`, so `import worldmind_plugin`
        fails. We pre-register the package namespace and load the official
        submodules by file, leaving every module's code untouched.
        """
        import importlib.util
        import sys
        import types

        if "worldmind_plugin.core" in sys.modules:
            return
        package = types.ModuleType("worldmind_plugin")
        package.__path__ = [str(PLUGIN / "worldmind_plugin")]
        sys.modules["worldmind_plugin"] = package
        for name in ("utils", "prompts", "config", "llm_client",
                     "state_summarizer", "discriminator", "reflector",
                     "experience_refiner", "knowledge_manager", "core"):
            module_name = f"worldmind_plugin.{name}"
            if module_name in sys.modules:
                continue
            spec = importlib.util.spec_from_file_location(
                module_name, PLUGIN / "worldmind_plugin" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

    def _build_modules(self) -> None:
        self._ensure_worldmind_importable()
        _ensure_worldmind_client_instrumented()
        from worldmind_plugin.config import WorldMindConfig
        from worldmind_plugin.core import (
            ExperienceRetrievalModule,
            GoalExperienceModule,
            ProcessExperienceModule,
        )

        # the official LLMClient feeds api_base straight into the OpenAI SDK,
        # which appends only "/chat/completions"; the SDK base therefore needs
        # the /v1 suffix (unlike the pinned OpenETA backend).
        self.config = WorldMindConfig(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            api_base=f"{BASE_URL}/v1",
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
        global _ACTIVE_OWNER
        _ACTIVE_OWNER = self
        self._pending_prediction = None
        self._pending_action = None
        # per-episode usage windows (P1 accounting)
        self._sidecar_usage = dict(ZERO_USAGE)
        self._component_usage = dict(ZERO_USAGE)
        self._sidecar_wall_s = 0.0
        self.update_errors = []
        self.prediction_errors = []
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
            accountant.record_request_dump(          # P1-10: sidecar request audit
                url=f"{BASE_URL.rstrip('/')}/v1/chat/completions",
                body=result.get("request_body") or {},
                role="worldmind_sidecar",
            )
        self._pending_prediction = result.get("text") or ""
        self._pending_action = dict(action)
        return self._pending_prediction

    def after_step(self, step_public_record: dict) -> None:
        global _ACTIVE_OWNER
        _ACTIVE_OWNER = self
        if not self._guard_update() or self._pending_prediction is None:
            self._pending_prediction = None
            return
        from worldmind_plugin.core import ProcessTrajectoryStep

        prediction = self._pending_prediction
        action = self._pending_action or {}
        self._pending_prediction = None
        self._pending_action = None
        instruction = str(step_public_record.get("task_instruction") or "")
        action_text = f"{action.get('name')} {json.dumps(action.get('parameters') or {})}".strip()
        state_before = _state_text(step_public_record.get("state_before"))
        state_after = _state_text(step_public_record.get("state_after"))
        feedback = step_public_record.get("public_feedback") or ""
        # Official semantics: state_after = step.observation, state_before is the
        # explicit kwarg; discriminator compares prediction vs state_after (P0-1).
        try:
            # Official semantics: the first return value is "the prediction was
            # wrong and a process experience was extracted" -- the productive
            # path of the module, NOT an internal failure. Genuine module
            # exceptions still propagate to rsi_update_error below.
            prediction_error, experiences = self.process_module.process_single_step(
                task_instruction=instruction,
                step=ProcessTrajectoryStep(
                    observation=state_after,
                    action=action_text,
                    predicted_state=prediction,
                    env_feedback=feedback,
                ),
                state_before=state_before,
            )
            if prediction_error:
                self.prediction_errors.append(
                    {"step_action": action_text, "experiences": list(experiences or [])})
                self._log({"prediction_error_reflected": len(experiences or [])})
        except Exception as exc:  # noqa: BLE001
            message = f"process_step_error: {type(exc).__name__}: {exc}"
            self.update_errors.append(message)
            self._log({"process_step_error": f"{type(exc).__name__}: {exc}"})
            if self.strict_updates:
                raise

    def after_episode(self, trajectory_public: dict, outcome_public: dict) -> None:
        global _ACTIVE_OWNER
        _ACTIVE_OWNER = self
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
                )
            except Exception as exc:  # noqa: BLE001
                self.update_errors.append(f"goal_extract_error: {type(exc).__name__}: {exc}")
                self._log({"goal_extract_error": f"{type(exc).__name__}: {exc}"})
                if self.strict_updates:
                    raise RuntimeError(f"WorldMind goal extraction failed: {exc}") from exc
        try:
            self.retrieval_module.reload_experiences()
        except Exception as exc:  # noqa: BLE001
            self.update_errors.append(f"reload_error: {type(exc).__name__}: {exc}")
            self._log({"reload_error": f"{type(exc).__name__}: {exc}"})
            if self.strict_updates:
                raise RuntimeError(f"WorldMind experience reload failed: {exc}") from exc

    def get_last_update_usage(self) -> dict:
        """This episode's component tokens, excluding the sidecar (accounted
        separately by the environment's sidecar hook)."""
        return dict(self._component_usage)

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
