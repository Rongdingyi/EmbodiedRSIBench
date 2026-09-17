"""Canonical Multimodal ReAct agent (taskbook v0.8, Phase I).

`decide()` never touches the environment: it produces exactly one validated
action. Retries are fixed canonical-agent recovery (identical for every RSI
method) and, when exhausted, raise PlannerProtocolError instead of executing a
random or guessed action.
"""
from __future__ import annotations

import copy
import json

from benchmark.agent.action_schema import parse_agent_json, validate_agent_action
from benchmark.agent.backend import PlannerProtocolError
from benchmark.agent.prompt import (
    CANONICAL_SYSTEM_PROMPT,
    build_invalid_response_note,
    build_turn_prompt,
    render_visual_inputs,
)
from benchmark.agent.types import AgentDecision, WorkingMemoryStep
from benchmark.agent.vision import observation_image_parts
from benchmark.agent.working_memory import EpisodicWorkingMemory

STUCK_NOTE = ("Recovery note: the previous action pattern has repeated. Re-evaluate the current "
              "observation and choose a different legal action if the repeated action is not "
              "making progress.")


class CanonicalMultimodalReActAgent:
    def __init__(self, *, backend, config, accountant=None):
        self.backend = backend
        self.config = config
        self.accountant = accountant
        self.memory = EpisodicWorkingMemory(
            max_history_tokens=config.working_memory.max_history_tokens)
        self.tool_specs: list[dict] = []
        self.instruction = ""
        self.rsi_injection = None
        self.planner_calls_episode = 0
        self._last_action_sig: tuple[str, str] | None = None
        self._identical_streak = 0
        self._streak_feedbacks: list[str] = []

    # ------------------------------------------------------------------ session
    def start_episode(self, *, instruction: str, tool_specs: list[dict],
                      rsi_injection) -> None:
        self.memory.clear()
        self.instruction = instruction or ""
        self.tool_specs = copy.deepcopy(list(tool_specs or []))
        self.rsi_injection = rsi_injection
        self.planner_calls_episode = 0
        self._last_action_sig = None
        self._identical_streak = 0
        self._streak_feedbacks = []

    def close_episode(self) -> None:
        self.memory.clear()

    # ------------------------------------------------------------------- decide
    @staticmethod
    def _current_image_roles(observation) -> list[str]:
        """Allowlisted public image roles; never private metadata."""
        metadata = getattr(observation, "public_metadata", None)
        roles = metadata.get("image_roles") if isinstance(metadata, dict) else None
        images = list(getattr(observation, "images", None) or [])
        if not isinstance(roles, list) or len(roles) != len(images):
            roles = ["current_view"] * len(images)
        return [str(role) for role in roles]

    def decide(self, observation) -> AgentDecision:
        stuck_warning = self._stuck_warning()
        base_prompt = build_turn_prompt(
            instruction=self.instruction,
            tool_specs=self.tool_specs,
            working_memory_text=self.memory.render(),
            public_feedback=observation.text_feedback,
            rsi_injection=self.rsi_injection,
            stuck_warning=stuck_warning,
            visual_inputs_text=render_visual_inputs(self._current_image_roles(observation)),
        )
        images = observation_image_parts(observation, max_images=self.config.planner.max_images)

        errors: list[str] = []
        attempts = 1 + max(0, int(self.config.planner.max_validation_retries))
        for attempt in range(attempts):
            user_text = base_prompt
            if attempt > 0:
                user_text = f"{base_prompt}\n\n{build_invalid_response_note(errors)}"
            result = self.backend.complete(
                system_prompt=CANONICAL_SYSTEM_PROMPT, user_text=user_text,
                images=images, role="canonical_planner")
            self.planner_calls_episode += 1
            raw = result.get("content") or ""

            payload, parse_errors = parse_agent_json(raw)
            if payload is None:
                errors.extend(parse_errors)
                self._record_retry(kind="json")
                continue
            name, parameters, validation_errors = validate_agent_action(payload, self.tool_specs)
            if name is None:
                errors.extend(validation_errors)
                self._record_retry(kind="action")
                continue
            return AgentDecision(
                action_name=name,
                parameters=parameters or {},
                thought=str(payload.get("thought") or ""),
                raw_response=raw,
                planner_calls=self.planner_calls_episode,
                validation_errors=list(errors),
            )
        raise PlannerProtocolError("planner_protocol_failure: " + "; ".join(errors[-6:]))

    # -------------------------------------------------------------- transitions
    def record_transition(self, *, action_name: str, parameters: dict, outcome) -> None:
        signature = (action_name, json.dumps(parameters, sort_keys=True))
        feedback = outcome.public_feedback or ""
        if signature == self._last_action_sig:
            self._identical_streak += 1
        else:
            self._identical_streak = 1
            self._streak_feedbacks = []
        self._last_action_sig = signature
        self._streak_feedbacks.append(feedback)
        self.memory.append(WorkingMemoryStep(
            step_idx=len(self.memory.steps) + 1,
            action_name=action_name,
            parameters=dict(parameters),
            action_success=outcome.action_success,
            public_feedback=feedback,
            terminated=bool(outcome.terminated),
            truncated=bool(outcome.truncated),
        ))

    # ------------------------------------------------------------------ helpers
    def _stuck_warning(self) -> str:
        threshold = int(self.config.recovery.stuck_warning_after_identical_actions)
        if self._identical_streak < threshold:
            return ""
        recent = self._streak_feedbacks[-threshold:]
        if len(set(recent)) > 1:
            return ""
        return STUCK_NOTE

    def _record_retry(self, *, kind: str) -> None:
        if self.accountant is None:
            return
        if kind == "json":
            self.accountant.record_invalid_json()
        else:
            self.accountant.record_invalid_action()
