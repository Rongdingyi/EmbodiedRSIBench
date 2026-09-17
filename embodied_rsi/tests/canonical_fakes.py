"""Deterministic fakes shared by the canonical-agent test suite (no real API)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark.adapters.base import PublicObservation, StepOutcome
from benchmark.agent.config import AgentConfig, ProtocolBudgets, ProtocolConfig


class FakeAdapter:
    """Scripted public adapter; records calls and dynamic tool specs."""

    source = "fake"

    def __init__(self, outcomes=None, tool_specs=None, reset_returns_specs=False):
        self.outcomes = list(outcomes or [])
        self._tool_specs = tool_specs or [{
            "name": "MoveAhead", "description": "move",
            "parameters": {"type": "object", "properties": {},
                           "required": [], "additionalProperties": False}}]
        self.reset_returns_specs = reset_returns_specs
        self.reset_calls = 0
        self.step_calls: list[tuple[str, dict]] = []
        self.closed = False

    def reset(self, task):
        self.reset_calls += 1
        if self.reset_returns_specs:
            self._tool_specs = [{"name": "Dynamic", "description": "after reset",
                                 "parameters": {"type": "object", "properties": {},
                                                "required": [], "additionalProperties": False}}]
        return PublicObservation("do the thing", [np.zeros((4, 4, 3), dtype=np.uint8)],
                                 "started", {"image_roles": ["current_view"]})

    def build_tool_specs(self, task):
        return list(self._tool_specs)

    def step(self, name, parameters):
        self.step_calls.append((name, dict(parameters)))
        if self.outcomes:
            outcome = self.outcomes.pop(0)
        else:
            outcome = dict(action_success=True, terminated=True, truncated=False,
                           feedback="done", success=True)
        obs = PublicObservation("do the thing", [np.zeros((4, 4, 3), dtype=np.uint8)],
                                outcome.get("feedback", ""), {"image_roles": ["current_view"]})
        return StepOutcome(observation=obs, action_success=outcome.get("action_success"),
                           reward_public=None, terminated=bool(outcome.get("terminated")),
                           truncated=bool(outcome.get("truncated")),
                           public_feedback=outcome.get("feedback", ""))

    def private_evaluate(self):
        return {"success": True}

    def close(self):
        self.closed = True


class FakeBackend:
    """Scripted planner backend that participates in accountant accounting."""

    def __init__(self, responses, accountant=None):
        self.responses = list(responses)
        self.accountant = accountant
        self.requests: list[dict] = []

    def complete(self, *, system_prompt, user_text, images, role="canonical_planner"):
        self.requests.append({"system_prompt": system_prompt, "user_text": user_text,
                              "images": list(images), "role": role})
        assert self.responses, "FakeBackend ran out of scripted responses"
        response = self.responses.pop(0)
        if callable(response):
            response = response(self.requests[-1])
        if isinstance(response, str):
            content = response
        else:
            content = json.dumps(response.get("content", response))
        body = {"model": "fake", "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [*images, {"type": "text", "text": user_text}]}]}
        if self.accountant is not None:
            self.accountant.record_planner_exchange(
                url="fake://planner", body=body,
                response={"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
                wall_s=0.01, role=role)
        return {"content": content, "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


def make_protocol(max_actions: int = 5, timeout_s: float = 60.0) -> ProtocolConfig:
    return ProtocolConfig(
        budgets=ProtocolBudgets(max_agent_actions={"fake": max_actions},
                                episode_timeout_s=timeout_s))


def make_task(gid: str = "fake_task_0001") -> dict:
    return {"global_task_id": gid, "source_dataset": "fake", "instruction": "do the thing"}


def valid_response(name="MoveAhead", parameters=None, thought="x"):
    return json.dumps({"thought": thought,
                       "action": {"name": name, "parameters": parameters or {}}})
