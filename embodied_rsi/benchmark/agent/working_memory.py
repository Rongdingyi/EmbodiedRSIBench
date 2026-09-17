"""Episode-local working memory for the canonical agent (taskbook v0.8, Phase H).

Only public step facts are stored. When history is too long, the oldest steps
are dropped deterministically; no extra LLM summarization calls (that would be
an RSI-dependent confound).
"""
from __future__ import annotations

import json

from benchmark.agent.types import WorkingMemoryStep
from benchmark.rsi.context import count_tokens


class EpisodicWorkingMemory:
    def __init__(self, max_history_tokens: int = 12000):
        self.max_history_tokens = int(max_history_tokens)
        self.steps: list[WorkingMemoryStep] = []
        self.history_steps_dropped = 0

    def append(self, step: WorkingMemoryStep) -> None:
        self.steps.append(step)
        self._trim()

    def _trim(self) -> None:
        while len(self.steps) > 1 and count_tokens(self.render()) > self.max_history_tokens:
            self.steps.pop(0)
            self.history_steps_dropped += 1

    def render(self) -> str:
        if not self.steps:
            return ""
        lines = []
        for step in self.steps:
            parameters = json.dumps(step.parameters, ensure_ascii=False, sort_keys=True)
            lines.append(
                f"{step.step_idx}. action={step.action_name} parameters={parameters} "
                f"success={step.action_success} feedback={step.public_feedback!r} "
                f"terminated={step.terminated} truncated={step.truncated}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.steps = []
        self.history_steps_dropped = 0
