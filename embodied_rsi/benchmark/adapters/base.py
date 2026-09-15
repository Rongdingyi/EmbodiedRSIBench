"""Benchmark environment adapter interface (guide section 10).

Public observations never contain private evaluator data: golden actions, goal
predicates, success conditions, target poses, hidden state, pair ids, or
taxonomy labels.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PublicObservation:
    instruction: str
    images: list[Any]
    text_feedback: str
    public_metadata: dict = field(default_factory=dict)


@dataclass
class StepOutcome:
    observation: PublicObservation
    action_success: bool | None
    reward_public: float | None
    terminated: bool
    truncated: bool
    public_feedback: str


class BenchmarkEnvAdapter(ABC):
    """One live episode of one source benchmark."""

    source: str = ""

    @abstractmethod
    def reset(self, task_record: dict) -> PublicObservation:
        ...

    @abstractmethod
    def build_tool_specs(self, task_record: dict) -> list[dict]:
        ...

    @abstractmethod
    def step(self, action_name: str, parameters: dict) -> StepOutcome:
        ...

    @abstractmethod
    def private_evaluate(self) -> dict:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
