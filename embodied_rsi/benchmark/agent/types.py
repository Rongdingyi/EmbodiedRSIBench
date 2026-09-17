"""Canonical agent data types (taskbook v0.8, Phase B).

No simulator or private fields may enter these dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentDecision:
    action_name: str
    parameters: dict
    thought: str = ""
    raw_response: str = ""
    planner_calls: int = 1
    validation_errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkingMemoryStep:
    step_idx: int
    action_name: str
    parameters: dict
    action_success: bool | None
    public_feedback: str
    terminated: bool
    truncated: bool


@dataclass
class AgentEpisodeState:
    instruction: str
    steps: list[WorkingMemoryStep] = field(default_factory=list)


@dataclass
class CanonicalEpisodeResult:
    task: dict
    public_trajectory: list[dict]
    private_evaluation: dict
    outcome: dict
    env_steps: int
    planner_calls: int
    wall_time_s: float
    accounting: dict
    leakage_violations: list[str]
    status: str
    error: str | None = None
    stop_reason: str = ""
    rsi_update_error: str | None = None
