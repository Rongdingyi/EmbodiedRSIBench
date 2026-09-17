"""Canonical agent configuration (taskbook v0.8, Phase C).

Loaded from `configs/agent/canonical_react.yaml`. The resolved dataclasses are
hashed for the freeze/provenance records so every method runs under the exact
same agent configuration.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PlannerConfig:
    temperature: float = 0.0
    max_output_tokens: int = 2048
    max_validation_retries: int = 2
    max_images: int = 2
    json_mode: str = "prompt_only"


@dataclass(frozen=True)
class ObservationConfig:
    current_images_only: bool = True
    include_public_feedback: bool = True
    include_public_metadata: bool = False
    include_target_image_when_source_exposes_it: bool = True


@dataclass(frozen=True)
class WorkingMemoryConfig:
    include_action_history: bool = True
    include_feedback_history: bool = True
    include_reasoning_history: bool = False
    max_history_tokens: int = 12000
    truncation: str = "drop_oldest_steps_deterministically"
    model_generated_summary: bool = False


@dataclass(frozen=True)
class ExecutionConfig:
    one_environment_action_per_turn: bool = True
    random_fallback_action: bool = False
    trust_agent_success_claim: bool = False


@dataclass(frozen=True)
class RSIConfig:
    max_injection_tokens: int = 8192
    single_shared_injection_slot: bool = True


@dataclass(frozen=True)
class RecoveryConfig:
    deterministic_json_extract: bool = True
    planner_retry_on_invalid_json: bool = True
    planner_retry_on_invalid_action: bool = True
    stuck_warning_after_identical_actions: int = 3
    auto_choose_recovery_action: bool = False


@dataclass(frozen=True)
class AgentConfig:
    agent_name: str = "canonical_multimodal_react_v1"
    architecture: str = "multimodal_react_planner_executor"
    reference: str = "embodiedbench_vlmplanner"
    reference_commit: str = "9be4e980e9cd6bcb38373cd4aab7c32724bdd401"
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    working_memory: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    rsi: RSIConfig = field(default_factory=RSIConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_agent_config(path: Path) -> AgentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return AgentConfig(
        agent_name=raw.get("agent_name", AgentConfig.agent_name),
        architecture=raw.get("architecture", AgentConfig.architecture),
        reference=raw.get("reference", AgentConfig.reference),
        reference_commit=raw.get("reference_commit", AgentConfig.reference_commit),
        planner=PlannerConfig(**{**asdict(PlannerConfig()), **(raw.get("planner") or {})}),
        observation=ObservationConfig(**{**asdict(ObservationConfig()),
                                         **(raw.get("observation") or {})}),
        working_memory=WorkingMemoryConfig(**{**asdict(WorkingMemoryConfig()),
                                              **(raw.get("working_memory") or {})}),
        execution=ExecutionConfig(**{**asdict(ExecutionConfig()),
                                     **(raw.get("execution") or {})}),
        rsi=RSIConfig(**{**asdict(RSIConfig()), **(raw.get("rsi") or {})}),
        recovery=RecoveryConfig(**{**asdict(RecoveryConfig()),
                                   **(raw.get("recovery") or {})}),
    )


def agent_config_hash(config: AgentConfig) -> str:
    blob = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
