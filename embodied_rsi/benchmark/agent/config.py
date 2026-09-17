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


@dataclass(frozen=True)
class ProtocolBudgets:
    episode_timeout_s: float = 1200.0
    planner_max_output_tokens: int = 2048
    planner_validation_retries: int = 2
    max_rsi_injection_tokens: int = 8192
    max_agent_actions: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProtocolConfig:
    seed: int = 20260915
    roles: dict = field(default_factory=dict)
    probe_checkpoints: list = field(default_factory=lambda: ["S000", "S030"])
    probe_update_enabled: bool = False
    agent: AgentConfig = field(default_factory=AgentConfig)
    budgets: ProtocolBudgets = field(default_factory=ProtocolBudgets)

    @property
    def episode_timeout_s(self) -> float:
        return float(self.budgets.episode_timeout_s)

    @property
    def max_agent_actions(self) -> dict:
        return dict(self.budgets.max_agent_actions)


def validate_agent_config(config: AgentConfig) -> None:
    """Every frozen field must be truly consumed; unsupported values fail fast.

    The canonical agent implements exactly one protocol variant. Fields that
    cannot be toggled are validated here instead of being silently ignored, so
    the frozen YAML can never advertise behavior the runtime does not have.
    """
    unsupported: list[str] = []
    if config.planner.json_mode != "prompt_only":
        unsupported.append(f"planner.json_mode={config.planner.json_mode!r}")
    if config.observation.current_images_only is not True:
        unsupported.append("observation.current_images_only != true")
    if config.observation.include_public_feedback is not True:
        unsupported.append("observation.include_public_feedback != true")
    if config.observation.include_public_metadata is not False:
        unsupported.append("observation.include_public_metadata != false")
    if config.observation.include_target_image_when_source_exposes_it is not True:
        unsupported.append("observation.include_target_image_when_source_exposes_it != true")
    if config.working_memory.include_action_history is not True:
        unsupported.append("working_memory.include_action_history != true")
    if config.working_memory.include_feedback_history is not True:
        unsupported.append("working_memory.include_feedback_history != true")
    if config.working_memory.include_reasoning_history is not False:
        unsupported.append("working_memory.include_reasoning_history != false")
    if config.working_memory.truncation != "drop_oldest_steps_deterministically":
        unsupported.append(f"working_memory.truncation={config.working_memory.truncation!r}")
    if config.working_memory.model_generated_summary is not False:
        unsupported.append("working_memory.model_generated_summary != false")
    if config.execution.one_environment_action_per_turn is not True:
        unsupported.append("execution.one_environment_action_per_turn != true")
    if config.execution.random_fallback_action is not False:
        unsupported.append("execution.random_fallback_action != false")
    if config.execution.trust_agent_success_claim is not False:
        unsupported.append("execution.trust_agent_success_claim != false")
    if config.rsi.single_shared_injection_slot is not True:
        unsupported.append("rsi.single_shared_injection_slot != true")
    if config.recovery.deterministic_json_extract is not True:
        unsupported.append("recovery.deterministic_json_extract != true")
    if config.recovery.planner_retry_on_invalid_json is not True:
        unsupported.append("recovery.planner_retry_on_invalid_json != true")
    if config.recovery.planner_retry_on_invalid_action is not True:
        unsupported.append("recovery.planner_retry_on_invalid_action != true")
    if config.recovery.auto_choose_recovery_action is not False:
        unsupported.append("recovery.auto_choose_recovery_action != false")
    if unsupported:
        raise ValueError("unsupported canonical agent config values: " + ", ".join(unsupported))


def load_protocol_config(path: Path, *, repo_root: Path | None = None) -> ProtocolConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    root = Path(repo_root) if repo_root is not None else Path(path).resolve().parents[1]
    agent_path = root / (raw.get("agent", {}) or {}).get(
        "config", "configs/agent/canonical_react.yaml")
    budgets_raw = raw.get("budgets") or {}
    budgets = ProtocolBudgets(
        episode_timeout_s=budgets_raw.get("episode_timeout_s", 1200),
        planner_max_output_tokens=budgets_raw.get("planner_max_output_tokens", 2048),
        planner_validation_retries=budgets_raw.get("planner_validation_retries", 2),
        max_rsi_injection_tokens=budgets_raw.get("max_rsi_injection_tokens", 8192),
        max_agent_actions=dict(budgets_raw.get("max_agent_actions") or {}),
    )
    agent_config = load_agent_config(agent_path)
    validate_agent_config(agent_config)
    return ProtocolConfig(
        seed=int(raw.get("seed", 20260915)),
        roles=dict(raw.get("roles") or {}),
        probe_checkpoints=list(raw.get("probe_checkpoints") or ["S000", "S030"]),
        probe_update_enabled=bool(raw.get("probe_update_enabled", False)),
        agent=agent_config,
        budgets=budgets,
    )


def load_agent_config(path: Path, *, validate: bool = True) -> AgentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    config = AgentConfig(
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
    if validate:
        validate_agent_config(config)
    return config


def agent_config_hash(config: AgentConfig) -> str:
    blob = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
