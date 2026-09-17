"""Canonical agent freeze (taskbook v0.8, Phase S).

Deterministic hashes of the canonical agent's critical files, its config and
its system prompt, plus the static invariants the benchmark relies on. No API
calls, no simulator access.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmark.agent.config import AgentConfig, agent_config_hash, load_agent_config
from benchmark.agent.prompt import CANONICAL_SYSTEM_PROMPT

CRITICAL_FILES = [
    "benchmark/agent/prompt.py",
    "benchmark/agent/backend.py",
    "benchmark/agent/action_schema.py",
    "benchmark/agent/working_memory.py",
    "benchmark/agent/react_agent.py",
    "benchmark/agent/episode_runner.py",
    "configs/agent/canonical_react.yaml",
]

FORBIDDEN_IMPORT_MARKERS = ("openeta_bridge", "agent.runtime", "agentkit.")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def scan_forbidden_imports(repo_root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted((Path(repo_root) / "benchmark" / "agent").glob("*.py")):
        if path.name == "freeze.py":   # holds the marker list itself
            continue
        text = path.read_text()
        for marker in FORBIDDEN_IMPORT_MARKERS:
            if marker in text:
                violations.append(f"{path.name}: {marker}")
    return violations


def build_freeze(repo_root: Path, config_path: Path | None = None) -> dict:
    repo_root = Path(repo_root)
    config_path = config_path or (repo_root / "configs" / "agent" / "canonical_react.yaml")
    config: AgentConfig = load_agent_config(config_path)
    critical = {rel: sha256_file(repo_root / rel) for rel in CRITICAL_FILES}
    violations = scan_forbidden_imports(repo_root)
    freeze = {
        "agent_name": config.agent_name,
        "reference": "EmbodiedBench/VLMPlanner",
        "reference_commit": config.reference_commit,
        "system_prompt_sha256": sha256_text(CANONICAL_SYSTEM_PROMPT),
        "agent_config_sha256": agent_config_hash(config),
        "critical_file_sha256": critical,
        "one_action_per_turn": bool(config.execution.one_environment_action_per_turn),
        "cross_episode_agent_memory": False,
        "rsi_only_persistent_state": True,
        "forbidden_import_violations": violations,
        "status": "PASS" if not violations else "FAIL",
    }
    return freeze


def freeze_to_json(freeze: dict) -> str:
    return json.dumps(freeze, indent=2, sort_keys=True) + "\n"
