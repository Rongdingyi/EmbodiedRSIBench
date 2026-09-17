"""Canonical agent freeze (taskbook v0.8, Phase S).

Deterministic hashes of the canonical agent's critical files, its config and
its system prompt, plus the static invariants the benchmark relies on. No API
calls, no simulator access.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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

# deterministic (fake-backend) tests that pin the protocol invariants
INVARIANT_TESTS = [
    "tests/test_agent_one_action_per_turn.py",
    "tests/test_agent_no_cross_episode_state.py",
    "tests/test_agent_prompt_contract.py",
    "tests/test_agent_action_validation.py",
    "tests/test_agent_budget_semantics.py",
    "tests/test_worldmind_step_timing_canonical.py",
    "tests/test_rsi_context_decoupled.py",
]


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


def run_invariant_tests(repo_root: Path, *, timeout_s: int = 300) -> dict:
    """Run the deterministic invariant tests and record hashes + result."""
    repo_root = Path(repo_root)
    files = {rel: sha256_file(repo_root / rel) for rel in INVARIANT_TESTS
             if (repo_root / rel).exists()}
    missing = [rel for rel in INVARIANT_TESTS if rel not in files]
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *files.keys()],
            cwd=repo_root, capture_output=True, text=True, timeout=timeout_s)
        tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
        passed = proc.returncode == 0
    except Exception as exc:  # noqa: BLE001
        tail = [f"invariant tests failed to run: {type(exc).__name__}: {exc}"]
        passed = False
    return {"status": "PASS" if passed and not missing else "FAIL",
            "test_file_sha256": files, "missing_tests": missing,
            "summary": tail[0] if tail else ""}


def build_freeze(repo_root: Path, config_path: Path | None = None,
                 *, run_invariants: bool = True) -> dict:
    repo_root = Path(repo_root)
    config_path = config_path or (repo_root / "configs" / "agent" / "canonical_react.yaml")
    config: AgentConfig = load_agent_config(config_path)
    critical = {rel: sha256_file(repo_root / rel) for rel in CRITICAL_FILES}
    violations = scan_forbidden_imports(repo_root)
    invariants = (run_invariant_tests(repo_root) if run_invariants
                  else {"status": "SKIPPED", "test_file_sha256": {}})
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
        "static_checks_status": "PASS" if not violations else "FAIL",
        "invariant_tests": invariants,
        "status": ("PASS" if not violations and invariants.get("status") == "PASS"
                   else "FAIL"),
    }
    return freeze


def freeze_to_json(freeze: dict) -> str:
    return json.dumps(freeze, indent=2, sort_keys=True) + "\n"
