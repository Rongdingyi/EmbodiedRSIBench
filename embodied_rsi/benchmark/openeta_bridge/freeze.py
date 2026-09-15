"""OpenETA freeze: critical file hashes, planner prompt hash, native SI disable."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OPENETA = PROJECT / "external" / "OpenETA"
OPENETA_PIN = "7d4a0a1522ba8ebbd362bde880bad81d2a98f15e"

CRITICAL_FILES = [
    "agent/runtime/planner.py",
    "agent/runtime/episode.py",
    "agent/runtime/runtime.py",
    "agent/runtime/runtime_assembly.py",
    "agent/runtime/self_improvement.py",
    "agent/runtime/memory.py",
    "agent/runtime/planner_prompts.py",
    "agent/runtime/actions.py",
    "agent/runtime/pipeline.py",
    "agent/backends/planner.py",
    "adapter/protocol.py",
]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=OPENETA).decode().strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def critical_hashes() -> dict[str, str]:
    return {rel: sha256_file(OPENETA / rel) for rel in CRITICAL_FILES}


def git_status_clean() -> bool:
    return _git("status", "--porcelain") == ""


def head_commit() -> str:
    return _git("rev-parse", "HEAD")


def diff_against_pin() -> str:
    return subprocess.run(["git", "diff", "--stat", OPENETA_PIN], cwd=OPENETA,
                          capture_output=True, text=True).stdout.strip()


def planner_prompt(openeta_dir: Path = OPENETA) -> tuple[str, str]:
    """Return (system_prompt, sha256). Imports the pinned upstream composer."""
    import sys

    sys.path.insert(0, str(openeta_dir))
    from agent.runtime.planner import _agent_owned_tool_planner_system_prompt
    from agent.runtime.planner_prompts import compose_main_planner_prompt

    prompt, meta = compose_main_planner_prompt(_agent_owned_tool_planner_system_prompt())
    return prompt, meta["sha256"]


def disable_native_self_improvement(runtime) -> None:
    """Guide section 1.2: hard-disable OpenETA's own post-episode SI."""
    reviewer = runtime.self_improvement_reviewer
    reviewer.config = replace(
        reviewer.config,
        enabled=False,
        auto_apply_reviewed=False,
    )
    reviewer.auto_applier = None


def assert_native_self_improvement_disabled(runtime) -> None:
    reviewer = runtime.self_improvement_reviewer
    assert reviewer.config.enabled is False, "native self-improvement still enabled"
    assert reviewer.config.auto_apply_reviewed is False, "native auto-apply still enabled"
    assert reviewer.auto_applier is None, "native auto_applier not cleared"


def freeze_manifest(runtime=None) -> dict:
    prompt, prompt_hash = planner_prompt()
    manifest = {
        "repo_commit": head_commit(),
        "pinned_commit": OPENETA_PIN,
        "git_worktree_clean": git_status_clean(),
        "diff_vs_pin": diff_against_pin(),
        "critical_file_sha256": critical_hashes(),
        "planner_prompt_sha256": prompt_hash,
        "planner_prompt_chars": len(prompt),
        "native_self_improvement": False,
        "native_auto_apply": False,
        "native_iterate_allowed": False,
        "cross_session_openeta_memory": False,
        "enabled_tool_policy": "benchmark_source_actions_only",
        "web_tools_enabled": False,
        "python_exec_enabled": False,
    }
    if runtime is not None:
        reviewer = runtime.self_improvement_reviewer
        manifest["runtime_self_improvement_enabled"] = bool(reviewer.config.enabled)
        manifest["runtime_auto_apply_reviewed"] = bool(reviewer.config.auto_apply_reviewed)
        manifest["runtime_auto_applier"] = reviewer.auto_applier is not None
    return manifest
