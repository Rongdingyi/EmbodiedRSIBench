#!/usr/bin/env python
"""03_verify_openeta_freeze.py -- Phase B / Gate G1.

Run with the OpenETA uv environment:
    external/OpenETA/.venv/bin/python scripts/03_verify_openeta_freeze.py
Checks pinned commit, clean worktree, no diff vs pin, critical file hashes,
planner prompt hash, and that the bridge runtime builder disables native SI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.openeta_bridge import freeze  # noqa: E402

OUT = PROJECT / "outputs" / "preflight"
OUT.mkdir(parents=True, exist_ok=True)
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + ("" if cond else f"  ({detail})"))
    if not cond:
        FAILS.append(f"{name}: {detail}")


def main() -> int:
    head = freeze.head_commit()
    check("HEAD == pinned commit", head == freeze.OPENETA_PIN, head)
    check("worktree clean", freeze.git_status_clean())
    check("no diff vs pin", freeze.diff_against_pin() == "", freeze.diff_against_pin()[:120])

    hashes = freeze.critical_hashes()
    check("critical files present and hashed", len(hashes) == len(freeze.CRITICAL_FILES),
          f"{len(hashes)}/{len(freeze.CRITICAL_FILES)}")

    prompt, prompt_hash = freeze.planner_prompt()
    check("planner prompt non-empty", len(prompt) > 1000, str(len(prompt)))

    # native SI must be disabled on the bridge runtime (built in Phase E; here we
    # verify the disable helper itself on a minimal runtime).
    try:
        from agent.runtime.runtime import OpenEtaAgentRuntime  # noqa: E402

        runtime = OpenEtaAgentRuntime(rollout_enabled=False)
        freeze.disable_native_self_improvement(runtime)
        freeze.assert_native_self_improvement_disabled(runtime)
        check("native self-improvement disabled on runtime", True)
    except Exception as exc:  # noqa: BLE001
        check("native self-improvement disabled on runtime", False, str(exc))

    manifest = freeze.freeze_manifest()
    (PROJECT / "manifests" / "openeta_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    audit = {"status": "PASS" if not FAILS else "FAIL", "failures": FAILS,
             "planner_prompt_sha256": prompt_hash, "head": head}
    (OUT / "OPENETA_FREEZE.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"\nG1 OPENETA FREEZE GATE: {audit['status']}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
