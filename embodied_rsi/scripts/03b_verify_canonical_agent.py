#!/usr/bin/env python
"""03b_verify_canonical_agent.py -- canonical agent freeze (deterministic, no API).

Checks the frozen canonical agent files/config/prompt and the static invariants:
- one planner decision -> exactly one environment action;
- no cross-episode agent memory (working memory is episode-local);
- only the RSI method may persist cross-episode state;
- canonical agent code imports no historical OpenETA / runtime modules.

Writes outputs/preflight/CANONICAL_AGENT_FREEZE.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.agent.freeze import build_freeze, freeze_to_json  # noqa: E402

OUT = PROJECT / "outputs" / "preflight"


def main() -> int:
    freeze = build_freeze(PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CANONICAL_AGENT_FREEZE.json").write_text(freeze_to_json(freeze))
    print(freeze_to_json(freeze))
    print(f"CANONICAL AGENT FREEZE: {freeze['status']}")
    return 0 if freeze["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
