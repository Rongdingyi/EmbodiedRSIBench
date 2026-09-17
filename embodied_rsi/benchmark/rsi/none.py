"""Condition A: `none` -- zero-persistence baseline (guide section 17)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from benchmark.rsi.context import RSIInjection, make_injection
from benchmark.rsi.base import RSIMethod


class NoneRSI(RSIMethod):
    name = "none"

    def before_episode(self, task_public_view: dict) -> RSIInjection:
        return make_injection("")   # empty injection, shared channel

    def after_episode(self, trajectory_public: dict, outcome_public: dict) -> None:
        return None

    def snapshot(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "state.json").write_text('{"state": "empty"}\n')

    def _reload_state(self) -> None:
        return None

    def state_hash(self) -> str:
        return hashlib.sha256(b"none:constant-empty-state").hexdigest()[:24]
