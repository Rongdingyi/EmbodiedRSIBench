"""Condition E: EmbodiSkill -- integration assessed and BLOCKED for Pilot v0.3.

Call-path audit (guide section 21.2) is documented in
`outputs/preflight/EMBODISKILL_CALL_PATH.md`. The official implementation is
built around MS-Agent/ALFWorld `StateChain`/agentkit execution and its
reflection epoch files assume the ALFWorld runner owns the trajectory. Wiring it
to an external OpenETA trajectory without rewriting its reflection/update
semantics is not possible within the guide's <=200-line limit, so this method is
recorded as BLOCKED instead of shipping a pseudo-EmbodiSkill.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from benchmark.openeta_bridge.context_injection import RSIInjection, make_injection
from benchmark.rsi.base import RSIMethod


class EmbodiSkillRSI(RSIMethod):
    name = "embodiskill"
    BLOCKED = True
    BLOCK_REASON = (
        "official EmbodiSkill is ALFWorld/agentkit-specific; external OpenETA "
        "trajectory injection requires rewriting reflection/update semantics "
        "(>200 lines) -- see outputs/preflight/EMBODISKILL_CALL_PATH.md and BLOCKERS.md"
    )

    def init_run(self, run_ctx: dict) -> None:
        super().init_run(run_ctx)
        (self.state_dir / "BLOCKED.txt").write_text(self.BLOCK_REASON + "\n")

    def before_episode(self, task_public_view: dict) -> RSIInjection:
        return make_injection("")   # never provides guidance while blocked

    def after_episode(self, trajectory_public: dict, outcome_public: dict) -> None:
        return None

    def snapshot(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "BLOCKED.txt").write_text(self.BLOCK_REASON + "\n")

    def load_snapshot(self, input_dir: Path) -> None:
        return None

    def state_hash(self) -> str:
        return hashlib.sha256(b"embodiskill:blocked").hexdigest()[:24]
