"""Condition E: EmbodiSkill -- integration assessed and BLOCKED for Pilot v0.3.

Call-path audit (guide section 21.2) is documented in
`outputs/preflight/EMBODISKILL_CALL_PATH.md`. The official implementation is
built around MS-Agent/ALFWorld `StateChain`/agentkit execution and its
reflection epoch files assume the ALFWorld runner owns the trajectory. Wiring it
to a canonical public trajectory without rewriting its reflection/update
semantics is not possible within the guide's <=200-line limit, so this method is
recorded as BLOCKED instead of shipping a pseudo-EmbodiSkill.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark.rsi.context import RSIInjection, count_tokens, make_injection
from benchmark.rsi.base import RSIMethod

WORKER = Path(__file__).resolve().parents[1] / "adapters" / "workers" / "embodiskill_worker.py"


class EmbodiSkillRSI(RSIMethod):
    name = "embodiskill"
    # Excluded from the pilot table until the thin-adapter PoC is executed in its
    # dependency environment; the converter itself is implemented (P1-8 review).
    BLOCKED = True
    STATUS = "POC_READY_UNVERIFIED"
    BLOCK_REASON = (
        "thin adapter implemented in benchmark/adapters/workers/embodiskill_worker.py "
        "(official init_task_context -> move_skill_state -> save_task_context -> "
        "reflect_episode -> revise_manual; <=200 lines, no upstream edits) but the "
        "PoC has not been executed yet: it needs the EmbodiSkill dependency env "
        "(langchain-chroma + sentence-transformers + finch). See "
        "outputs/preflight/EMBODISKILL_CALL_PATH.md"
    )

    def __init__(self) -> None:
        super().__init__()
        self.worker: subprocess.Popen | None = None
        self._request_id = 0

    def init_run(self, run_ctx: dict) -> None:
        super().init_run(run_ctx)
        (self.state_dir / "STATUS.txt").write_text(
            f"{self.STATUS}\n{self.BLOCK_REASON}\n")
        if os.environ.get("EMBODISKILL_ENABLE", "0") == "1":
            try:
                self._start_worker()
            except Exception as exc:  # noqa: BLE001 - stays blocked, never faked
                self.worker = None
                self.STATUS = "POC_READY_UNVERIFIED"
                self.BLOCK_REASON = f"{self.BLOCK_REASON} | worker start failed: {exc}"

    # ------------------------------------------------------------- worker client
    def _python(self) -> str:
        return os.environ.get("EMBODISKILL_PYTHON", sys.executable)

    def _start_worker(self) -> None:
        assert self.state_dir is not None, "init_run first"
        env = dict(os.environ, EMBODISKILL_STATE_ROOT=str(self.state_dir))
        self.worker = subprocess.Popen(
            [self._python(), str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)

    def _stop_worker(self) -> None:
        if self.worker is not None:
            try:
                self.worker.terminate()
                self.worker.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    self.worker.kill()
                except Exception:  # noqa: BLE001
                    pass
            self.worker = None

    def _reload_state(self) -> None:
        """Restart the worker so it reads the (possibly copied) state dir."""
        if os.environ.get("EMBODISKILL_ENABLE", "0") == "1":
            self._stop_worker()
            self._start_worker()

    def _call(self, op: str, args: dict | None = None) -> dict:
        assert self.worker is not None, "worker not running"
        self._request_id += 1
        request = {"id": self._request_id, "op": op, "args": args or {}}
        self.worker.stdin.write(json.dumps(request) + "\n")
        self.worker.stdin.flush()
        deadline = time.time() + 600
        while time.time() < deadline:
            line = self.worker.stdout.readline()
            if not line:
                break
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if response.get("id") == self._request_id:
                if not response.get("ok"):
                    raise RuntimeError(f"embodiskill worker error: {response.get('error')}")
                return response.get("result") or {}
        raise RuntimeError("embodiskill worker did not answer")

    # ------------------------------------------------------------------- hooks
    def before_episode(self, task_public_view: dict) -> RSIInjection:
        if self.worker is None:
            return make_injection("")   # blocked: no guidance
        guidance = str(self._call("before_episode").get("guidance") or "").strip()
        provenance = [f"manual:{self.worker.pid}"] if guidance else []
        if guidance and count_tokens(guidance) > 8192:
            guidance = guidance[: 8192 * 4]
        return make_injection(guidance, provenance_ids=provenance)

    def after_episode(self, trajectory_public: dict, outcome_public: dict) -> None:
        if self.worker is None or not self._guard_update():
            return
        started = time.time()
        result = self._call("after_episode", {"trajectory": trajectory_public,
                                              "outcome": outcome_public})
        self._last_update_usage = result.get("usage") or {}
        self._last_update_wall_s = time.time() - started

    def snapshot(self, output_dir: Path) -> None:
        self._stop_worker()
        self.copy_tree(self.state_dir, Path(output_dir))
        (Path(output_dir) / "STATUS.txt").write_text(f"{self.STATUS}\n{self.BLOCK_REASON}\n")
        if os.environ.get("EMBODISKILL_ENABLE", "0") == "1":
            self._start_worker()

    def load_snapshot(self, input_dir: Path) -> None:
        super().load_snapshot(input_dir)   # copies + _reload_state (worker restart)

    def state_hash(self) -> str:
        if self.state_dir is None or not (self.state_dir / "STATUS.txt").exists():
            return hashlib.sha256(b"embodiskill:blocked").hexdigest()[:24]
        return self.hash_dir(self.state_dir)
