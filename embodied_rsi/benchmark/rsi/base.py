"""Unified RSI plugin interface (guide section 14).

Hard rules:
  * an RSIMethod never holds a simulator object;
  * it reads only public task / trajectory / feedback / outcome data;
  * it cannot touch the canonical agent internals, simulator, action schema,
    private evaluator, or benchmark task selector;
  * all cross-episode state lives in its own `rsi_state/` directory.

Snapshot semantics (P0-4): a clone is always a **new instance** operating on a
temporary copy of the snapshot directory. Live plugins (OpenAI clients, official
modules) are never deep-copied, and a probe can never write back into the
snapshot it was cloned from.
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from benchmark.rsi.context import RSIInjection, make_injection


class RSIMethod(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.update_enabled: bool = True
        self.run_ctx: dict = {}
        self.state_dir: Path | None = None
        self._last_update_usage: dict | None = None
        self._last_update_wall_s: float = 0.0

    # ------------------------------------------------------------------ hooks
    def init_run(self, run_ctx: dict) -> None:
        self.run_ctx = dict(run_ctx)
        state_root = Path(run_ctx["state_root"])
        state_root.mkdir(parents=True, exist_ok=True)
        self.state_dir = state_root

    @abstractmethod
    def before_episode(self, task_public_view: dict) -> RSIInjection:
        ...

    def predict_sidecar(self, *, public_observation: dict, action: dict,
                        accountant=None) -> str | None:
        """Optional pre-step prediction (WorldMind).

        Runs after the canonical agent has selected and validated the action,
        and before the environment executes it; the result never returns to the
        planner (guide 20.2).
        """
        return None

    def after_step(self, step_public_record: dict) -> None:
        return None

    @abstractmethod
    def after_episode(self, trajectory_public: dict, outcome_public: dict) -> None:
        ...

    @abstractmethod
    def snapshot(self, output_dir: Path) -> None:
        ...

    @abstractmethod
    def state_hash(self) -> str:
        ...

    def set_update_enabled(self, enabled: bool) -> None:
        self.update_enabled = bool(enabled)

    def get_last_update_usage(self) -> dict | None:
        """Tokens charged by the most recent after_episode update (or None)."""
        return self._last_update_usage

    def get_last_update_wall_s(self) -> float:
        return float(self._last_update_wall_s)

    def advance_step(self, index: int, total: int) -> None:
        """Experience-stream counters consumed by ACE's curator prompts."""
        self.run_ctx["step"] = int(index)
        self.run_ctx["total_steps"] = int(total)

    # -------------------------------------------------------------- snapshots
    def _reload_state(self) -> None:
        """Subclass hook: refresh in-memory structures from `self.state_dir`."""

    def load_snapshot(self, input_dir: Path) -> None:
        """Copy a snapshot INTO this instance's own state dir and reload it."""
        assert self.state_dir is not None, "init_run must be called before load_snapshot"
        self.copy_tree(Path(input_dir), self.state_dir)
        self._reload_state()

    def clone_from_snapshot(self, snapshot_dir: Path) -> "RSIMethod":
        """New instance + temporary copy of the snapshot, updates disabled.

        The temporary `state_root` must win over any state root inherited from the
        live run context, otherwise the probe could write into live RSI state.
        """
        clone = type(self)()
        tmp = Path(tempfile.mkdtemp(prefix=f"rsi_probe_{self.name}_"))
        clone_ctx = {
            **self.run_ctx,
            "clone_of": self.name,
            "probe": True,
            "state_root": str(tmp),
        }
        clone.init_run(clone_ctx)
        clone.load_snapshot(Path(snapshot_dir))
        clone.set_update_enabled(False)
        # hard invariants (P0-1)
        assert clone.state_dir.resolve() == tmp.resolve(), \
            "clone state dir must be the temporary probe directory"
        if self.state_dir is not None:
            assert clone.state_dir.resolve() != self.state_dir.resolve(), \
                "probe clone must not share the live state directory"
        assert clone.state_dir.resolve() != Path(snapshot_dir).resolve(), \
            "probe clone must not write into the snapshot directory"
        return clone

    # ------------------------------------------------------------------ utils
    def _guard_update(self) -> bool:
        return bool(self.update_enabled)

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    @staticmethod
    def hash_dir(path: Path) -> str:
        if not path.exists():
            return hashlib.sha256(b"<missing>").hexdigest()[:24]
        digest = hashlib.sha256()
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(file.relative_to(path)).encode())
            digest.update(file.read_bytes())
        return digest.hexdigest()[:24]

    @staticmethod
    def copy_tree(source: Path, target: Path) -> None:
        source, target = Path(source), Path(target)
        if target.exists():
            shutil.rmtree(target)
        if source.exists():
            shutil.copytree(source, target)
        else:
            target.mkdir(parents=True, exist_ok=True)
