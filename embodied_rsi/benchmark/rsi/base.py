"""Unified RSI plugin interface (guide section 14).

Hard rules:
  * an RSIMethod never holds a simulator object;
  * it reads only public task / trajectory / feedback / outcome data;
  * it cannot touch the environment, tool registry, evaluator or OpenETA core;
  * all cross-episode state lives in its own `rsi_state/` directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from benchmark.openeta_bridge.context_injection import RSIInjection, make_injection


class RSIMethod(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.update_enabled: bool = True
        self.run_ctx: dict = {}
        self.state_dir: Path | None = None

    # ------------------------------------------------------------------ hooks
    def init_run(self, run_ctx: dict) -> None:
        self.run_ctx = dict(run_ctx)
        state_root = Path(run_ctx["state_root"])
        self.state_dir = state_root
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def before_episode(self, task_public_view: dict) -> RSIInjection:
        ...

    def after_step(self, step_public_record: dict) -> None:
        return None

    @abstractmethod
    def after_episode(self, trajectory_public: dict, outcome_public: dict) -> None:
        ...

    @abstractmethod
    def snapshot(self, output_dir: Path) -> None:
        ...

    @abstractmethod
    def load_snapshot(self, input_dir: Path) -> None:
        ...

    @abstractmethod
    def state_hash(self) -> str:
        ...

    def set_update_enabled(self, enabled: bool) -> None:
        self.update_enabled = bool(enabled)

    # ------------------------------------------------------------------ utils
    def clone_from_snapshot(self, snapshot_dir: Path) -> "RSIMethod":
        clone = copy.deepcopy(self)
        clone.load_snapshot(snapshot_dir)
        clone.set_update_enabled(False)
        return clone

    def _guard_update(self) -> bool:
        return bool(self.update_enabled)

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    @staticmethod
    def hash_dir(path: Path) -> str:
        """Deterministic hash of a state directory (all files, sorted)."""
        if not path.exists():
            return hashlib.sha256(b"<missing>").hexdigest()[:24]
        digest = hashlib.sha256()
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(file.relative_to(path)).encode())
            digest.update(file.read_bytes())
        return digest.hexdigest()[:24]

    @staticmethod
    def copy_tree(source: Path, target: Path) -> None:
        if target.exists():
            shutil.rmtree(target)
        if source.exists():
            shutil.copytree(source, target)
        else:
            target.mkdir(parents=True, exist_ok=True)
