"""P0: --delete-existing-run truly removes a previous run root (fail-closed otherwise)."""
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def _load_runner_8b():
    spec = importlib.util.spec_from_file_location(
        "run08b", PROJECT / "scripts" / "08b_run_pilot60_canonical.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nonempty_root_aborts_without_delete_flag(tmp_path):
    module = _load_runner_8b()
    root = tmp_path / "run"
    (root / "rsi_state").mkdir(parents=True)
    (root / "rsi_state" / "playbook.txt").write_text("old state")
    with pytest.raises(SystemExit):
        module.prepare_run_root(root, delete_existing=False)
    assert (root / "rsi_state" / "playbook.txt").exists()


def test_delete_existing_run_removes_old_state(tmp_path):
    module = _load_runner_8b()
    root = tmp_path / "run"
    (root / "rsi_state").mkdir(parents=True)
    (root / "rsi_state" / "playbook.txt").write_text("old state")
    (root / "rsi_updates.jsonl").write_text("{}\n")
    module.prepare_run_root(root, delete_existing=True)
    assert root.exists()
    assert list(root.iterdir()) == []
