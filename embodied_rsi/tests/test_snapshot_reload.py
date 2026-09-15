"""Gate: snapshot hash == reload hash, same injection, new-instance clones."""
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.rsi.ace_context import AceContextRSI  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from benchmark.rsi.raw_memory import RawMemoryRSI  # noqa: E402


def _simulate(method, n=3):
    method.after_episode(
        {"episode_id": "e1", "instruction": "pick up object 1",
         "source_dataset": "x", "actions": [{"action": "pick", "public_feedback": "ok"}]},
        {"success": True, "terminated": True, "num_steps": 1})


def test_snapshot_reload():
    tmp = Path(tempfile.mkdtemp(prefix="rsi_snapshot_test_"))
    for cls in (NoneRSI, RawMemoryRSI, AceContextRSI):
        method = cls()
        method.init_run({"state_root": str(tmp / cls.__name__), "run_id": "t"})
        _simulate(method)
        snap = tmp / f"{cls.__name__}_snap"
        method.snapshot(snap)
        h_a = method.state_hash()
        # reload into a fresh instance's own directory (P0-4 semantics)
        reloaded = cls()
        reloaded.init_run({"state_root": str(tmp / f"{cls.__name__}_reload"), "run_id": "t"})
        reloaded.load_snapshot(snap)
        assert reloaded.state_hash() == h_a, f"{cls.__name__}: reload hash mismatch"
        inj_a = method.before_episode({"instruction": "pick up object 1"})
        inj_b = reloaded.before_episode({"instruction": "pick up object 1"})
        assert inj_a.context_text == inj_b.context_text, f"{cls.__name__}: injection differs"
        # clone must be a new instance operating on a temp copy, updates disabled
        clone = method.clone_from_snapshot(snap)
        assert clone is not method and clone.update_enabled is False
        assert clone.state_hash() == h_a
        clone.after_episode({"episode_id": "probe", "instruction": "probe",
                             "actions": [], "success": True},
                            {"success": True, "terminated": True, "num_steps": 0})
        assert clone.state_hash() == h_a, "clone mutated its own copy on a probe update"
        assert method.state_hash() == h_a, "clone wrote back into the source snapshot"
    return True


if __name__ == "__main__":
    assert test_snapshot_reload()
    print("test_snapshot_reload: PASS")
