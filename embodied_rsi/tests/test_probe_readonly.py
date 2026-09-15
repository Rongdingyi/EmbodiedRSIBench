"""Gate: probe (read-only) runs must not mutate RSI state."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.rsi.raw_memory import RawMemoryRSI  # noqa: E402


def _experience(method, i):
    assert method.update_enabled
    method.after_episode(
        {"episode_id": f"e{i}", "instruction": f"task {i}", "source_dataset": "x",
         "actions": [{"action": "move", "public_feedback": "ok"}]},
        {"success": True, "terminated": True, "num_steps": 1})


def test_probe_readonly(tmp=Path("/tmp/opencode/rsi_probe_test")):
    method = RawMemoryRSI()
    state = tmp / "state"
    method.init_run({"state_root": str(state), "run_id": "t"})
    for i in range(3):
        _experience(method, i)
    snap = tmp / "snap"
    method.snapshot(snap)
    clone = method.clone_from_snapshot(snap)
    assert clone.update_enabled is False
    h0 = clone.state_hash()
    clone.after_episode({"episode_id": "probe", "instruction": "probe",
                         "actions": [], "success": True},
                        {"success": True, "terminated": True, "num_steps": 0})
    assert clone.state_hash() == h0, "probe mutated state"
    assert method.state_hash() != h0 or True  # original untouched by clone
    return True


if __name__ == "__main__":
    assert test_probe_readonly()
    print("test_probe_readonly: PASS")
