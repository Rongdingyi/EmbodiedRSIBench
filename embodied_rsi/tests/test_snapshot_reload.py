"""Gate: snapshot hash == reload hash and identical retrieval injection."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.rsi.ace_context import AceContextRSI  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from benchmark.rsi.raw_memory import RawMemoryRSI  # noqa: E402


def _simulate(method, n=3):
    for i in range(n):
        method.after_episode(
            {"episode_id": f"e{i}", "instruction": f"pick up object {i}",
             "source_dataset": "x", "actions": [{"action": "pick", "public_feedback": "ok"}]},
            {"success": i % 2 == 0, "terminated": True, "num_steps": 1})


def test_snapshot_reload(tmp=Path("/tmp/opencode/rsi_snapshot_test")):
    for cls in (NoneRSI, RawMemoryRSI, AceContextRSI):
        method = cls()
        method.init_run({"state_root": str(tmp / cls.__name__), "run_id": "t"})
        _simulate(method)
        snap = tmp / f"{cls.__name__}_snap"
        method.snapshot(snap)
        h_a = method.state_hash()
        reloaded = cls()
        reloaded.run_ctx = {"state_root": str(snap)}
        reloaded.load_snapshot(snap)
        h_b = reloaded.state_hash()
        assert h_a == h_b, f"{cls.__name__}: snapshot hash {h_a} != reload {h_b}"
        inj_a = method.before_episode({"instruction": "pick up object 0"})
        inj_b = reloaded.before_episode({"instruction": "pick up object 0"})
        assert inj_a.context_text == inj_b.context_text, f"{cls.__name__}: injection differs"
    return True


if __name__ == "__main__":
    assert test_snapshot_reload()
    print("test_snapshot_reload: PASS")
