"""Gate: BLOCKED is never coerced into PASS; two-level release status."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from benchmark.runner import gates  # noqa: E402


def test_combine():
    assert gates.combine({"a": gates.PASS, "b": gates.PASS}) == gates.PASS
    assert gates.combine({"a": gates.PASS, "b": gates.BLOCKED}) == gates.BLOCKED
    assert gates.combine({"a": gates.BLOCKED, "b": gates.FAIL}) == gates.FAIL
    return True


def test_release_status():
    full = {"none": gates.PASS, "raw_memory": gates.PASS, "ace_context": gates.PASS,
            "worldmind": gates.PASS, "embodiskill": gates.PASS}
    assert gates.release_status(full) == gates.FULL_PILOT_PASS
    with_blocker = {**full, "embodiskill": gates.BLOCKED}
    assert gates.release_status(with_blocker) == gates.PIPELINE_PASS_WITH_BLOCKER
    failed = {**full, "worldmind": gates.FAIL}
    assert gates.release_status(failed) == gates.FAIL
    # a blocked method can never become a full pass
    assert gates.release_status(with_blocker) != gates.FULL_PILOT_PASS
    return True


if __name__ == "__main__":
    assert test_combine() and test_release_status()
    print("test_gate_status: PASS")
