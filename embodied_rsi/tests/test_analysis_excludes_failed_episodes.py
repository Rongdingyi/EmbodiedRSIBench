"""P1: success rates / paired transitions never count non-PASS (protocol/infra) episodes."""
import importlib.util
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def _load_analysis():
    spec = importlib.util.spec_from_file_location(
        "analyze09", PROJECT / "scripts" / "09_analyze_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_success_rate_excludes_failed_episodes():
    module = _load_analysis()
    summary = {"tasks": {
        "t1": {"role": "id", "status": "PASS", "success": True},
        "t2": {"role": "id", "status": "FAIL", "success": True},   # infra-failed
        "t3": {"role": "id", "status": "PASS", "success": False},
        "t4": {"role": "id", "status": "PASS", "success": None},   # not scored
    }}
    rate, n = module.success_rate(summary, "id")
    assert n == 2
    assert rate == 0.5


def test_paired_transitions_require_pass_on_both_sides():
    module = _load_analysis()
    first = {"tasks": {
        "a": {"role": "id", "status": "PASS", "success": False},
        "b": {"role": "id", "status": "FAIL", "success": False},   # protocol-broken
    }}
    last = {"tasks": {
        "a": {"role": "id", "status": "PASS", "success": True},
        "b": {"role": "id", "status": "PASS", "success": True},
    }}
    counts = module.paired_transitions(first, last, "id")
    assert counts["n"] == 1
    assert counts["fail_to_success"] == 1
    assert counts["net_gain"] == 1
