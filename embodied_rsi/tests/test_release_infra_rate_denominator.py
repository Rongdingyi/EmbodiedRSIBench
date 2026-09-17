"""P1: release-audit infra rates use honest denominators (episodes vs attempts)."""
import importlib.util
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "audit10b", PROJECT / "scripts" / "10b_audit_canonical_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_infra_rates_count_attempts_and_episodes_separately(tmp_path):
    module = _load_audit()
    root = tmp_path / "pilot60_canonical" / "none" / "seed_20260915"
    (root / "probes" / "S000").mkdir(parents=True)
    updates = [
        {"attempts": 1, "infrastructure_error": "WorkerError: timeout"},
        {"attempts": 2, "infrastructure_error": None},
    ]
    (root / "rsi_updates.jsonl").write_text(
        "\n".join(json.dumps(u) for u in updates) + "\n")
    summary = {"tasks": {
        "t1": {"role": "id", "status": "PASS", "success": True, "attempts": 1},
        "t2": {"role": "id", "status": "FAIL", "success": None,
               "error": "WorkerError: reset", "attempts": 2},
        "t3": {"role": "id", "status": "PASS", "success": False, "attempts": 1},
    }}
    (root / "probes" / "S000" / "summary.json").write_text(json.dumps(summary))

    rates = module.summarize_infra(tmp_path / "pilot60_canonical")
    assert rates["experience_episodes"] == 2
    assert rates["experience_attempts"] == 3
    assert rates["experience_infra_errors"] == 1
    assert rates["probe_infra_errors"] == 1
    assert rates["probe_attempts"] == 4
    # experience-only denominator must NOT be inflated by probe errors
    assert rates["rate_experience_episodes"] == 0.5
    # all-attempts denominator: (1 + 1) errors / (3 + 4) attempts
    assert abs(rates["rate_all_attempts"] - (2 / 7)) < 1e-9
