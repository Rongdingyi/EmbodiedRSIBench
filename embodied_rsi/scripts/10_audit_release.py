#!/usr/bin/env python
"""10_audit_release.py -- final hard checks (guide section 58).

Asserts every gate artifact and only then writes FINAL_STATUS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from benchmark.runner import gates  # noqa: E402
OUT = PROJECT / "outputs" / "preflight"
PILOT = PROJECT / "outputs" / "pilot150"

FAILS: list[str] = []


def load(path: Path) -> dict:
    if not path.exists():
        FAILS.append(f"missing artifact: {path}")
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        FAILS.append(f"unreadable artifact {path}: {exc}")
        return {}


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + ("" if cond else f"  ({detail})"))
    if not cond:
        FAILS.append(f"{name}: {detail}")


def main() -> int:
    dataset = load(OUT / "DATASET_AUDIT.json")
    check("dataset gate PASS", dataset.get("status") == "PASS", str(dataset.get("status")))

    freeze = load(OUT / "OPENETA_FREEZE.json")
    check("freeze gate PASS", freeze.get("status") == "PASS", str(freeze.get("status")))

    api = load(PROJECT / "outputs" / "api_smoke" / "deepseek_flash.json")
    check("model gate PASS", api.get("status") == "PASS", str(api.get("status")))

    adapters = load(OUT / "ADAPTER_VALIDATION.json")
    if adapters:
        check("adapter crash rate < 2%", adapters.get("crash_rate", 1.0) < 0.02,
              str(adapters.get("crash_rate")))
        check("adapter leakage == 0", not adapters.get("leakage_hits"),
              str(adapters.get("leakage_hits")[:2]))
    else:
        check("adapter validation artifact present", False, "ADAPTER_VALIDATION.json missing")

    baseline = load(OUT / "OPENETA_BASELINE.json")
    if baseline:
        check("baseline gate PASS", baseline.get("status") == "PASS", str(baseline.get("status")))

    rsi_smoke = load(OUT / "RSI_SMOKE.json")
    if rsi_smoke:
        check("rsi smoke gate PASS", rsi_smoke.get("status") == "PASS",
              str(rsi_smoke.get("status")))

    # pilot completion, two-level status (review amendment)
    method_status: dict[str, str] = {}
    probe_mutations = 0
    snapshot_failures = 0
    infra_crashes = 0
    leakage_violations = 0
    total_episodes = 0
    if PILOT.exists():
        for method_dir in PILOT.iterdir():
            if not method_dir.is_dir():
                continue
            for seed_dir in method_dir.glob("seed_*"):
                metrics_file = seed_dir / "metrics.json"
                if not metrics_file.exists():
                    continue
                metrics = json.loads(metrics_file.read_text())
                method_status[method_dir.name] = metrics.get("status", gates.FAIL)
                if metrics.get("probe_state_unchanged") is False:
                    probe_mutations += 1
                if metrics.get("probe_state_unchanged") is None:
                    snapshot_failures += 1
                infra_crashes += int(metrics.get("infra_errors") or 0)
                leakage_violations += int(metrics.get("leakage_violations") or 0)
                total_episodes += int(metrics.get("experience") or 0)
    check("probe mutation count == 0", probe_mutations == 0, str(probe_mutations))
    check("snapshot reload failures == 0", snapshot_failures == 0, str(snapshot_failures))
    check("private leakage violations == 0", leakage_violations == 0, str(leakage_violations))

    primary = {"none", "raw_memory", "ace_context", "worldmind"}
    completed = set(method_status)
    missing = sorted(primary - completed)
    check("all primary conditions completed", not missing, f"missing: {missing}")

    # blocked methods must be declared as BLOCKED (never silently PASS)
    embodiskill_status = method_status.get("embodiskill", gates.BLOCKED)
    check("embodiskill recorded as BLOCKED (not PASS)", embodiskill_status == gates.BLOCKED,
          embodiskill_status)

    infra_rate = infra_crashes / max(total_episodes, 1)
    check("infrastructure crash rate < 2%", infra_rate < 0.02,
          f"{infra_crashes}/{total_episodes}")

    status = gates.FAIL if FAILS else gates.release_status(
        {**method_status, "embodiskill": embodiskill_status})
    lines = [f"FINAL_STATUS: {status}",
             f"METHOD_STATUS: {json.dumps(method_status, sort_keys=True)}"]
    if FAILS:
        lines += [f"- {f}" for f in FAILS]
    (PROJECT / "outputs" / "FINAL_STATUS.txt").write_text("\n".join(lines) + "\n")
    if PILOT.exists():
        (PILOT / "FINAL_STATUS.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
