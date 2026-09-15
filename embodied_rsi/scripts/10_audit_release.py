#!/usr/bin/env python
"""10_audit_release.py -- final hard checks (guide section 58).

Asserts every gate artifact and only then writes FINAL_STATUS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
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

    # pilot completion
    completed = []
    probe_mutations = 0
    if PILOT.exists():
        for method_dir in PILOT.iterdir():
            if not method_dir.is_dir():
                continue
            for seed_dir in method_dir.glob("seed_*"):
                metrics_file = seed_dir / "metrics.json"
                status_file = seed_dir / "FINAL_STATUS.txt"
                if metrics_file.exists() and status_file.exists():
                    metrics = json.loads(metrics_file.read_text())
                    if metrics.get("probe_state_unchanged") is False:
                        probe_mutations += 1
                    completed.append(method_dir.name)
    check("probe mutation count == 0", probe_mutations == 0, str(probe_mutations))
    primary = {"none", "raw_memory", "ace_context", "worldmind"}
    missing = sorted(primary - set(completed))
    check("all primary conditions completed", not missing, f"missing: {missing}")

    status = "PASS" if not FAILS else "FAIL"
    lines = [f"FINAL_STATUS: {status}"]
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
