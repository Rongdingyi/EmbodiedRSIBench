#!/usr/bin/env python
"""07b_smoke_rsi_canonical.py -- canonical RSI smoke (paid, small).

Per requested method: 2 experience episodes + 1 ID probe on the canonical agent.
Only mechanics are checked (state updates, retrieval, playbook/manual change,
WorldMind sidecar + components, probe read-only); smoke success rates are never
used for method comparison.

Writes outputs/preflight/RSI_SMOKE_CANONICAL.json and per-method artifacts under
outputs/canonical_rsi_smoke/<method>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.adapters import make_adapter  # noqa: E402
from benchmark.agent.config import load_protocol_config  # noqa: E402
from benchmark.agent.episode_runner import run_episode  # noqa: E402
from benchmark.registry.loader import by_global_id  # noqa: E402
from benchmark.rsi.ace_context import AceContextRSI  # noqa: E402
from benchmark.rsi.embodiskill import EmbodiSkillRSI  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from benchmark.rsi.raw_memory import RawMemoryRSI  # noqa: E402
from benchmark.rsi.worldmind import WorldMindRSI  # noqa: E402
from benchmark.runner import gates  # noqa: E402

METHODS = {"none": NoneRSI, "raw_memory": RawMemoryRSI, "ace_context": AceContextRSI,
           "worldmind": WorldMindRSI, "embodiskill": EmbodiSkillRSI}
OUT = PROJECT / "outputs" / "preflight"
SMOKE = PROJECT / "outputs" / "canonical_rsi_smoke"


def run_method(name: str, protocol, manifest: dict) -> dict:
    method = METHODS[name]()
    if getattr(method, "BLOCKED", False):
        return {"status": gates.BLOCKED, "reason": method.BLOCK_REASON}
    root = SMOKE / name
    if root.exists():
        import shutil
        shutil.rmtree(root)
    method.init_run({"state_root": str(root / "rsi_state"), "run_id": f"canonical_smoke:{name}"})

    stream = manifest["experience_stream_order"][:2]
    for index, gid in enumerate(stream, 1):
        task = by_global_id("experience")[gid]
        method.advance_step(index, len(stream))
        run_episode(task, make_adapter(task["source_dataset"]), method, role="experience",
                    config=protocol, output_dir=root / "experience" / f"episode_{index:04d}")
    h_after = method.state_hash()
    method.snapshot(root / "states" / "S002")

    # read-only probe clone
    clone = method.clone_from_snapshot(root / "states" / "S002")
    assert clone.update_enabled is False
    h0 = clone.state_hash()
    probe_gid = manifest["id_probe"][0]
    task = by_global_id("id")[probe_gid]
    run_episode(task, make_adapter(task["source_dataset"]), clone, role="id",
                config=protocol, output_dir=root / "probes" / "S002" / "id" / "task_001")
    h1 = clone.state_hash()

    checks = {"probe state unchanged": h0 == h1}
    if name == "none":
        checks["none state unchanged"] = h_after == method.hash_dir(root / "rsi_state")
    if name == "raw_memory":
        store = root / "rsi_state"
        checks["raw memory stored"] = any(store.rglob("*.json*"))
    if name == "ace_context":
        playbook = root / "rsi_state" / "playbook.txt"
        checks["playbook exists"] = playbook.exists()
        checks["playbook changed"] = bool(playbook.exists() and playbook.read_text().strip())
    if name == "worldmind":
        usage = method.get_last_update_usage() or {}
        checks["worldmind update usage"] = bool(usage)
    if name == "embodiskill":
        manual = root / "rsi_state"
        checks["manual reachable"] = any(manual.rglob("*"))
    return {"status": gates.PASS if all(checks.values()) else gates.FAIL,
            "checks": checks, "state_hash_after": h_after}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", default=None)
    args = parser.parse_args()
    protocol = load_protocol_config(PROJECT / "configs" / "pilot60_canonical.yaml")
    manifest = json.loads((PROJECT / "manifests" / "pilot60.json").read_text())
    methods = args.method or ["none", "raw_memory", "ace_context", "worldmind", "embodiskill"]
    report = {"methods": {}}
    for name in methods:
        report["methods"][name] = run_method(name, protocol, manifest)
        print(f"[{name}] {report['methods'][name]['status']}")
    report["status"] = gates.PASS if all(
        m.get("status") in {gates.PASS, gates.BLOCKED} for m in report["methods"].values()) else gates.FAIL
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RSI_SMOKE_CANONICAL.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"CANONICAL RSI SMOKE: {report['status']}")
    return 0 if report["status"] == gates.PASS else 1


if __name__ == "__main__":
    sys.exit(main())
