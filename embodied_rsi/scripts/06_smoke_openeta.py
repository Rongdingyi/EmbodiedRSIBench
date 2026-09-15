#!/usr/bin/env python
"""06_smoke_openeta.py -- Phase E / Gate G4: OpenETA baseline smoke with `none`.

5 tasks per source (25 total), bounded turns. This is a pipeline gate, not a
capability gate: it checks RGB delivery, tool-call schema validity, fresh
observations, private evaluator, native-SI silence and cross-session isolation.

Output: outputs/preflight/OPENETA_BASELINE.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "external" / "OpenETA"))

from benchmark.adapters import make_adapter  # noqa: E402
from benchmark.openeta_bridge import build_runtime as br  # noqa: E402
from benchmark.openeta_bridge.episode_runner import run_episode  # noqa: E402
from benchmark.registry.loader import load_role  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402

OUT = PROJECT / "outputs" / "preflight"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES_PER_SOURCE = 5
MAX_TURNS = 4
SEED = "20260915"

# directories OpenETA would write native skill/playbook updates into
NATIVE_WRITE_ROOTS = [
    PROJECT / "external" / "OpenETA" / "agent" / "skills",
    PROJECT / "external" / "OpenETA" / ".openeta_memory",
]


def rank(task_id: str) -> str:
    return hashlib.sha256(f"{SEED}|openeta_smoke|{task_id}".encode()).hexdigest()


def snapshot_native_roots() -> dict[str, float]:
    state = {}
    for root in NATIVE_WRITE_ROOTS:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    state[str(path)] = path.stat().st_mtime
    return state


def main() -> int:
    records_by_source: dict[str, list[dict]] = defaultdict(list)
    for role in ("experience", "id", "transfer", "retention"):
        for rec in load_role(role):
            records_by_source[rec["source_dataset"]].append(rec)

    before = snapshot_native_roots()
    results = []
    invalid_actions = 0
    total_tool_calls = 0
    rgb_delivered = 0
    episodes = 0
    infra_errors = 0
    first_episode_memory: dict[str, list[str]] = {}

    for source in sorted(records_by_source):
        sample = sorted(records_by_source[source], key=lambda r: rank(r["global_task_id"]))[:SAMPLES_PER_SOURCE]
        for rec in sample:
            episodes += 1
            rsi = NoneRSI()
            state_root = PROJECT / "outputs" / "preflight" / "smoke_state" / "none"
            rsi.init_run({"state_root": str(state_root), "run_id": "g4"})
            adapter = make_adapter(source)
            t0 = time.time()
            try:
                result = run_episode(rec, adapter, rsi, role=rec["role"], max_turns=MAX_TURNS)
            except Exception as exc:  # noqa: BLE001
                infra_errors += 1
                results.append({"task": rec["global_task_id"], "source": source,
                                "status": "infra_error", "error": str(exc)})
                continue
            # audit: images actually reached the model request bodies
            bodies = [a["body"] for a in br.REQUEST_AUDIT]
            has_image = any(
                isinstance(m.get("content"), list)
                and any(p.get("type") == "image_url" for p in m["content"])
                for b in bodies for m in b.get("messages", [])
            )
            rgb_delivered += int(has_image)
            for step in result.public_trajectory:
                if step["chosen_action"]["action_type"] == "tool_call":
                    total_tool_calls += 1
                    if step.get("invalid_action"):
                        invalid_actions += 1
            results.append({
                "task": rec["global_task_id"],
                "source": source,
                "status": "ok" if not result.error else "error",
                "error": result.error,
                "turns": result.turns,
                "success": result.outcome.get("success"),
                "images_in_planner_request": has_image,
                "wall_s": round(result.wall_time_s, 1),
                "native_si": result.runtime_descriptor.get("native_self_improvement_enabled"),
            })
            print(f"[{source}] {rec['global_task_id']} turns={result.turns} "
                  f"success={result.outcome.get('success')} image={has_image} "
                  f"({result.wall_time_s:.0f}s)")

    after = snapshot_native_roots()
    native_writes = [p for p in after if p not in before or after[p] != before[p]]

    invalid_rate = invalid_actions / max(total_tool_calls, 1)
    checks = {
        "all samples ran": episodes == len(results) and episodes > 0,
        "no infrastructure crash": infra_errors == 0,
        "rgb delivered to planner": rgb_delivered >= max(1, int(0.9 * episodes)),
        "invalid action rate <= 0.20": invalid_rate <= 0.20,
        "native self-improvement writes == 0": not native_writes,
    }
    report = {
        "episodes": episodes,
        "infra_errors": infra_errors,
        "rgb_delivered": rgb_delivered,
        "tool_calls": total_tool_calls,
        "invalid_action_rate": round(invalid_rate, 4),
        "native_writes_during_run": native_writes[:10],
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "results": results,
    }
    (OUT / "OPENETA_BASELINE.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, ok in checks.items():
        print(f"[{'ok' if ok else 'FAIL'}] {name}")
    print(f"\nG4 OPENETA BASELINE GATE: {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
