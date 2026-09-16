#!/usr/bin/env python
"""06_smoke_openeta.py -- G4 OpenETA baseline smoke with `none`.

5 tasks per source (25 total) through the upstream OpenEtaEpisodeRunner.
Pipeline gate (not a capability gate): RGB delivery, tool schema validity,
fresh observations, private evaluator, native-SI silence, leakage-free requests.

Output: outputs/preflight/OPENETA_BASELINE.json  (PASS/FAIL)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "external" / "OpenETA"))

from benchmark.adapters import make_adapter  # noqa: E402
from benchmark.openeta_bridge.episode_runner import run_episode  # noqa: E402
from benchmark.registry.loader import load_role  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from benchmark.runner import gates  # noqa: E402

OUT = PROJECT / "outputs" / "preflight"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES_PER_SOURCE = 5
SEED = "20260915"

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
    protocol = yaml.safe_load((PROJECT / "configs" / "pilot150.yaml").read_text())
    budgets = protocol["budgets"]
    records_by_source: dict[str, list[dict]] = defaultdict(list)
    for role in ("experience", "id", "transfer", "retention"):
        for rec in load_role(role):
            records_by_source[rec["source_dataset"]].append(rec)

    before = snapshot_native_roots()
    results = []
    episode_index: dict[str, str] = {}
    episode_counter = 0
    invalid_actions = 0
    total_tool_calls = 0
    rgb_delivered = 0
    episodes = 0
    infra_errors = 0
    episode_failures = 0
    leakage_total = 0

    for source in sorted(records_by_source):
        sample = sorted(records_by_source[source],
                        key=lambda r: rank(r["global_task_id"]))[:SAMPLES_PER_SOURCE]
        for rec in sample:
            episodes += 1
            episode_counter += 1
            rsi = NoneRSI()
            rsi.init_run({"state_root": str(OUT / "smoke_state" / "none"), "run_id": "g4"})
            adapter = make_adapter(source)
            # neutral dir name: artifact paths are visible to the planner, so they
            # must not carry benchmark identifiers (mapping kept in a side file)
            ep_dir = OUT / "openeta_baseline" / f"ep{episode_counter:03d}"
            episode_index[str(ep_dir)] = rec["global_task_id"]
            res = run_episode(
                rec, adapter, rsi, role=rec["role"], output_dir=ep_dir,
                config={"max_turns": 4, "max_tool_calls": 16, "timeout_s": 600,
                        "max_total_tokens": 1_500_000,
                        "planner_max_output_tokens": budgets["planner_max_output_tokens"],
                        "max_env_steps": budgets["max_env_steps"].get(source)},
            )
            if res.error:
                infra_errors += 1
            if res.status != gates.PASS:
                episode_failures += 1
            leakage_total += len(res.leakage_violations)
            dump = ep_dir / "public_context_dump.jsonl"
            has_image = False
            if dump.exists():
                for line in dump.read_text().splitlines():
                    if '"has_image": true' in line:
                        has_image = True
                        break
            rgb_delivered += int(has_image)
            for step in res.public_trajectory:
                if step["chosen_action"]["action_type"] == "tool_call":
                    total_tool_calls += 1
            results.append({
                "task": rec["global_task_id"], "source": source,
                "status": res.status, "error": res.error, "turns": res.turns,
                "env_steps": res.env_steps, "success": res.outcome.get("success"),
                "images_in_planner_request": has_image,
                "wall_s": round(res.wall_time_s, 1),
            })
            print(f"[{source}] {rec['global_task_id']} turns={res.turns} "
                  f"env_steps={res.env_steps} success={res.outcome.get('success')} "
                  f"image={has_image} status={res.status} ({res.wall_time_s:.0f}s)")

    after = snapshot_native_roots()
    native_writes = [p for p in after if p not in before or after[p] != before[p]]
    checks = {
        "all samples ran": episodes == len(results) and episodes > 0,
        "no infrastructure error": infra_errors == 0,
        "no episode-level failure": episode_failures == 0,
        "rgb delivered to planner (>=90%)": rgb_delivered >= max(1, int(0.9 * episodes)),
        "no private leakage in requests": leakage_total == 0,
        "native self-improvement writes == 0": not native_writes,
    }
    status = gates.PASS if all(checks.values()) else gates.FAIL
    report = {
        "episodes": episodes,
        "infra_errors": infra_errors,
        "episode_failures": episode_failures,
        "rgb_delivered": rgb_delivered,
        "tool_calls": total_tool_calls,
        "leakage_violations": leakage_total,
        "native_writes_during_run": native_writes[:10],
        "checks": checks,
        "status": status,
        "results": results,
    }
    (OUT / "OPENETA_BASELINE.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "OPENETA_BASELINE_EPISODE_INDEX.json").write_text(
        json.dumps({"note": "maps neutral episode dirs to task ids (never model-visible)",
                    "map": episode_index}, indent=2) + "\n")
    for name, ok in checks.items():
        print(f"[{'ok' if ok else 'FAIL'}] {name}")
    print(f"\nG4 OPENETA BASELINE GATE: {status}")
    return 0 if status == gates.PASS else 1


if __name__ == "__main__":
    sys.exit(main())
