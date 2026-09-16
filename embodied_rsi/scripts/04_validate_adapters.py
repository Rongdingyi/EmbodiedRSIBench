#!/usr/bin/env python
"""04_validate_adapters.py -- Phase D / Gate G3.

20 tasks per source (100 total): reset -> RGB check -> one legal action ->
fresh observation -> private evaluator -> close. Crash rate must be <= 2% and
the private-leakage scan must be clean.

Output: outputs/preflight/ADAPTER_VALIDATION.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.adapters import make_adapter  # noqa: E402
from benchmark.registry.loader import load_role  # noqa: E402
from benchmark.runner import gates  # noqa: E402

OUT = PROJECT / "outputs" / "preflight"
OUT.mkdir(parents=True, exist_ok=True)

FORBIDDEN_KEYS = {
    "golden_actions", "goal_preds", "success_conditions", "target_pose",
    "physical_task_id", "transfer_pair_id", "retention_anchor_id",
    "private_eval_metadata", "target", "goal",
}
SAMPLES_PER_SOURCE = 20
SEED = "20260915"


def rank(task_id: str) -> str:
    return hashlib.sha256(f"{SEED}|adapter_smoke|{task_id}".encode()).hexdigest()


def one_legal_action(source: str, tools: list[dict]) -> tuple[str, dict]:
    by_name = {t["name"]: t for t in tools}
    pick = {
        "TVRBench": ("RotateRight", {}),
        "SpatialWorld": ("Rotate", {"direction": "right", "degrees": 90}),
        "AsgardBench": ("find", {"object": "CounterTop"}),
        "EB-ALFRED": ("find", {"object": "CounterTop"}),
        "EB-Habitat": (None, {}),
    }[source]
    if source == "EB-Habitat":
        name = tools[0]["name"]
        return name, {}
    return pick


def scan_leakage(payload: dict, where: str, hits: list[str]) -> None:
    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in FORBIDDEN_KEYS:
                    hits.append(f"{where}:{path}/{key}")
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node[:20]):
                walk(value, f"{path}[{i}]")

    walk(payload)


def main() -> int:
    records_by_source: dict[str, list[dict]] = defaultdict(list)
    for role in ("experience", "id", "transfer", "retention"):
        for rec in load_role(role):
            records_by_source[rec["source_dataset"]].append(rec)

    results = []
    leakage_hits: list[str] = []
    crashes = 0
    total = 0
    schema_failures = 0

    for source, records in sorted(records_by_source.items()):
        records = sorted(records, key=lambda r: rank(r["global_task_id"]))
        # spread across roles: take a deterministic mix
        sample = records[:SAMPLES_PER_SOURCE]
        for rec in sample:
            total += 1
            entry = {"source": source, "task": rec["global_task_id"],
                     "role": rec["role"], "status": "ok", "steps": 0}
            # fresh worker per task (P2): no cross-task simulator/controller reuse
            adapter = make_adapter(source)
            try:
                try:
                    obs = adapter.reset(rec)
                except Exception as first_exc:  # one retry for transient simulator timeouts
                    if "timed out" not in str(first_exc).lower():
                        raise
                    adapter.close()
                    adapter = make_adapter(source)
                    obs = adapter.reset(rec)
                assert obs.images, "reset returned no RGB"
                frame = obs.images[0]
                assert getattr(frame, "size", 0) > 0, "empty RGB"
                scan_leakage(obs.public_metadata, f"{source}:reset", leakage_hits)

                tools = adapter.build_tool_specs(rec)
                if not tools:
                    schema_failures += 1
                    raise AssertionError("tool schema empty after reset")
                entry["tool_count"] = len(tools)
                name, params = one_legal_action(source, tools)
                out = adapter.step(name, params)
                assert out.observation.images, "step returned no RGB"
                scan_leakage(out.observation.public_metadata, f"{source}:step", leakage_hits)
                entry["steps"] = 1
                entry["action_success"] = out.action_success

                evaluation = adapter.private_evaluate()
                assert "success" in evaluation, "private evaluator returned no success field"
                entry["private_success"] = evaluation.get("success")
            except Exception as exc:  # noqa: BLE001
                crashes += 1
                entry["status"] = "crash"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()[-1500:]
            finally:
                try:
                    adapter.close()
                except Exception:
                    pass
            results.append(entry)
        print(f"[{source}] done: {sum(1 for r in results if r['source'] == source)} tasks")

    crash_rate = crashes / max(total, 1)
    checks = {
        "crash rate <= 2%": crash_rate <= 0.02,
        "no private leakage": not leakage_hits,
        "non-empty tool schema for every task": schema_failures == 0,
    }
    status = gates.PASS if all(checks.values()) else gates.FAIL
    report = {
        "tasks_total": total,
        "crashes": crashes,
        "crash_rate": round(crash_rate, 4),
        "schema_failures": schema_failures,
        "leakage_hits": leakage_hits,
        "checks": checks,
        "status": status,
        "per_source": {s: sum(1 for r in results if r["source"] == s)
                       for s in sorted(records_by_source)},
        "results": results,
    }
    (OUT / "ADAPTER_VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nG3 ADAPTER GATE: {status}  (total={total} crashes={crashes} "
          f"crash_rate={crash_rate:.3f} schema_failures={schema_failures} "
          f"leakage={len(leakage_hits)})")
    return 0 if status == gates.PASS else 1


if __name__ == "__main__":
    sys.exit(main())
