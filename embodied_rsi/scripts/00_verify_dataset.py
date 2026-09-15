#!/usr/bin/env python
"""00_verify_dataset.py -- Phase A / Gate G0: verify the Core v1.0 delivery.

Reads release parquet files (via EMBODIED_RSI_DATA, default ../data -> the
EmbodiedRSIBench/data symlink) and hard-asserts every number the pilot guide
requires. Any mismatch => exit != 0 and status FAIL.

Output: outputs/preflight/DATASET_AUDIT.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("EMBODIED_RSI_DATA", PROJECT.parent / "data")).resolve()
OUT = PROJECT / "outputs" / "preflight"
OUT.mkdir(parents=True, exist_ok=True)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILS.append(f"{name}: {detail}")
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + ("" if cond else f"  ({detail})"))


def rows(rel: str) -> list[dict]:
    path = DATA / rel
    if not path.exists():
        raise SystemExit(f"[FAIL] missing dataset file: {path}")
    return pq.read_table(path).to_pylist()


def main() -> int:
    print(f"dataset root: {DATA}")

    raw = rows("registry/task_registry_raw.parquet")
    normalized = rows("registry/task_registry_normalized.parquet")
    selected = rows("core_v1/selected_core.parquet")
    experience = rows("core_v1/experience.parquet")
    id_probe = rows("core_v1/id_probe.parquet")
    transfer_probe = rows("core_v1/transfer_probe.parquet")
    retention_probe = rows("core_v1/retention_probe.parquet")
    id_pairs = rows("core_v1/id_pairs.parquet")
    transfer_pairs = rows("core_v1/transfer_pairs.parquet")

    check("raw registry == 8046", len(raw) == 8046, str(len(raw)))
    check("normalized registry == 8046", len(normalized) == 8046, str(len(normalized)))
    check("selected_core == 2200", len(selected) == 2200, str(len(selected)))
    check("experience == 1500", len(experience) == 1500, str(len(experience)))
    check("id_probe == 300", len(id_probe) == 300, str(len(id_probe)))
    check("transfer_probe == 300", len(transfer_probe) == 300, str(len(transfer_probe)))
    check("retention_probe == 100", len(retention_probe) == 100, str(len(retention_probe)))
    check("id_pairs == 300", len(id_pairs) == 300, str(len(id_pairs)))
    check("transfer_pairs == 300", len(transfer_pairs) == 300, str(len(transfer_pairs)))

    pids = [r["physical_task_id"] for r in selected]
    check("selected_core physical_task_id nunique == 2200", len(set(pids)) == 2200, str(len(set(pids))))

    roles = {
        "experience": {r["physical_task_id"] for r in experience},
        "id_probe": {r["physical_task_id"] for r in id_probe},
        "transfer_probe": {r["physical_task_id"] for r in transfer_probe},
        "retention_probe": {r["physical_task_id"] for r in retention_probe},
    }
    names = list(roles)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = roles[a] & roles[b]
            overlaps[f"{a}x{b}"] = len(ov)
    check("role physical_task_id pairwise disjoint", all(v == 0 for v in overlaps.values()),
          str(overlaps))

    core_by_source = Counter(r["source_dataset"] for r in selected)
    expected_core = {"TVRBench": 1408, "SpatialWorld": 410, "AsgardBench": 104,
                     "EB-ALFRED": 137, "EB-Habitat": 141}
    for src, n in expected_core.items():
        check(f"core quota {src} == {n}", core_by_source.get(src) == n, str(core_by_source.get(src)))
    check("core total == 2200", sum(core_by_source.values()) == 2200, str(sum(core_by_source.values())))

    check("source manifest exists", (DATA / "source_manifest.json").exists())
    stats = json.loads((DATA / "stats.json").read_text()) if (DATA / "stats.json").exists() else {}
    check("stats.json raw_entries == 8046", stats.get("raw_entries") == 8046)
    check("stats.json core_tasks == 2200", stats.get("core_tasks") == 2200)

    audit = {
        "raw_count": len(raw),
        "normalized_count": len(normalized),
        "core_count": len(selected),
        "roles": {k: len(v) for k, v in roles.items()},
        "role_source_matrix": {
            src: {role: sum(1 for r in {"experience": experience, "id_probe": id_probe,
                                        "transfer_probe": transfer_probe,
                                        "retention_probe": retention_probe}[role]
                           if r["source_dataset"] == src)
                  for role in ("experience", "id_probe", "transfer_probe", "retention_probe")}
            for src in expected_core
        },
        "physical_overlap": overlaps,
        "id_pairs": len(id_pairs),
        "transfer_pairs": len(transfer_pairs),
        "status": "PASS" if not FAILS else "FAIL",
        "failures": FAILS,
    }
    (OUT / "DATASET_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")

    print()
    print(f"G0 DATASET GATE: {audit['status']}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
