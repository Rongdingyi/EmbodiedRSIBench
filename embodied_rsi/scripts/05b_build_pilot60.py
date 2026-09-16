#!/usr/bin/env python
"""05b_build_pilot60.py -- Pilot-60 manifest (Pilot v0.7 protocol).

Composition (user-approved protocol, supersedes Pilot-150):
    Experience      30  (5 sources x 6)
    ID probe        10  (5 sources x 2)
    Transfer probe  10  (TVR 3 / SpatialWorld 3 / EB-ALFRED 2 / EB-Habitat 2)
    Retention probe 10  (5 sources x 2)
    total           60 unique tasks, two measurement points S000 -> S030.

Selection reuses the Pilot-150 ranking namespaces, so Pilot-60 is a strict
prefix-subset of the Pilot-150 sample (same seed, same frozen tasks).

Outputs: manifests/pilot60.json, outputs/preflight/PILOT60_AUDIT.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.registry.loader import by_global_id, load_pairs, load_role  # noqa: E402

OUT = PROJECT / "outputs" / "preflight"
MANIFESTS = PROJECT / "manifests"
OUT.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)

SEED = "20260915"
EXPERIENCE_COUNT = 30
ID_COUNT = 10
TRANSFER_COUNT = 10
RETENTION_COUNT = 10
TARGET_EXPERIENCE_PER_SOURCE = 6
SOURCES = ("TVRBench", "SpatialWorld", "AsgardBench", "EB-ALFRED", "EB-Habitat")
ID_QUOTA = {src: 2 for src in SOURCES}
TRANSFER_QUOTA = {"TVRBench": 3, "SpatialWorld": 3, "EB-ALFRED": 2, "EB-Habitat": 2}
RETENTION_QUOTA = {src: 2 for src in SOURCES}


def rank(name: str, task_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{name}|{task_id}".encode()).hexdigest()


def main() -> int:
    experience = by_global_id("experience")
    probes = {"id": by_global_id("id"), "transfer": by_global_id("transfer"),
              "retention": by_global_id("retention")}
    all_by_id = {**experience, **probes["id"], **probes["transfer"], **probes["retention"]}
    id_pairs = load_pairs("id")
    transfer_pairs = load_pairs("transfer")

    # ---- ID probes: fixed 2 per source (Pilot-60 protocol)
    id_by_source = defaultdict(list)
    for pair in id_pairs:
        probe = probes["id"].get(pair["probe_task_id"])
        if probe:
            id_by_source[probe["source_dataset"]].append(pair)
    chosen_id_pairs = []
    for source, n in ID_QUOTA.items():
        pairs = sorted(id_by_source[source], key=lambda p: rank("id", p["probe_task_id"]))
        chosen_id_pairs.extend(pairs[:n])

    # ---- Transfer probes: fixed quota per source
    transfer_by_source = defaultdict(list)
    for pair in transfer_pairs:
        transfer_by_source[pair["source_dataset"]].append(pair)
    chosen_transfer_pairs = []
    for source, n in TRANSFER_QUOTA.items():
        pairs = sorted(transfer_by_source[source], key=lambda p: rank("transfer", p["probe_task_id"]))
        chosen_transfer_pairs.extend(pairs[:n])

    # ---- Retention anchors: fixed 2 per source
    retention = load_role("retention")
    chosen_retention = []
    for source, n in RETENTION_QUOTA.items():
        rows = sorted([r for r in retention if r["source_dataset"] == source],
                      key=lambda r: rank("retention", r["global_task_id"]))
        chosen_retention.extend(rows[:n])

    # ---- Experience: mandatory pair anchors first, then deficit filling
    chosen_experience_ids: list[str] = []
    seen: set[str] = set()

    def add_experience(global_id: str) -> None:
        if global_id and global_id in experience and global_id not in seen:
            seen.add(global_id)
            chosen_experience_ids.append(global_id)

    for pair in chosen_id_pairs + chosen_transfer_pairs:
        add_experience(pair["experience_task_id"])

    pool = sorted(experience.values(), key=lambda r: rank("experience", r["global_task_id"]))
    have = Counter(experience[g]["source_dataset"] for g in chosen_experience_ids)
    target = {src: TARGET_EXPERIENCE_PER_SOURCE for src in SOURCES}
    exhausted: set[str] = set()
    while len(chosen_experience_ids) < EXPERIENCE_COUNT:
        deficits = {src: target[src] - have[src] for src in SOURCES if src not in exhausted}
        positive = [src for src, d in deficits.items() if d > 0]
        if not positive:
            break
        src = max(positive, key=lambda s: (deficits[s], s))
        added = False
        for rec in pool:
            if rec["source_dataset"] != src or rec["global_task_id"] in seen:
                continue
            add_experience(rec["global_task_id"])
            have[src] += 1
            added = True
            break
        if not added:
            exhausted.add(src)
    for rec in pool:                      # final top-up with any remaining experience
        if len(chosen_experience_ids) >= EXPERIENCE_COUNT:
            break
        add_experience(rec["global_task_id"])

    # ---- stream order (same for every method; prefix of Pilot-150 order)
    stream = sorted(chosen_experience_ids, key=lambda g: rank("experience_stream", g))

    manifest = {
        "pilot_version": "Pilot-60-v0.1",
        "seed": SEED,
        "quota": {"experience": EXPERIENCE_COUNT, "id": ID_COUNT,
                  "transfer": TRANSFER_COUNT, "retention": RETENTION_COUNT},
        "experience": chosen_experience_ids,
        "id_probe": [p["probe_task_id"] for p in chosen_id_pairs],
        "transfer_probe": [p["probe_task_id"] for p in chosen_transfer_pairs],
        "retention_probe": [r["global_task_id"] for r in chosen_retention],
        "id_pairs": chosen_id_pairs,
        "transfer_pairs": chosen_transfer_pairs,
        "experience_stream_order": stream,
    }

    # ---- checks
    role_ids = {
        "experience": set(manifest["experience"]),
        "id": set(manifest["id_probe"]),
        "transfer": set(manifest["transfer_probe"]),
        "retention": set(manifest["retention_probe"]),
    }
    physical = {role: {all_by_id[g]["physical_task_id"] for g in ids} for role, ids in role_ids.items()}
    overlap = {}
    names = list(role_ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a}x{b}"] = len(physical[a] & physical[b])

    audit = {
        "protocol": "Pilot-60-v0.1",
        "counts": {k: len(v) for k, v in role_ids.items()},
        "physical_overlap": overlap,
        "id_by_source": dict(Counter(all_by_id[g]["source_dataset"] for g in manifest["id_probe"])),
        "transfer_by_source": dict(Counter(all_by_id[g]["source_dataset"] for g in manifest["transfer_probe"])),
        "retention_by_source": dict(Counter(all_by_id[g]["source_dataset"] for g in manifest["retention_probe"])),
        "experience_by_source": dict(Counter(all_by_id[g]["source_dataset"] for g in manifest["experience"])),
        "status": "PASS",
    }
    expected = {"experience": EXPERIENCE_COUNT, "id": ID_COUNT,
                "transfer": TRANSFER_COUNT, "retention": RETENTION_COUNT}
    if audit["counts"] != expected:
        audit["status"] = "FAIL"
    if any(v != 0 for v in overlap.values()):
        audit["status"] = "FAIL"
    if len(role_ids["experience"] | role_ids["id"] | role_ids["transfer"] | role_ids["retention"]) != 60:
        audit["status"] = "FAIL"

    (MANIFESTS / "pilot60.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "PILOT60_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"\nG-PILOT-60 MANIFEST: {audit['status']}")
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
