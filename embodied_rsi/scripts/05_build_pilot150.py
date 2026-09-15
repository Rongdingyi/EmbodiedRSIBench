#!/usr/bin/env python
"""05_build_pilot150.py -- Pilot-150 manifest (guide sections 23/24).

Composition: Experience 75, ID 25, Transfer 30, Retention 20.
Selection: pick pairs first (ID/Transfer), then their Experience anchors, then
retention anchors (+ a same-source/skill experience anchor each), then a
deterministic fill to 75 Experience with coverage across all five sources.

Outputs: manifests/pilot150.json, outputs/preflight/PILOT150_AUDIT.json
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
ID_COUNT, TRANSFER_COUNT, RETENTION_COUNT, EXPERIENCE_COUNT = 25, 30, 20, 75
TARGET_EXPERIENCE_PER_SOURCE = 15
TRANSFER_QUOTA = {"TVRBench": 10, "SpatialWorld": 10, "EB-ALFRED": 5, "EB-Habitat": 5}


def rank(name: str, task_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{name}|{task_id}".encode()).hexdigest()


def largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    s = sum(weights.values())
    exact = {k: total * v / s for k, v in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k, _ in sorted(exact.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0])):
        if sum(out.values()) >= total:
            break
        out[k] += 1
    return out


def main() -> int:
    experience = by_global_id("experience")
    probes = {"id": by_global_id("id"), "transfer": by_global_id("transfer"),
              "retention": by_global_id("retention")}
    all_by_id = {**experience, **probes["id"], **probes["transfer"], **probes["retention"]}
    id_pairs = load_pairs("id")
    transfer_pairs = load_pairs("transfer")

    # ---- ID probes: proportional quota over the Core ID-pair source mix
    id_by_source = defaultdict(list)
    for pair in id_pairs:
        probe = probes["id"].get(_pid_of(pair["probe_task_id"], probes["id"]))
        if probe:
            id_by_source[probe["source_dataset"]].append(pair)
    id_weights = {k: len(v) for k, v in id_by_source.items()}
    id_quota = largest_remainder(id_weights, ID_COUNT)
    chosen_id_pairs = []
    for source, n in id_quota.items():
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

    # ---- Retention anchors: proportional to the Core retention mix
    retention = load_role("retention")
    ret_by_source = Counter(r["source_dataset"] for r in retention)
    ret_quota = largest_remainder(dict(ret_by_source), RETENTION_COUNT)
    chosen_retention = []
    for source, n in ret_quota.items():
        rows = sorted([r for r in retention if r["source_dataset"] == source],
                      key=lambda r: rank("retention", r["global_task_id"]))
        chosen_retention.extend(rows[:n])

    # ---- Experience: mandatory pair anchors first, then deficit filling
    # (review amendment: retention anchors are NOT paired with a synthetic
    # experience anchor; they are fixed held-out tasks re-measured per checkpoint)
    chosen_experience_ids: list[str] = []
    seen = set()

    def add_experience(global_id: str) -> None:
        if global_id and global_id not in seen:
            seen.add(global_id)
            chosen_experience_ids.append(global_id)

    for pair in chosen_id_pairs + chosen_transfer_pairs:
        add_experience(pair["experience_task_id"])

    # deficit filling: repeatedly add to the most-deficient source until the cap
    pool = sorted(experience.values(), key=lambda r: rank("experience", r["global_task_id"]))
    have = Counter(experience[g]["source_dataset"] for g in chosen_experience_ids
                   if g in experience)
    sources = sorted({r["source_dataset"] for r in experience.values()})
    target = {src: TARGET_EXPERIENCE_PER_SOURCE for src in sources}
    exhausted: set[str] = set()
    while len(chosen_experience_ids) < EXPERIENCE_COUNT:
        deficits = {src: target[src] - have[src] for src in sources if src not in exhausted}
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

    # ---- stream order (same for every method)
    stream = sorted(chosen_experience_ids, key=lambda g: rank("experience_stream", g))

    manifest = {
        "pilot_version": "Pilot-150-v0.1",
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

    (MANIFESTS / "pilot150.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "PILOT150_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"\nG-PILOT-150 MANIFEST: {audit['status']}")
    return 0 if audit["status"] == "PASS" else 1


def _pid_of(global_id: str, probe_pool: dict) -> str:
    return global_id if global_id in probe_pool else global_id


if __name__ == "__main__":
    sys.exit(main())
