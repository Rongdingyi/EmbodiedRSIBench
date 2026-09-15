"""Release registry loader: joins role parquet + JSONL private metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("EMBODIED_RSI_DATA", PROJECT.parent / "data")).resolve()

_ROLE_FILES = {
    "experience": ("experience.parquet", "experience.jsonl"),
    "id": ("id_probe.parquet", "id_probe.jsonl"),
    "transfer": ("transfer_probe.parquet", "transfer_probe.jsonl"),
    "retention": ("retention_probe.parquet", "retention_probe.jsonl"),
}
_CACHE: dict[str, list[dict]] = {}


def load_role(role: str) -> list[dict]:
    """Return task records for one pilot role, deterministic order (file order)."""
    if role in _CACHE:
        return _CACHE[role]
    parquet_name, jsonl_name = _ROLE_FILES[role]
    rows = pq.read_table(DATA / "core_v1" / parquet_name).to_pylist()
    private = {}
    with open(DATA / "core_v1" / jsonl_name) as f:
        for line in f:
            rec = json.loads(line)
            private[rec["task_id"]] = rec.get("private_eval_metadata") or {}

    records = []
    for row in rows:
        raw_meta = {}
        try:
            raw_meta = json.loads(row.get("raw_metadata_json") or "{}")
        except Exception:
            pass
        rec = {
            "global_task_id": row["global_raw_id"],
            "physical_task_id": row["physical_task_id"],
            "role": role,
            "source_dataset": row["source_dataset"],
            "source_backend": row["source_backend"],
            "source_split": row["source_split"],
            "source_task_id": row["source_task_id"],
            "source_entry_index": row["source_entry_index"],
            "source_path": row["source_path"],
            "instruction": row["instruction"],
            "scene_id_canonical": row["scene_id_canonical"],
            "scene_id_raw": row["scene_id_raw"],
            "native_task_type": row["native_task_type"],
            "difficulty_canonical": row["difficulty_canonical"],
            "skill_family_primary": row["skill_family_primary"],
            "skill_family_set": row["skill_family_set"],
            "primitive_set": row["primitive_set"],
            # private-side fields (worker / evaluator only)
            "private_eval_metadata": private.get(row["global_raw_id"], {}),
            "raw_metadata": raw_meta,
        }
        records.append(rec)
    _CACHE[role] = records
    return records


def load_pairs(kind: str) -> list[dict]:
    name = {"id": "id_pairs.parquet", "transfer": "transfer_pairs.parquet"}[kind]
    return pq.read_table(DATA / "core_v1" / name).to_pylist()


def retention_schedule() -> dict:
    return json.loads((DATA / "core_v1" / "retention_schedule.json").read_text())


def by_global_id(role: str) -> dict[str, dict]:
    return {r["global_task_id"]: r for r in load_role(role)}
