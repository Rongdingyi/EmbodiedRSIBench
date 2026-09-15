#!/usr/bin/env python
"""08_run_pilot150.py -- canonical Pilot-150 runner (guide sections 26/27/41).

Usage:
    python scripts/08_run_pilot150.py --method none [--methods none,raw_memory]
    python scripts/08_run_pilot150.py --method none --max-experience 2 --max-probes 2

Per method: S000 probe -> 75 experience (snapshots S025/S050/S075) -> S075 probe.
Probes run on a snapshot clone with updates disabled and the state hash asserted
unchanged. All methods share the same experience stream order (manifest).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "external" / "OpenETA"))

from benchmark.adapters import make_adapter  # noqa: E402
from benchmark.openeta_bridge.context_injection import log_injection  # noqa: E402
from benchmark.openeta_bridge.episode_runner import run_episode  # noqa: E402
from benchmark.registry.loader import by_global_id  # noqa: E402
from benchmark.rsi.ace_context import AceContextRSI  # noqa: E402
from benchmark.rsi.embodiskill import EmbodiSkillRSI  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from benchmark.rsi.raw_memory import RawMemoryRSI  # noqa: E402
from benchmark.rsi.worldmind import WorldMindRSI  # noqa: E402

METHODS = {"none": NoneRSI, "raw_memory": RawMemoryRSI, "ace_context": AceContextRSI,
           "worldmind": WorldMindRSI, "embodiskill": EmbodiSkillRSI}
SEED = 20260915


def public_episode_dir(out: Path, index: int) -> Path:
    return out / f"episode_{index:04d}"


def run_probe_checkpoint(method, out_root: Path, checkpoint: str, roles: list[str],
                         manifest: dict, max_probes: int | None) -> dict:
    snapshot = out_root / "states" / checkpoint
    method.snapshot(snapshot)
    clone = method.clone_from_snapshot(snapshot)
    assert clone.update_enabled is False, "probe clone must be read-only"
    h0 = clone.state_hash()
    results = {"checkpoint": checkpoint, "hash_before": h0, "tasks": {}}
    for role, key in (("id", "id_probe"), ("transfer", "transfer_probe"),
                      ("retention", "retention_probe")):
        ids = manifest[key]
        if max_probes:
            ids = ids[:max_probes]
        for gid in ids:
            task = by_global_id(role)[gid]
            probe_dir = out_root / "probes" / checkpoint / role / gid
            probe_dir.mkdir(parents=True, exist_ok=True)
            adapter = make_adapter(task["source_dataset"])
            res = run_episode(task, adapter, clone, role=role, output_dir=probe_dir,
                              max_turns=8)
            results["tasks"][gid] = {
                "role": role, "success": res.outcome.get("success"),
                "turns": res.turns, "error": res.error,
                "wall_s": round(res.wall_time_s, 1),
            }
            (probe_dir / "public_trajectory.json").write_text(
                json.dumps(res.public_trajectory, ensure_ascii=False, indent=2) + "\n")
            (probe_dir / "private_eval.json").write_text(
                json.dumps(res.private_evaluation, ensure_ascii=False, indent=2) + "\n")
            (probe_dir / "outcome.json").write_text(
                json.dumps(res.outcome, ensure_ascii=False, indent=2) + "\n")
    h1 = clone.state_hash()
    results["hash_after"] = h1
    results["probe_state_unchanged"] = h0 == h1
    return results


def run_experience_stream(method, out_root: Path, manifest: dict, checkpoint_every: int,
                          max_experience: int | None) -> dict:
    stream = manifest["experience_stream_order"]
    if max_experience:
        stream = stream[:max_experience]
    updates = []
    for index, gid in enumerate(stream, 1):
        task = by_global_id("experience")[gid]
        ep_dir = public_episode_dir(out_root / "experience", index)
        ep_dir.mkdir(parents=True, exist_ok=True)
        before = method.state_hash()
        adapter = make_adapter(task["source_dataset"])
        res = run_episode(task, adapter, method, role="experience", output_dir=ep_dir,
                          max_turns=8)
        after = method.state_hash()
        (ep_dir / "public_trajectory.json").write_text(
            json.dumps(res.public_trajectory, ensure_ascii=False, indent=2) + "\n")
        (ep_dir / "private_eval.json").write_text(
            json.dumps(res.private_evaluation, ensure_ascii=False, indent=2) + "\n")
        (ep_dir / "outcome.json").write_text(
            json.dumps(res.outcome, ensure_ascii=False, indent=2) + "\n")
        updates.append({
            "update_id": f"{method.name}:{gid}",
            "method": method.name,
            "source_episode_id": gid,
            "state_hash_before": before,
            "state_hash_after": after,
            "changed": before != after,
            "success": res.outcome.get("success"),
            "turns": res.turns,
            "wall_s": round(res.wall_time_s, 1),
            "infrastructure_error": res.error,
        })
        with open(out_root / "rsi_updates.jsonl", "a") as f:
            f.write(json.dumps(updates[-1], ensure_ascii=False) + "\n")
        print(f"[{method.name}] experience {index}/{len(stream)} {gid} "
              f"success={res.outcome.get('success')} state_changed={before != after} "
              f"({res.wall_time_s:.0f}s)")
        if checkpoint_every and index % checkpoint_every == 0:
            method.snapshot(out_root / "states" / f"S{index:03d}")
    method.snapshot(out_root / "states" / f"S{len(stream):03d}")
    return {"updates": updates, "stream": stream}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--max-experience", type=int, default=None)
    parser.add_argument("--max-probes", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    manifest = json.loads((PROJECT / "manifests" / "pilot150.json").read_text())
    method = METHODS[args.method]()
    if getattr(method, "BLOCKED", False):
        print(f"[{args.method}] BLOCKED: {method.BLOCK_REASON}")
        return 2

    out_root = PROJECT / "outputs" / "pilot150" / args.method / f"seed_{SEED}"
    out_root.mkdir(parents=True, exist_ok=True)
    state_root = out_root / "rsi_state"
    method.init_run({"state_root": str(state_root), "run_id": f"pilot150:{args.method}",
                     "total_steps": len(manifest["experience"])})

    (out_root / "config_resolved.yaml").write_text(
        (PROJECT / "configs" / "agent" / "openeta_frozen.yaml").read_text())
    (out_root / "provenance.json").write_text(json.dumps({
        "benchmark_version": "Core-v1.0",
        "pilot_version": "Pilot-150-v0.1",
        "seed": SEED,
        "model": "deepseek-flash",
        "model_provider": "DeepSeek official API",
        "model_base_url": "https://api.deepseek.com",
        "rsi_method": args.method,
        "openeta_commit": "7d4a0a1522ba8ebbd362bde880bad81d2a98f15e",
    }, indent=2) + "\n")
    for name in ("external_sources.json", "openeta_freeze_manifest.json"):
        shutil.copy(PROJECT / "manifests" / name, out_root / name)

    t0 = time.time()
    # S000 probe
    s000 = run_probe_checkpoint(method, out_root, "S000", ["id", "transfer", "retention"],
                                manifest, args.max_probes)
    print(f"[{args.method}] S000 probe done (unchanged={s000['probe_state_unchanged']})")
    method.snapshot(out_root / "states" / "S000")

    # experience stream with snapshots
    stream_info = run_experience_stream(method, out_root, manifest,
                                        args.checkpoint_every, args.max_experience)
    final_checkpoint = f"S{len(stream_info['stream']):03d}"

    # S075 probe
    s_final = run_probe_checkpoint(method, out_root, final_checkpoint,
                                   ["id", "transfer", "retention"], manifest, args.max_probes)
    print(f"[{args.method}] {final_checkpoint} probe done "
          f"(unchanged={s_final['probe_state_unchanged']})")

    metrics = {
        "method": args.method,
        "experience": len(stream_info["updates"]),
        "state_updates": sum(1 for u in stream_info["updates"] if u["changed"]),
        "probe_state_unchanged": s000["probe_state_unchanged"] and s_final["probe_state_unchanged"],
        "wall_s": round(time.time() - t0, 1),
    }
    (out_root / "probes" / "S000" / "summary.json").write_text(json.dumps(s000, indent=2) + "\n")
    (out_root / "probes" / final_checkpoint / "summary.json").write_text(
        json.dumps(s_final, indent=2) + "\n")
    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_root / "FINAL_STATUS.txt").write_text(
        "PASS\n" if metrics["probe_state_unchanged"] else "FAIL\n")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["probe_state_unchanged"] else 1


if __name__ == "__main__":
    sys.exit(main())
