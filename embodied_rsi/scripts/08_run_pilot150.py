#!/usr/bin/env python
"""08_run_pilot150.py -- canonical Pilot-150 runner (P0/P1 amended).

Uses the upstream `OpenEtaEpisodeRunner` through `run_episode`, enforces the
OpenETA-turn / tool-call / environment-step budgets separately, advances the RSI
experience-stream counters (ACE), and writes the guide's per-condition layout
including token accounting, leakage audit and three-state FINAL_STATUS.

Usage:
    python scripts/08_run_pilot150.py --method none
    python scripts/08_run_pilot150.py --method none --max-experience 2 --max-probes 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "external" / "OpenETA"))

from benchmark.adapters import make_adapter  # noqa: E402
from benchmark.openeta_bridge.episode_runner import run_episode  # noqa: E402
from benchmark.registry.loader import by_global_id  # noqa: E402
from benchmark.rsi.ace_context import AceContextRSI  # noqa: E402
from benchmark.rsi.embodiskill import EmbodiSkillRSI  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from benchmark.rsi.raw_memory import RawMemoryRSI  # noqa: E402
from benchmark.rsi.worldmind import WorldMindRSI  # noqa: E402
from benchmark.runner import gates  # noqa: E402

METHODS = {"none": NoneRSI, "raw_memory": RawMemoryRSI, "ace_context": AceContextRSI,
           "worldmind": WorldMindRSI, "embodiskill": EmbodiSkillRSI}
SEED = 20260915


def load_protocol() -> dict:
    return yaml.safe_load((PROJECT / "configs" / "pilot150.yaml").read_text())


def episode_config(protocol: dict, source: str) -> dict:
    budgets = protocol["budgets"]
    return {
        "max_turns": budgets["openeta_turns"],
        "max_tool_calls": budgets["openeta_tool_calls"],
        "timeout_s": budgets["episode_timeout_s"],
        "max_total_tokens": budgets["max_total_tokens"],
        "planner_max_output_tokens": budgets["planner_max_output_tokens"],
        "max_env_steps": budgets["max_env_steps"].get(source),
        "max_rsi_injection_tokens": budgets["max_rsi_injection_tokens"],
    }


def run_probe_checkpoint(method, out_root: Path, checkpoint: str, manifest: dict,
                         protocol: dict, max_probes: int | None) -> dict:
    snapshot = out_root / "states" / checkpoint
    method.snapshot(snapshot)
    clone = method.clone_from_snapshot(snapshot)     # new instance + temp copy (P0-4)
    assert clone.update_enabled is False, "probe clone must be read-only"
    h0 = clone.state_hash()
    results = {"checkpoint": checkpoint, "hash_before": h0, "tasks": {},
               "infra_errors": 0, "leakage": 0}
    for role, key in (("id", "id_probe"), ("transfer", "transfer_probe"),
                      ("retention", "retention_probe")):
        ids = manifest[key]
        if max_probes:
            ids = ids[:max_probes]
        for gid in ids:
            task = by_global_id(role)[gid]
            probe_dir = out_root / "probes" / checkpoint / role / gid
            adapter = make_adapter(task["source_dataset"])
            res = run_episode(task, adapter, clone, role=role, output_dir=probe_dir,
                              config=episode_config(protocol, task["source_dataset"]))
            results["tasks"][gid] = {
                "role": role, "success": res.outcome.get("success"),
                "turns": res.turns, "env_steps": res.env_steps,
                "status": res.status, "error": res.error,
                "leakage": res.leakage_violations,
                "wall_s": round(res.wall_time_s, 1),
            }
            if res.error:
                results["infra_errors"] += 1
            results["leakage"] += len(res.leakage_violations)
    h1 = clone.state_hash()
    results["hash_after"] = h1
    results["probe_state_unchanged"] = h0 == h1
    results["status"] = gates.PASS if (results["probe_state_unchanged"]
                                       and results["infra_errors"] == 0
                                       and results["leakage"] == 0) else gates.FAIL
    return results


def run_experience_stream(method, out_root: Path, manifest: dict, protocol: dict,
                          checkpoint_every: int, max_experience: int | None) -> dict:
    stream = manifest["experience_stream_order"]
    if max_experience:
        stream = stream[:max_experience]
    updates = []
    infra_errors = 0
    leakage = 0
    aggregate = {}
    for index, gid in enumerate(stream, 1):
        task = by_global_id("experience")[gid]
        ep_dir = out_root / "experience" / f"episode_{index:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        method.advance_step(index, len(stream))       # ACE curator counters (P1-7)
        before = method.state_hash()
        adapter = make_adapter(task["source_dataset"])
        res = run_episode(task, adapter, method, role="experience", output_dir=ep_dir,
                          config=episode_config(protocol, task["source_dataset"]))
        after = method.state_hash()
        acc = res.accounting
        for key in ("planner_input_tokens", "planner_output_tokens", "planner_calls",
                    "rsi_update_input_tokens", "rsi_update_output_tokens",
                    "worldmind_prediction_sidecar_tokens", "simulator_steps"):
            aggregate[key] = aggregate.get(key, 0) + int(acc.get(key) or 0)
        aggregate["rsi_injection_tokens"] = (
            aggregate.get("rsi_injection_tokens", 0) + int(acc.get("rsi_injection_tokens") or 0))
        if res.error:
            infra_errors += 1
        leakage += len(res.leakage_violations)
        updates.append({
            "update_id": f"{method.name}:{gid}",
            "method": method.name,
            "source_episode_id": gid,
            "source_dataset": task["source_dataset"],
            "state_hash_before": before,
            "state_hash_after": after,
            "changed": before != after,
            "success": res.outcome.get("success"),
            "turns": res.turns,
            "env_steps": res.env_steps,
            "status": res.status,
            "infrastructure_error": res.error,
            "leakage": res.leakage_violations,
            "accounting": acc,
            "wall_s": round(res.wall_time_s, 1),
        })
        with open(out_root / "rsi_updates.jsonl", "a") as f:
            f.write(json.dumps(updates[-1], ensure_ascii=False) + "\n")
        print(f"[{method.name}] experience {index}/{len(stream)} {gid} "
              f"success={res.outcome.get('success')} env_steps={res.env_steps} "
              f"state_changed={before != after} ({res.wall_time_s:.0f}s)")
        if checkpoint_every and index % checkpoint_every == 0:
            method.snapshot(out_root / "states" / f"S{index:03d}")
    method.snapshot(out_root / "states" / f"S{len(stream):03d}")
    return {"updates": updates, "stream": stream, "infra_errors": infra_errors,
            "leakage": leakage, "accounting": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--max-experience", type=int, default=None)
    parser.add_argument("--max-probes", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    protocol = load_protocol()
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
        "episode_orchestration": "upstream OpenEtaEpisodeRunner",
    }, indent=2) + "\n")
    for name in ("external_sources.json", "openeta_freeze_manifest.json"):
        shutil.copy(PROJECT / "manifests" / name, out_root / name)

    t0 = time.time()
    s000 = run_probe_checkpoint(method, out_root, "S000", manifest, protocol, args.max_probes)
    print(f"[{args.method}] S000 probe {s000['status']} "
          f"(unchanged={s000['probe_state_unchanged']} infra={s000['infra_errors']})")
    method.snapshot(out_root / "states" / "S000")

    stream_info = run_experience_stream(method, out_root, manifest, protocol,
                                        args.checkpoint_every, args.max_experience)
    final_checkpoint = f"S{len(stream_info['stream']):03d}"
    s_final = run_probe_checkpoint(method, out_root, final_checkpoint, manifest, protocol,
                                   args.max_probes)
    print(f"[{args.method}] {final_checkpoint} probe {s_final['status']} "
          f"(unchanged={s_final['probe_state_unchanged']} infra={s_final['infra_errors']})")

    status = gates.FAIL
    if (s000["status"] == gates.PASS and s_final["status"] == gates.PASS
            and stream_info["infra_errors"] == 0 and stream_info["leakage"] == 0):
        status = gates.PASS
    metrics = {
        "method": args.method,
        "status": status,
        "experience": len(stream_info["updates"]),
        "state_updates": sum(1 for u in stream_info["updates"] if u["changed"]),
        "probe_state_unchanged": s000["probe_state_unchanged"] and s_final["probe_state_unchanged"],
        "infra_errors": stream_info["infra_errors"] + s000["infra_errors"] + s_final["infra_errors"],
        "leakage_violations": stream_info["leakage"] + s000["leakage"] + s_final["leakage"],
        "accounting": stream_info["accounting"],
        "updater_tokens": (stream_info["accounting"].get("rsi_update_input_tokens", 0)
                           + stream_info["accounting"].get("rsi_update_output_tokens", 0)
                           + stream_info["accounting"].get("worldmind_prediction_sidecar_tokens", 0)),
        "env_steps": stream_info["accounting"].get("simulator_steps", 0),
        "wall_s": round(time.time() - t0, 1),
    }
    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_root / "FINAL_STATUS.txt").write_text(status + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "accounting"}, indent=2))
    return 0 if status == gates.PASS else 1


if __name__ == "__main__":
    sys.exit(main())
