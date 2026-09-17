#!/usr/bin/env python
"""08b_run_pilot60_canonical.py -- Pilot-60 under the canonical ReAct agent.

One planner decision = one source environment action. Same manifest, same task
order, same probes as the historical OpenETA Pilot-60; only the RSI method
varies. Outputs go to outputs/pilot60_canonical/<method>/seed_<seed>/.

Usage:
    python scripts/08b_run_pilot60_canonical.py --method none
    python scripts/08b_run_pilot60_canonical.py --method none --max-experience 2 --max-probes 2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.adapters import make_adapter  # noqa: E402
from benchmark.agent.config import agent_config_hash, load_protocol_config  # noqa: E402
from benchmark.agent.episode_runner import run_episode  # noqa: E402
from benchmark.agent.freeze import build_freeze, freeze_to_json, sha256_file  # noqa: E402
from benchmark.registry.loader import by_global_id  # noqa: E402
from benchmark.rsi.ace_context import AceContextRSI  # noqa: E402
from benchmark.rsi.embodiskill import EmbodiSkillRSI  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from benchmark.rsi.raw_memory import RawMemoryRSI  # noqa: E402
from benchmark.rsi.worldmind import WorldMindRSI  # noqa: E402
from benchmark.runner import gates  # noqa: E402
from benchmark.utils import normalize_base_url  # noqa: E402

METHODS = {"none": NoneRSI, "raw_memory": RawMemoryRSI, "ace_context": AceContextRSI,
           "worldmind": WorldMindRSI, "embodiskill": EmbodiSkillRSI}


def prepare_run_root(out_root: Path, *, delete_existing: bool) -> None:
    """Fail-closed run root: non-empty roots abort unless deletion is explicit.

    A canonical run must start from EMPTY RSI state. `--delete-existing-run`
    really removes the previous root (state, logs, metrics) instead of merely
    allowing writes into it, so a re-run can never read or append old state.
    """
    out_root = Path(out_root)
    if out_root.exists():
        if delete_existing:
            shutil.rmtree(out_root)
        elif any(out_root.iterdir()):
            raise SystemExit(
                f"[FAIL] canonical run output already exists: {out_root}\n"
                "        canonical runs must start from empty RSI state.\n"
                "        archive it, pass --delete-existing-run, or use --output-root.")
    out_root.mkdir(parents=True, exist_ok=True)


def repo_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_probe_checkpoint(method, out_root: Path, checkpoint: str, manifest: dict,
                         protocol, max_probes: int | None) -> dict:
    snapshot = out_root / "states" / checkpoint
    method.snapshot(snapshot)
    clone = method.clone_from_snapshot(snapshot)
    assert clone.update_enabled is False, "probe clone must be read-only"
    h0 = clone.state_hash()
    results = {"checkpoint": checkpoint, "hash_before": h0, "tasks": {},
               "infra_errors": 0, "leakage": 0}
    for role, key in (("id", "id_probe"), ("transfer", "transfer_probe"),
                      ("retention", "retention_probe")):
        ids = manifest[key]
        if max_probes:
            ids = ids[:max_probes]
        for index, gid in enumerate(ids, 1):
            task = by_global_id(role)[gid]
            task_dir = f"task_{index:03d}"
            probe_dir = out_root / "probes" / checkpoint / role / task_dir
            attempts = 0
            res = None
            for attempt in range(3):
                attempts = attempt + 1
                adapter = make_adapter(task["source_dataset"])
                res = run_episode(task, adapter, clone, role=role, config=protocol,
                                  output_dir=probe_dir)
                if not res.error:
                    break
                print(f"[probe retry {attempts}/3] {checkpoint} {role}/{task_dir} "
                      f"{str(res.error)[:90]}")
            results["tasks"][gid] = {
                "role": role, "dir": f"{role}/{task_dir}", "attempts": attempts,
                "success": res.outcome.get("success"),
                "turns": res.env_steps, "env_steps": res.env_steps,
                "status": res.status, "error": res.error,
                "stop_reason": res.stop_reason,
                "planner_calls": res.planner_calls,
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
    (out_root / "probes" / checkpoint).mkdir(parents=True, exist_ok=True)
    (out_root / "probes" / checkpoint / "summary.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n")
    return results


def run_experience_stream(method, out_root: Path, manifest: dict, protocol,
                          max_experience: int | None) -> dict:
    stream = manifest["experience_stream_order"]
    if max_experience:
        stream = stream[:max_experience]
    updates = []
    infra_errors = 0
    experience_failures = 0
    planner_protocol_failures = 0
    leakage = 0
    aggregate: dict = {}
    for index, gid in enumerate(stream, 1):
        task = by_global_id("experience")[gid]
        ep_dir = out_root / "experience" / f"episode_{index:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        method.advance_step(index, len(stream))
        before = method.state_hash()
        attempts = 0
        res = None
        while attempts < 3:
            attempts += 1
            adapter = make_adapter(task["source_dataset"])
            res = run_episode(task, adapter, method, role="experience",
                              config=protocol, output_dir=ep_dir)
            if not res.error:
                break
            if res.env_steps > 0 or method.state_hash() != before:
                break
            print(f"[experience retry {attempts}/3] {gid} {str(res.error)[:90]}")
        after = method.state_hash()
        acc = res.accounting
        for key in ("planner_input_tokens", "planner_output_tokens", "planner_calls",
                    "planner_validation_retries", "planner_invalid_json_count",
                    "planner_invalid_action_count",
                    "rsi_update_input_tokens", "rsi_update_output_tokens",
                    "worldmind_prediction_sidecar_tokens", "simulator_steps"):
            aggregate[key] = aggregate.get(key, 0) + int(acc.get(key) or 0)
        aggregate["rsi_injection_tokens"] = (
            aggregate.get("rsi_injection_tokens", 0) + int(acc.get("rsi_injection_tokens") or 0))
        if res.error:
            infra_errors += 1
        if res.stop_reason == "planner_protocol_failure":
            planner_protocol_failures += 1
        if res.status != gates.PASS:
            experience_failures += 1
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
            "env_steps": res.env_steps,
            "planner_calls": res.planner_calls,
            "stop_reason": res.stop_reason,
            "status": res.status,
            "infrastructure_error": res.error,
            "rsi_update_error": res.rsi_update_error,
            "attempts": attempts,
            "leakage": res.leakage_violations,
            "accounting": acc,
            "wall_s": round(res.wall_time_s, 1),
        })
        with open(out_root / "rsi_updates.jsonl", "a") as f:
            f.write(json.dumps(updates[-1], ensure_ascii=False, default=str) + "\n")
        print(f"[{method.name}] experience {index}/{len(stream)} {gid} "
              f"success={res.outcome.get('success')} env_steps={res.env_steps} "
              f"planner_calls={res.planner_calls} state_changed={before != after} "
              f"({res.wall_time_s:.0f}s)")
    method.snapshot(out_root / "states" / f"S{len(stream):03d}")
    return {"updates": updates, "stream": stream, "infra_errors": infra_errors,
            "experience_failures": experience_failures,
            "planner_protocol_failures": planner_protocol_failures,
            "leakage": leakage, "accounting": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--max-experience", type=int, default=None)
    parser.add_argument("--max-probes", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="override the run directory (dev smokes use a separate root)")
    parser.add_argument("--delete-existing-run", action="store_true",
                        help="delete a previous run root before starting (destructive)")
    parser.add_argument("--overwrite", action="store_true",
                        help="deprecated alias of --delete-existing-run")
    args = parser.parse_args()

    protocol = load_protocol_config(PROJECT / "configs" / "pilot60_canonical.yaml")
    manifest = json.loads((PROJECT / "manifests" / "pilot60.json").read_text())
    method = METHODS[args.method]()
    if getattr(method, "BLOCKED", False):
        print(f"[{args.method}] BLOCKED: {method.BLOCK_REASON}")
        return 2

    out_root = (args.output_root if args.output_root is not None
                else PROJECT / "outputs" / "pilot60_canonical" / args.method
                / f"seed_{protocol.seed}")
    if args.overwrite and not args.delete_existing_run:
        print("warning: --overwrite is deprecated; use --delete-existing-run")
    prepare_run_root(out_root, delete_existing=args.delete_existing_run or args.overwrite)
    method.init_run({"state_root": str(out_root / "rsi_state"),
                     "run_id": f"pilot60_canonical:{args.method}",
                     "total_steps": len(manifest["experience"])})

    from dataclasses import asdict

    freeze = build_freeze(PROJECT)
    (out_root / "canonical_agent_freeze.json").write_text(freeze_to_json(freeze))
    (out_root / "agent_config_source.yaml").write_text(
        (PROJECT / "configs" / "agent" / "canonical_react.yaml").read_text())
    protocol_dict = asdict(protocol)
    protocol_dict.pop("agent", None)   # agent config is resolved separately below
    resolved_config = {
        "protocol_source": str(PROJECT / "configs" / "pilot60_canonical.yaml"),
        "agent_source": str(PROJECT / "configs" / "agent" / "canonical_react.yaml"),
        "protocol": protocol_dict,
        "agent": protocol.agent.to_dict(),
        "cli": {"method": args.method, "max_experience": args.max_experience,
                "max_probes": args.max_probes, "output_root": str(out_root),
                "delete_existing_run": bool(args.delete_existing_run or args.overwrite)},
        "resolved": {
            "seed": protocol.seed,
            "probe_checkpoints": list(protocol.probe_checkpoints),
            "max_agent_actions_per_source": protocol.max_agent_actions,
            "episode_timeout_s": protocol.episode_timeout_s,
            "planner_max_output_tokens": protocol.agent.planner.max_output_tokens,
            "planner_validation_retries": protocol.agent.planner.max_validation_retries,
            "max_rsi_injection_tokens": protocol.agent.rsi.max_injection_tokens,
            "temperature": protocol.agent.planner.temperature,
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
            "model_base_url": normalize_base_url(os.environ.get("DEEPSEEK_BASE_URL", "")),
        },
        "hashes": {"agent_config_sha256": agent_config_hash(protocol.agent),
                   "manifest_sha256": sha256_file(PROJECT / "manifests" / "pilot60.json")},
    }
    (out_root / "config_resolved.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
    external = json.loads((PROJECT / "manifests" / "external_sources.json").read_text())
    rsi_upstream = external.get({"none": "OpenETA", "raw_memory": "OpenETA",
                                 "ace_context": "ACE", "worldmind": "WorldMind",
                                 "embodiskill": "EmbodiSkill"}[args.method], {})
    provenance = {
        "benchmark_repo_commit": repo_commit(),
        "benchmark_version": "Core-v1.0",
        "protocol": "Pilot-60 Canonical ReAct v0.8",
        "pilot_manifest_sha256": sha256_file(PROJECT / "manifests" / "pilot60.json"),
        "seed": protocol.seed,
        "canonical_agent_name": protocol.agent.agent_name,
        "canonical_agent_config_sha256": agent_config_hash(protocol.agent),
        "canonical_prompt_sha256": freeze["system_prompt_sha256"],
        "agent_reference_repo": "EmbodiedBench/EmbodiedBench",
        "agent_reference_commit": protocol.agent.reference_commit,
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        "model_base_url": normalize_base_url(os.environ.get("DEEPSEEK_BASE_URL", "")),
        "temperature": protocol.agent.planner.temperature,
        "planner_max_output_tokens": protocol.agent.planner.max_output_tokens,
        "rsi_method": args.method,
        "rsi_upstream_repo": rsi_upstream.get("repo", ""),
        "rsi_upstream_commit": rsi_upstream.get("commit", ""),
        "one_action_per_turn": True,
        "cross_episode_agent_memory": False,
        "probe_update_enabled": False,
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for name in ("external_sources.json", "openeta_freeze_manifest.json",
                 "canonical_agent_reference.json"):
        source = PROJECT / "manifests" / name
        if source.exists():
            shutil.copy(source, out_root / name)

    t0 = time.time()
    s000 = run_probe_checkpoint(method, out_root, "S000", manifest, protocol, args.max_probes)
    print(f"[{args.method}] S000 probe {s000['status']} "
          f"(unchanged={s000['probe_state_unchanged']} infra={s000['infra_errors']})")
    method.snapshot(out_root / "states" / "S000")

    stream_info = run_experience_stream(method, out_root, manifest, protocol, args.max_experience)
    final_checkpoint = f"S{len(stream_info['stream']):03d}"
    s_final = run_probe_checkpoint(method, out_root, final_checkpoint, manifest, protocol,
                                   args.max_probes)
    print(f"[{args.method}] {final_checkpoint} probe {s_final['status']} "
          f"(unchanged={s_final['probe_state_unchanged']} infra={s_final['infra_errors']})")

    full_run = (args.max_experience is None and args.max_probes is None)
    checks = {
        "experience count == manifest": len(stream_info["updates"]) == len(manifest["experience"]),
        "S000 probe PASS": s000["status"] == gates.PASS,
        f"{final_checkpoint} probe PASS": s_final["status"] == gates.PASS,
        "probe task counts == manifest": all(
            sum(1 for t in summary["tasks"].values() if t["role"] == role) == len(manifest[key])
            for summary in (s000, s_final)
            for role, key in (("id", "id_probe"), ("transfer", "transfer_probe"),
                              ("retention", "retention_probe"))),
        "infrastructure crash rate < 2%": (stream_info["infra_errors"]
                                           / max(len(stream_info["stream"]), 1)) < 0.02,
        "no experience failure": stream_info["experience_failures"] == 0,
        "no planner protocol failure": stream_info["planner_protocol_failures"] == 0,
        "no leakage": stream_info["leakage"] == 0,
        f"final checkpoint is {final_checkpoint}": final_checkpoint == f"S{len(manifest['experience']):03d}",
    }
    status = gates.PASS if all(checks.values()) else gates.FAIL
    metrics = {
        "method": args.method,
        "status": status,
        "full_run": full_run,
        "checks": checks,
        "experience": len(stream_info["updates"]),
        "state_updates": sum(1 for u in stream_info["updates"] if u["changed"]),
        "probe_state_unchanged": (s000["probe_state_unchanged"]
                                  and s_final["probe_state_unchanged"]),
        "infra_errors": stream_info["infra_errors"] + s000["infra_errors"] + s_final["infra_errors"],
        "experience_failures": stream_info["experience_failures"],
        "planner_protocol_failures": stream_info["planner_protocol_failures"],
        "leakage_violations": stream_info["leakage"] + s000["leakage"] + s_final["leakage"],
        "accounting": stream_info["accounting"],
        "planner_calls": stream_info["accounting"].get("planner_calls", 0),
        "planner_input_tokens": stream_info["accounting"].get("planner_input_tokens", 0),
        "planner_output_tokens": stream_info["accounting"].get("planner_output_tokens", 0),
        "planner_validation_retries": stream_info["accounting"].get("planner_validation_retries", 0),
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
