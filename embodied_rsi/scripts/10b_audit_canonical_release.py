#!/usr/bin/env python
"""10b_audit_canonical_release.py -- canonical release audit (taskbook v0.8, Phase Y).

Hard checks over the canonical track only. The historical OPENETA_FREEZE gate is
deliberately NOT required here (historical track).

Writes outputs/FINAL_STATUS_CANONICAL.txt and, when present, the same status
into outputs/pilot60_canonical/FINAL_STATUS.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.runner import gates  # noqa: E402

OUT = PROJECT / "outputs" / "preflight"
PILOT = PROJECT / "outputs" / "pilot60_canonical"
CALIB = PROJECT / "outputs" / "canonical_calibration" / "CALIBRATION_REPORT.json"

FAILS: list[str] = []


def load(path: Path) -> dict:
    if not path.exists():
        FAILS.append(f"missing artifact: {path}")
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        FAILS.append(f"unreadable artifact {path}: {exc}")
        return {}


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + ("" if cond else f"  ({detail})"))
    if not cond:
        FAILS.append(f"{name}: {detail}")


def summarize_infra(pilot_root: Path) -> dict:
    """Infra crash rates with honest denominators.

    `rate_experience_episodes`: infra errors / executed experience episodes.
    `rate_all_attempts`: infra errors / every executed attempt, counting probe
    retries (a retried probe is an executed attempt, not an episode).
    """
    experience_infra = 0
    experience_episodes = 0
    experience_attempts = 0
    probe_infra = 0
    probe_tasks = 0
    probe_attempts = 0
    root = Path(pilot_root)
    if root.exists():
        for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for seed_dir in method_dir.glob("seed_*"):
                updates = seed_dir / "rsi_updates.jsonl"
                if updates.exists():
                    for line in updates.read_text().splitlines():
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        experience_episodes += 1
                        experience_attempts += int(entry.get("attempts") or 1)
                        if entry.get("infrastructure_error"):
                            experience_infra += 1
                probes = seed_dir / "probes"
                if probes.exists():
                    for ckpt in sorted(probes.glob("S*")):
                        summary = ckpt / "summary.json"
                        if not summary.exists():
                            continue
                        data = json.loads(summary.read_text())
                        for task in (data.get("tasks") or {}).values():
                            probe_tasks += 1
                            probe_attempts += int(task.get("attempts") or 1)
                            if task.get("error"):
                                probe_infra += 1

    def rate(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    return {
        "experience_infra_errors": experience_infra,
        "experience_episodes": experience_episodes,
        "experience_attempts": experience_attempts,
        "probe_infra_errors": probe_infra,
        "probe_tasks": probe_tasks,
        "probe_attempts": probe_attempts,
        "rate_experience_episodes": rate(experience_infra, experience_episodes),
        "rate_all_attempts": rate(experience_infra + probe_infra,
                                  experience_attempts + probe_attempts),
    }


def main() -> int:
    freeze = load(OUT / "CANONICAL_AGENT_FREEZE.json")
    check("canonical agent freeze PASS", freeze.get("status") == "PASS",
          str(freeze.get("status")))

    adapters = load(OUT / "ADAPTER_VALIDATION.json")
    check("adapter gate PASS", adapters.get("status") == "PASS", str(adapters.get("status")))

    calibration = load(CALIB) if CALIB.exists() else {}
    if CALIB.exists():
        cal_checks = calibration.get("checks") or {}
        check("calibration engineering invariants PASS",
              all(cal_checks.values()) if cal_checks else False, str(cal_checks))
    else:
        check("calibration engineering invariants PASS", False, f"missing {CALIB}")

    smoke = load(OUT / "RSI_SMOKE_CANONICAL.json")
    smoke_status = smoke.get("status")
    check("canonical RSI smoke PASS or protocol-legal BLOCKED",
          smoke_status in {gates.PASS, gates.BLOCKED}, str(smoke_status))

    manifest = load(PROJECT / "manifests" / "pilot60.json")
    expected_counts = {
        "experience": len(manifest.get("experience") or []),
        "id": len(manifest.get("id_probe") or []),
        "transfer": len(manifest.get("transfer_probe") or []),
        "retention": len(manifest.get("retention_probe") or []),
    }
    method_status: dict[str, str] = {}
    probe_mutations = 0
    snapshot_failures = 0
    infra_crashes = 0
    leakage_violations = 0
    total_episodes = 0
    prov_rows: dict[str, tuple] = {}
    if PILOT.exists():
        for method_dir in sorted(p for p in PILOT.iterdir() if p.is_dir()):
            for seed_dir in method_dir.glob("seed_*"):
                metrics_file = seed_dir / "metrics.json"
                if not metrics_file.exists():
                    check(f"{method_dir.name}: metrics.json present", False, str(seed_dir))
                    continue
                metrics = json.loads(metrics_file.read_text())
                method_status[method_dir.name] = metrics.get("status", gates.FAIL)
                complete = True
                if expected_counts.get("experience") and \
                        metrics.get("experience") != expected_counts["experience"]:
                    complete = False
                if metrics.get("full_run") is False:
                    complete = False
                final_name = f"S{expected_counts.get('experience', 0):03d}"
                summary_file = seed_dir / "probes" / final_name / "summary.json"
                if expected_counts.get("experience") and not summary_file.exists():
                    complete = False
                if complete and summary_file.exists():
                    summary = json.loads(summary_file.read_text())
                    for role, key in (("id", "id_probe"), ("transfer", "transfer_probe"),
                                      ("retention", "retention_probe")):
                        n = sum(1 for t in (summary.get("tasks") or {}).values()
                                if t.get("role") == role)
                        if expected_counts.get(role) and n != expected_counts[role]:
                            complete = False
                check(f"{method_dir.name}: complete canonical Pilot-60 run (counts + {final_name})",
                      complete, str(seed_dir))
                if metrics.get("probe_state_unchanged") is False:
                    probe_mutations += 1
                if metrics.get("probe_state_unchanged") is None:
                    snapshot_failures += 1
                infra_crashes += int(metrics.get("infra_errors") or 0)
                leakage_violations += int(metrics.get("leakage_violations") or 0)
                total_episodes += int(metrics.get("experience") or 0)
                prov_file = seed_dir / "provenance.json"
                if prov_file.exists():
                    prov = json.loads(prov_file.read_text())
                    prov_rows[f"{method_dir.name}/{seed_dir.name}"] = (
                        prov.get("model"), prov.get("model_base_url"),
                        prov.get("canonical_prompt_sha256"),
                        prov.get("canonical_agent_config_sha256"),
                        prov.get("pilot_manifest_sha256"))

    check("probe mutation count == 0", probe_mutations == 0, str(probe_mutations))
    check("snapshot reload failures == 0", snapshot_failures == 0, str(snapshot_failures))
    check("private leakage violations == 0", leakage_violations == 0, str(leakage_violations))

    primary = {"none", "raw_memory", "ace_context", "worldmind"}
    missing = sorted(primary - set(method_status))
    check("all primary conditions completed", not missing, f"missing: {missing}")

    if prov_rows:
        for idx, label in enumerate(("model", "base_url", "prompt_sha256",
                                     "agent_config_sha256", "manifest_sha256")):
            unique = {row[idx] for row in prov_rows.values()}
            check(f"all completed methods share the same {label}", len(unique) == 1,
                  str(sorted(str(u) for u in unique)))

    embodiskill_status = method_status.get("embodiskill", gates.BLOCKED)
    check("embodiskill recorded as BLOCKED (never silently PASS)",
          embodiskill_status == gates.BLOCKED, embodiskill_status)
    infra = summarize_infra(PILOT)
    check("infrastructure crash rate per experience episode < 2%",
          infra["rate_experience_episodes"] < 0.02,
          f"{infra['experience_infra_errors']}/{infra['experience_episodes']}")
    check("infrastructure crash rate per executed attempt < 2%",
          infra["rate_all_attempts"] < 0.02,
          f"{infra['experience_infra_errors'] + infra['probe_infra_errors']}"
          f"/{infra['experience_attempts'] + infra['probe_attempts']}")

    status = gates.FAIL if FAILS else gates.release_status(
        {**method_status, "embodiskill": embodiskill_status})
    lines = [f"FINAL_STATUS: {status}",
             f"METHOD_STATUS: {json.dumps(method_status, sort_keys=True)}"]
    if FAILS:
        lines += [f"- {f}" for f in FAILS]
    (PROJECT / "outputs" / "FINAL_STATUS_CANONICAL.txt").write_text("\n".join(lines) + "\n")
    if PILOT.exists():
        (PILOT / "FINAL_STATUS.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    return 0 if status in {gates.FULL_PILOT_PASS, gates.PIPELINE_PASS_WITH_BLOCKER} else 1


if __name__ == "__main__":
    sys.exit(main())
