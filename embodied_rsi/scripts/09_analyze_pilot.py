#!/usr/bin/env python
"""09_analyze_pilot.py -- Pilot-60 analysis (guide section 43/50).

Reads outputs/pilot150/*/seed_*/ and produces:
  outputs/pilot60/PILOT_REPORT.md
  outputs/pilot60/PILOT_RESULTS.csv
  outputs/pilot60/PILOT_RESULTS_BY_SOURCE.csv
  outputs/pilot60/PILOT_RESULTS_BY_SKILL.csv
  outputs/pilot60/PILOT_COST.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from benchmark.registry.loader import by_global_id  # noqa: E402

PILOT = PROJECT / "outputs" / "pilot60_canonical"
PROTOCOL = PROJECT / "configs" / "pilot60_canonical.yaml"


def load_probe_summary(root: Path, checkpoint: str) -> dict:
    path = root / "probes" / checkpoint / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def paired_transitions(first: dict, last: dict, role: str) -> dict:
    tasks0 = {g: t for g, t in (first.get("tasks") or {}).items() if t.get("role") == role}
    tasks1 = {g: t for g, t in (last.get("tasks") or {}).items() if t.get("role") == role}
    counts = {"n": 0, "fail_to_success": 0, "success_to_fail": 0,
              "success_to_success": 0, "fail_to_fail": 0}
    for gid, a in tasks0.items():
        b = tasks1.get(gid)
        if not b or not episode_counts_as_scored(a) or not episode_counts_as_scored(b):
            continue
        counts["n"] += 1
        if bool(a["success"]) == bool(b["success"]):
            counts["success_to_success" if a["success"] else "fail_to_fail"] += 1
        elif b["success"]:
            counts["fail_to_success"] += 1
        else:
            counts["success_to_fail"] += 1
    counts["net_gain"] = counts["fail_to_success"] - counts["success_to_fail"]
    return counts


def episode_counts_as_scored(task: dict) -> bool:
    """Protocol/infra-failed episodes never enter a success-rate denominator."""
    if task.get("status") != "PASS":
        return False
    return task.get("success") is not None


def success_rate(summary: dict, role: str) -> tuple[float | None, int]:
    tasks = [t for t in (summary.get("tasks") or {}).values() if t.get("role") == role]
    scored = [t for t in tasks if episode_counts_as_scored(t)]
    if not scored:
        return None, 0
    return sum(1 for t in scored if t["success"]) / len(scored), len(scored)


def main(argv: list[str] | None = None) -> int:
    global PILOT, PROTOCOL
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PILOT,
                        help="run root (canonical or historical)")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args(argv)
    PILOT = args.root
    PROTOCOL = args.protocol
    methods = sorted(p.name for p in PILOT.iterdir() if p.is_dir()) if PILOT.exists() else []
    rows = []
    paired_rows = []
    excluded_rows = []
    by_source_rows = []
    by_skill_rows = []
    cost_rows = []
    for method in methods:
        seeds = sorted((PILOT / method).glob("seed_*"))
        for seed_dir in seeds:
            metrics = {}
            mp = seed_dir / "metrics.json"
            if mp.exists():
                metrics = json.loads(mp.read_text())
            checkpoints = sorted((seed_dir / "probes").glob("S*"),
                                 key=lambda p: p.name) if (seed_dir / "probes").exists() else []
            first = load_probe_summary(seed_dir, checkpoints[0].name) if checkpoints else {}
            last = load_probe_summary(seed_dir, checkpoints[-1].name) if checkpoints else {}

            def gains(summary_first: dict, summary_last: dict, role: str):
                p0, n0 = success_rate(summary_first, role)
                p1, n1 = success_rate(summary_last, role)
                gain = (p1 - p0) if (p0 is not None and p1 is not None) else None
                return p0, p1, gain, n0, n1

            excluded = []
            for ckpt_name, summary in (("S000", first), ("S030", last)) if first or last else ():
                for gid, task in (summary.get("tasks") or {}).items():
                    if task.get("status") != "PASS":
                        excluded.append({
                            "method": method, "seed": seed_dir.name, "checkpoint": ckpt_name,
                            "role": task.get("role"), "task": gid,
                            "status": task.get("status"),
                            "stop_reason": task.get("stop_reason") or "",
                            "error": task.get("error") or "",
                            "attempts": task.get("attempts"),
                        })
            excluded_rows.extend(excluded)
            id0, id1, idg, idn0, idn1 = gains(first, last, "id")
            tr0, tr1, trg, trn0, trn1 = gains(first, last, "transfer")
            rt0, rt1, rtg, rtn0, rtn1 = gains(first, last, "retention")
            for role in ("id", "transfer", "retention"):
                paired_rows.append({"method": method, "seed": seed_dir.name, "role": role,
                                    **paired_transitions(first, last, role)})
            rows.append({
                "method": method, "seed": seed_dir.name,
                "experience": metrics.get("experience"),
                "initial_id": id0, "final_id": id1, "id_gain": idg, "id_n": idn0,
                "initial_transfer": tr0, "final_transfer": tr1, "transfer_gain": trg,
                "transfer_n": trn0,
                "initial_retention": rt0, "final_retention": rt1, "retention_gain": rtg,
                "updater_tokens": metrics.get("updater_tokens"),
                "env_steps": metrics.get("env_steps"),
                "probe_state_unchanged": metrics.get("probe_state_unchanged"),
                "planner_calls": metrics.get("planner_calls"),
                "planner_tokens": ((metrics.get("planner_input_tokens") or 0)
                                   + (metrics.get("planner_output_tokens") or 0)) or None,
                "planner_validation_retries": metrics.get("planner_validation_retries"),
                "planner_protocol_failures": metrics.get("planner_protocol_failures"),
                "excluded_tasks": len(excluded),
            })
            # by source / by skill
            for role, key in (("id", "initial_id"), ("transfer", "initial_transfer"),
                              ("retention", "initial_retention")):
                for ckpt_name, summary in ((checkpoints[0].name if checkpoints else "", first),
                                           (checkpoints[-1].name if checkpoints else "", last)):
                    for gid, t in (summary.get("tasks") or {}).items():
                        if t.get("role") != role:
                            continue
                        try:
                            rec = by_global_id(role)[gid]
                        except KeyError:
                            continue
                        by_source_rows.append({
                            "method": method, "checkpoint": ckpt_name,
                            "source": rec["source_dataset"],
                            "skill_family": rec.get("skill_family_primary") or "",
                            "role": role,
                            "success": t.get("success"), "turns": t.get("turns"),
                            "error": t.get("error") or "",
                        })
            cost_rows.append({
                "method": method,
                "experience_episodes": metrics.get("experience"),
                "state_updates": metrics.get("state_updates"),
                "wall_s": metrics.get("wall_s"),
                "probe_state_unchanged": metrics.get("probe_state_unchanged"),
                "updater_tokens": metrics.get("updater_tokens"),
                "env_steps": metrics.get("env_steps"),
            })

    # aggregate by source
    source_agg = defaultdict(lambda: {"n": 0, "success": 0})
    for r in by_source_rows:
        if r["success"] is None:
            continue
        key = (r["method"], r["source"])
        source_agg[key]["n"] += 1
        source_agg[key]["success"] += int(bool(r["success"]))

    PILOT.mkdir(parents=True, exist_ok=True)
    with open(PILOT / "PILOT_RESULTS.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                ["method", "seed"])
        writer.writeheader()
        writer.writerows(rows)
    with open(PILOT / "PILOT_RESULTS_BY_SOURCE.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "source", "n", "success_rate"])
        for (method, source), agg in sorted(source_agg.items()):
            writer.writerow([method, source, agg["n"], round(agg["success"] / agg["n"], 4)])
    with open(PILOT / "PILOT_RESULTS_BY_SKILL.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "skill_family", "role", "checkpoint", "n", "success_rate"])
        skill_agg = defaultdict(lambda: [0, 0])
        for r in by_source_rows:
            if r["success"] is None or not r.get("skill_family"):
                continue
            key = (r["method"], r["skill_family"], r["role"], r["checkpoint"])
            skill_agg[key][0] += 1
            skill_agg[key][1] += int(bool(r["success"]))
        for (method, family, role, ckpt), (n, ok) in sorted(skill_agg.items()):
            writer.writerow([method, family, role, ckpt, n, round(ok / n, 4)])
    with open(PILOT / "PILOT_COST.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cost_rows[0].keys()) if cost_rows else
                                ["method"])
        writer.writeheader()
        writer.writerows(cost_rows)
    with open(PILOT / "PILOT_EXCLUDED.csv", "w", newline="") as f:
        fields = ["method", "seed", "checkpoint", "role", "task", "status",
                  "stop_reason", "error", "attempts"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(excluded_rows)
    with open(PILOT / "PILOT_PAIRED.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "seed", "role", "n", "fail_to_success",
                           "success_to_fail", "success_to_success", "fail_to_fail", "net_gain"])
        writer.writeheader()
        writer.writerows(paired_rows)

    lines = ["# Pilot-60 Report", "",
             "| Method | Init ID | Final ID | ID Gain | Init Transfer | Final Transfer | "
             "Transfer Gain | Retention | Probe state OK |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        def fmt(v):
            return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| {method} | {a} | {b} | {c} | {d} | {e} | {f} | {g} | {h} |".format(
            method=r["method"], a=fmt(r["initial_id"]), b=fmt(r["final_id"]),
            c=fmt(r["id_gain"]), d=fmt(r["initial_transfer"]), e=fmt(r["final_transfer"]),
            f=fmt(r["transfer_gain"]), g=fmt(r["final_retention"]),
            h=r["probe_state_unchanged"]))
    stop_reasons = defaultdict(int)
    for method in methods:
        for seed_dir in (PILOT / method).glob("seed_*"):
            for ckpt in sorted((seed_dir / "probes").glob("S*")) if (seed_dir / "probes").exists() else []:
                summary_path = ckpt / "summary.json"
                if not summary_path.exists():
                    continue
                summary = json.loads(summary_path.read_text())
                for task in (summary.get("tasks") or {}).values():
                    reason = task.get("stop_reason")
                    if reason:
                        stop_reasons[(method, reason)] += 1
    if stop_reasons:
        lines += ["", "## Stop reasons (probe episodes)",
                  "| Method | Stop reason | n |", "|---|---|---:|"]
        for (method, reason), n in sorted(stop_reasons.items()):
            lines.append(f"| {method} | {reason} | {n} |")
    lines += ["", "## Excluded from success rates (non-PASS episodes)",
              "Protocol/infrastructure-failed episodes are counted here, never as task failures.",
              "",
              "| Method | Checkpoint | Role | n excluded |",
              "|---|---|---|---:|"]
    excl_agg = defaultdict(int)
    for row in excluded_rows:
        excl_agg[(row["method"], row["checkpoint"], row["role"] or "-")] += 1
    if excl_agg:
        for (method, ckpt, role), n in sorted(excl_agg.items()):
            lines.append(f"| {method} | {ckpt} | {role} | {n} |")
    else:
        lines.append("| - | - | - | 0 |")
    lines += ["", "## Paired transitions (S000 -> Sfinal)",
              "",
              "Same frozen tasks at both checkpoints; a task counts only when both runs "
              "produced a scored outcome.",
              "",
              "| Method | Role | n | fail->success | success->fail | success->success | fail->fail | Net gain |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in paired_rows:
        lines.append("| {method} | {role} | {n} | {fail_to_success} | {success_to_fail} | "
                     "{success_to_success} | {fail_to_fail} | {net_gain:+d} |".format(**r))
    lines += ["", "## Excluded conditions",
              "- `embodiskill`: BLOCKED (see BLOCKERS.md / EMBODISKILL_CALL_PATH.md).",
              "", "## Notes",
              "- Success rates use the official per-source private evaluators.",
              "- Probe checkpoints re-run the same frozen tasks with updates disabled "
              "(Pilot-60: S000 and S030).",
              "- Token accounting: planner calls are counted per episode; exact token "
              "totals are emitted by the rollout recorder when enabled."]
    (PILOT / "PILOT_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"analysis written for {len(rows)} (method, seed) runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
