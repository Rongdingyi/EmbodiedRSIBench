#!/usr/bin/env python
"""07_smoke_rsi.py -- Phase F / Gate G5: per-method RSI smoke.

Each method: 5 experience episodes + 3 frozen probe episodes (with the
probe-time state hash asserted unchanged). Method-specific evidence:
  none        -> state unchanged
  raw_memory  -> episodes stored, retrieval non-empty after first experience
  ace_context -> playbook changes after valid evidence, ops traceable
  worldmind   -> process/goal experience files appear, sidecar never touches action
  embodiskill -> BLOCKED (recorded, not faked)

Output: outputs/preflight/RSI_SMOKE.json
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

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

OUT = PROJECT / "outputs" / "preflight"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = {"none": NoneRSI, "raw_memory": RawMemoryRSI,
           "ace_context": AceContextRSI, "worldmind": WorldMindRSI,
           "embodiskill": EmbodiSkillRSI}
N_EXPERIENCE, N_PROBES = 5, 3
BUDGET = {"max_turns": 4, "max_tool_calls": 16, "timeout_s": 600,
          "max_total_tokens": 1_500_000, "max_env_steps": 60}


def main() -> int:
    manifest = json.loads((PROJECT / "manifests" / "pilot150.json").read_text())
    pilot_dir = PROJECT / "outputs" / "rsi_smoke"
    audits = {}
    statuses: dict[str, str] = {}
    for name, cls in METHODS.items():
        method = cls()
        state_root = pilot_dir / name / "rsi_state"
        state_root.mkdir(parents=True, exist_ok=True)
        method.init_run({"state_root": str(state_root), "run_id": "g5"})
        evidence = {"name": name, "blocked": getattr(method, "BLOCKED", False)}
        if evidence["blocked"]:
            evidence["reason"] = method.BLOCK_REASON
            evidence["status"] = gates.BLOCKED
            audits[name] = evidence
            statuses[name] = gates.BLOCKED
            print(f"[{name}] BLOCKED (recorded, not counted as PASS)")
            continue

        hashes = []
        exp_ids = manifest["experience"][:N_EXPERIENCE]
        probe_ids = manifest["id_probe"][:N_PROBES]
        start = time.time()
        for gid in exp_ids:
            task = by_global_id("experience")[gid]
            adapter = make_adapter(task["source_dataset"])
            try:
                res = run_episode(task, adapter, method, role="experience", config=BUDGET)
                if res.error:
                    evidence.setdefault("errors", []).append(f"exp {gid}: {res.error}")
            except Exception as exc:  # noqa: BLE001
                evidence.setdefault("errors", []).append(f"exp {gid}: {exc}")
            hashes.append(method.state_hash())
        # frozen probes: snapshot clone, read-only
        snapshot = pilot_dir / name / "snapshots" / "after_smoke"
        method.snapshot(snapshot)
        clone = method.clone_from_snapshot(snapshot)
        assert clone.update_enabled is False
        h0 = clone.state_hash()
        probe_success = []
        for gid in probe_ids:
            task = by_global_id("id")[gid]
            adapter = make_adapter(task["source_dataset"])
            try:
                res = run_episode(task, adapter, clone, role="id", config=BUDGET)
                probe_success.append(res.outcome.get("success"))
                if res.error:
                    evidence.setdefault("errors", []).append(f"probe {gid}: {res.error}")
            except Exception as exc:  # noqa: BLE001
                evidence.setdefault("errors", []).append(f"probe {gid}: {exc}")
        h1 = clone.state_hash()
        evidence.update({
            "state_hashes_after_experience": hashes,
            "probe_hash_before": h0,
            "probe_hash_after": h1,
            "probe_state_unchanged": h0 == h1,
            "probe_success": probe_success,
            "wall_s": round(time.time() - start, 1),
        })
        if name == "none":
            evidence["state_unchanged"] = len(set(hashes)) == 1
        if name == "raw_memory":
            episodes_file = state_root / "episodes.jsonl"
            stored = len(episodes_file.read_text().splitlines()) if episodes_file.exists() else 0
            injection = method.before_episode({"instruction": "pick up the mug",
                                               "source_dataset": "x"})
            evidence.update({"episodes_stored": stored,
                             "retrieval_nonempty": bool(injection.context_text),
                             "retrieval_tokens": injection.estimated_tokens})
        if name == "ace_context":
            pb = state_root / "playbook.txt"
            initial = "## STRATEGIES & INSIGHTS"
            current = pb.read_text() if pb.exists() else ""
            evidence["playbook_changed"] = current.strip() != "" and current != initial + "\n"
            evidence["curator_ops_file"] = (state_root / "curator_ops.jsonl").exists()
        if name == "worldmind":
            files = [str(p.relative_to(state_root)) for p in state_root.rglob("*") if p.is_file()]
            evidence["state_files"] = files[:12]
        audits[name] = evidence
        print(f"[{name}] probes_unchanged={evidence.get('probe_state_unchanged')} "
              f"evidence={ {k: v for k, v in evidence.items() if k in ('state_unchanged','episodes_stored','retrieval_nonempty','playbook_changed')} }")

    checks: dict[str, str] = {}
    for name, ev in audits.items():
        if ev.get("blocked"):
            checks[name] = gates.BLOCKED      # never coerced to PASS
            continue
        ok = (ev.get("probe_state_unchanged") is True) and not ev.get("errors")
        if name == "none":
            ok = ok and ev.get("state_unchanged") is True
        if name == "raw_memory":
            ok = ok and ev.get("retrieval_nonempty") is True
        if name == "ace_context":
            ok = ok and ev.get("playbook_changed") is True
        checks[name] = gates.PASS if ok else gates.FAIL
    status = gates.combine(checks)
    report = {"checks": checks, "methods": audits, "status": status}
    (OUT / "RSI_SMOKE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nG5 RSI SMOKE GATE: {status}  {checks}")
    return 0 if status in (gates.PASS, gates.BLOCKED) else 1


if __name__ == "__main__":
    sys.exit(main())
