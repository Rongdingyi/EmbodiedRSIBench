#!/usr/bin/env python
"""06b_smoke_canonical_agent.py -- one-episode canonical agent smoke (paid, small).

Verifies the canonical wire format end to end on a single TVR ID probe task
(current + target image both enter the request):
  - images attached to the planner request,
  - JSON response parses and validates against the legal action schema,
  - the accountant records planner usage,
  - the request dump contains no private sentinel.

Writes outputs/preflight/CANONICAL_AGENT_SMOKE.json and the episode under
outputs/canonical_smoke/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.adapters import make_adapter  # noqa: E402
from benchmark.agent.config import load_protocol_config  # noqa: E402
from benchmark.agent.episode_runner import run_episode  # noqa: E402
from benchmark.registry.loader import by_global_id  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402

OUT = PROJECT / "outputs" / "preflight"
SMOKE = PROJECT / "outputs" / "canonical_smoke"


def main() -> int:
    protocol = load_protocol_config(PROJECT / "configs" / "pilot60_canonical.yaml")
    manifest = json.loads((PROJECT / "manifests" / "pilot60.json").read_text())
    gid = manifest["id_probe"][0]
    task = by_global_id("id")[gid]
    episode_dir = SMOKE / gid
    episode_dir.mkdir(parents=True, exist_ok=True)

    method = NoneRSI()
    method.init_run({"state_root": str(SMOKE / "rsi_state"), "run_id": "canonical_smoke"})
    res = run_episode(task, make_adapter(task["source_dataset"]), method, role="id",
                      config=protocol, output_dir=episode_dir)

    dumps = []
    dump_path = episode_dir / "public_context_dump.jsonl"
    if dump_path.exists():
        dumps = [json.loads(line) for line in dump_path.read_text().splitlines() if line.strip()]
    planner_dumps = [d for d in dumps if d.get("role") == "canonical_planner"]
    with_image = [d for d in planner_dumps if d.get("has_image")]
    checks = {
        "episode ran": res.error is None,
        "planner request audited": len(planner_dumps) >= 1,
        "current image in request": len(with_image) >= 1,
        "planner usage recorded": res.planner_calls >= 1,
        "no private leakage": len(res.leakage_violations) == 0,
        "private evaluator decided success": res.outcome.get("success") is not None,
    }
    report = {
        "task_gid": gid,
        "source_dataset": task["source_dataset"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "planner_calls": res.planner_calls,
        "env_steps": res.env_steps,
        "stop_reason": res.stop_reason,
        "accounting": res.accounting,
        "image_count_first_request": (
            json.dumps(planner_dumps[0].get("redacted_body") or {}).count('"image_url"')
            if planner_dumps else 0),
        "wall_s": round(res.wall_time_s, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CANONICAL_AGENT_SMOKE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"CANONICAL AGENT SMOKE: {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
