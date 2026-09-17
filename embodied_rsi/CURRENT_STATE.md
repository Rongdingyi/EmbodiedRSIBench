# Current State

## Last completed milestone
G7 Pilot-60 COMPLETE: all four primary conditions PASS with zero infra
errors / zero leakage / zero probe mutation. Release audit:
`PIPELINE_PASS_WITH_BLOCKER` (C4 embodiskill BLOCKED per protocol).
`outputs/FINAL_STATUS.txt`, `outputs/pilot60/PILOT_REPORT.md`.

## Status
G0/G1/G2/G3/G4/G5/G6 ALL PASS (evidence in `outputs/preflight/`).
Pilot v0.7 / Pilot-60 protocol: 30 experience (6/source) + 10 ID + 10
transfer + 10 retention = 60 unique tasks, checkpoints S000 -> S030, 90
episodes/method (360 total). Manifest: `manifests/pilot60.json`
(`05b_build_pilot60.py`, audit PASS). Runner: `08_run_pilot60.py`
(`outputs/pilot60/`), analysis: `09_analyze_pilot.py` (adds paired
transitions), release audit: `10_audit_release.py` (30/10/10/10 + S030).
A canonical run must start from an empty `outputs/pilot60/`; use `--overwrite`
only after archiving a previous attempt.

## Regression evidence (all PASS)
- none: 1 experience + 1 probe, probe state hash unchanged, clone/live/snapshot
  dirs provably distinct.
- raw_memory: 1 experience, retrieval injection non-empty, usage accessor OK.
- WorldMind: 1 experience, sidecar 467 tok (2 calls) accounted separately,
  components 3256 tok (3 calls) via official LLMClient instrumentation,
  request dump roles = [planner, worldmind_sidecar, worldmind_component].
- SpatialWorld `EndTask`: terminated=True, action_success=True.
- EB-Habitat: episodes 16 / 172 / 102 (and 218) now resolve via pickle-
  authoritative physical signature; 70 tools, step ok, evaluator ok.

## Key fixes this round
1. `clone_from_snapshot` temp `state_root` wins (was overwritten by run_ctx);
   tests assert clone != live != snapshot directories.
2. `_last_update_usage` + `get_last_update_usage()` replaces the attribute/method
   name collision; runner uses the accessor.
3. `res.status != PASS` counts as an episode failure in the pilot runner, G4 and
   G5; `rsi_update_error` can no longer be swallowed.
4. WorldMind receives `task_instruction`, `state_before`, `state_after` — the
   official process module no longer compares a prediction against a task id.
5. G3 SpatialWorld smoke uses `Rotate`; SpatialWorld `EndTask` terminates.
6. EB-Habitat: release index -> official pickle episode -> full signature
   (scene + sampled_entities + start pose) -> unique dataset index (ids are never
   keys). Instruction fallback removed.
7. Release audit accepts PASS-or-BLOCKED for RSI smoke and two-level final
   statuses in the exit code.
8. Cost accounting: ACE official token fields + `max_tokens` for DeepSeek;
   sidecar no longer double counted; official WorldMind `LLMClient._call_api`
   instrumented (component tokens + per-request dumps).
9. Request audit covers planner + ACE updater + WorldMind sidecar/components;
   leakage sentinels are high precision (no bare-number false positives).
10. EmbodiSkill PoC constructor uses official dataclass fields and renders
    official `sections[].items`.
11. `PILOT_RESULTS_BY_SKILL.csv` aggregates registry skill labels.

## Known issues / upstream notes
- WorldMind pinned commit has a broken package `__init__` (`parse_llm_output`
  missing) — bridge loads official submodules by file (BLOCKERS.md B5).
- EmbodiSkill PoC still needs its dependency env to execute (B1).
- G3/G4/G5 and the pilot itself are simulator/API heavy; run on an idle machine.

## v0.6 fixes
1. WorldMind now passes `observation=state_after` and `state_before=` (official
   semantics), `has_error` honoured; spy test asserts both arguments.
2. Canonical runs fail closed on a non-empty output root (no silent state reuse).
3. Release audit requires full Pilot-60 counts, `S030` summaries with matching
   probe counts, and each Gate's own status.
4. WorldMind updater errors surface as `rsi_update_error` + episode FAIL; G5
   asserts sidecar/component calls and an empty `errors.jsonl`.
5. Accounting single authority (runner), `calls` from usage, per-episode reset.
6. EmbodiSkill worker uses the official `get_active_manual_text()`.

## Next step
1. `scripts/04_validate_adapters.py` (G3, fresh worker per task)
2. `scripts/06_smoke_openeta.py` (G4)
3. `scripts/07_smoke_rsi.py` (G5) -> then `08_run_pilot60.py` per method
4. `09_analyze_pilot.py` + `10_audit_release.py`
