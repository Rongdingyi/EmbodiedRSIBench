# Current State

## Last completed milestone
v0.5 correctness cleanup (second review round, 12 items) + deterministic regression.

## Status
G0/G1/G2 PASS. Deterministic regression PASS (none/raw_memory/WorldMind/
SpatialWorld EndTask/EB-Habitat fixed episodes). G3/G4/G5 must still be re-run
on the refactored stack before Pilot-150.

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

## Next step
1. `scripts/04_validate_adapters.py` (G3, fresh worker per task)
2. `scripts/06_smoke_openeta.py` (G4)
3. `scripts/07_smoke_rsi.py` (G5) -> then `08_run_pilot150.py` per method
4. `09_analyze_pilot.py` + `10_audit_release.py`
