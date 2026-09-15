# BLOCKERS

## B1 — EmbodiSkill (Condition E): POC_READY_UNVERIFIED

Status: **re-evaluated after review**. The official integration path
(`init_task_context → move_skill_state → save_task_context → reflect_episode →
revise_manual`) makes a thin adapter feasible without upstream edits. The
converter is implemented in `benchmark/adapters/workers/embodiskill_worker.py`
(≤200 lines) but has **not been executed yet** because it needs the EmbodiSkill
dependency env (`langchain-chroma`, `sentence-transformers`, `finch`), kept out
of the agent environment on purpose.

C4 stays out of the pilot table until the PoC runs and produces the four official
reflection categories. Audit: `outputs/preflight/EMBODISKILL_CALL_PATH.md`.

## B2 — EB-ALFRED rendering needs the machine's physical X display

The 2018 AI2-THOR build presents through NVIDIA Vulkan, which cannot create a
swapchain for a private Xvfb; episodes require a real X server (this machine
`:1`). Mitigation: `scripts_render/00_pick_display.sh`; display failures are
infrastructure errors, never method failures.

## B3 — EB-Habitat episode lookup (open; gates must re-run)

`EBHabEnv(eval_set=...)` re-orders/renames its internal dataset: neither
`source_entry_index` (pickle order) nor `source_task_id` (pickle episode_id)
addresses the loaded episode (e.g. index 41 → episode 32 while the release
expects 74). Current worker behaviour: resolve by exact instruction first, then
by episode_id, and fail loudly otherwise; the canonical 70-skill schema is
validated at reset.

If the instruction/id join still fails for some tasks, the next fix is the
physical signature already stored in the release:
`(basename(scene_id), canonical_json(sampled_entities), start_position[4dp])`.

Re-run `scripts/04_validate_adapters.py` (target crash rate ≤ 2%, schema
failures = 0, leakage = 0) and `scripts/06_smoke_openeta.py` (RGB delivery ≥ 90%)
after any change here.

## B4 — gate re-run cost

G3/G4/G5 plus the Pilot-150 run are simulator- and API-heavy (multi-hour per
gate/method). Run them on an idle GPU machine; the G3 validator now uses a fresh
worker per task, which is safer but slower.
