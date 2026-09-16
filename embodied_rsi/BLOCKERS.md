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

## B2 — EB-ALFRED rendering needs the machine's physical X display (OPEN)

The 2018 AI2-THOR build presents through NVIDIA Vulkan, which cannot create a
swapchain for a private Xvfb; episodes require a real X server. Display
failures are infrastructure errors, never method failures.

State as of 2026-09-16 20:00: gdm restarted and the physical X server was
renumbered `:1` -> `:0`; the only socket we can reach is gdm's, which requires
an Xauthority we cannot read. `:99` Xvfb re-tested the same evening: the Unity
player aborts (SIGABRT, exit -6) -- Xvfb remains unusable for EB-ALFRED.

Need: a session on the physical display that authorises us, e.g. as the
console user `xhost +SI:localuser:rongdingyi` (then `_pick_display()` picks
`:0`). Impact: 6 of the 30 probe tasks per Pilot-60 checkpoint and 6 of the 30
experience episodes are EB-ALFRED; Pilot-60 attempt #1 was invalidated by this
(see `outputs/aborted_pilot60_displayfail_20260916/`).

## B3 — EB-Habitat episode lookup (RESOLVED in v0.5, verification pending on full G3)

The dataset re-orders/renames episodes, so the join is now pickle-authoritative:
`source_entry_index` addresses the official pickle order; the pickle episode
provides the full physical signature (scene + sampled_entities + start pose);
the dataset index is the unique signature match; ids are never used as keys.
Verified on 4 previously failing episodes (tools=70, step ok, evaluator ok).
`scripts/04_validate_adapters.py` must still confirm the full 20-task sample.

### Historical note (pre-v0.5)

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

## B5 — WorldMind upstream packaging bug (workaround in bridge)

At the pinned commit `712b0fd`, `worldmind_plugin/__init__.py` imports
`parse_llm_output` while `utils.py` defines `parse_agent_output`, so
`import worldmind_plugin` fails. The bridge loads the official submodules by
file into a pre-registered package namespace (no upstream edits, no algorithm
change). Reported here because a future WorldMind upgrade should fix it
upstream.

## B4 — gate re-run cost

G3/G4/G5 plus the Pilot-150 run are simulator- and API-heavy (multi-hour per
gate/method). Run them on an idle GPU machine; the G3 validator now uses a fresh
worker per task, which is safer but slower.
