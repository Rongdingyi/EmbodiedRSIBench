# Current Problems & Status — Embodied RSI OpenETA Pilot (v0.6)

Date: 2026-09-15 · Project: `embodied_rsi/` · Model: DeepSeek `deepseek-flash`

Consolidates the post-review state. The review's P0/P1 items were implemented;
the gates must be re-run before the pilot. Evidence paths are relative to
`embodied_rsi/`.

## 0.1 v0.6 review round — all 6 items addressed

| # | Item | Fix |
|---|---|---|
| 1 | WorldMind compared prediction vs pre-action state | `process_single_step(step=ProcessTrajectoryStep(observation=state_after, ...), state_before=state_before)`; `has_error` from the official return is honoured. Deterministic spy test `tests/test_worldmind_timing.py` asserts the exact arguments |
| 2 | canonical run could reuse old RSI state | `08_run_pilot150.py` fails closed when the output root is non-empty (no auto-delete); `--output-root` for dev smokes, `--overwrite` for explicit resume |
| 3 | final audit could accept a small smoke as a full method | audit now requires `experience == 75`, full probe counts (25/30/20), `S075` summary present with matching task counts, and `full_run` — plus each Gate's own `status == PASS` (G5: PASS-or-BLOCKED) |
| 4 | WorldMind updater errors were swallowed | env records `after_step` exceptions into `rsi_step_errors` → `rsi_update_error` + episode status FAIL; WorldMind `strict_updates=True` re-raises official `has_error`/goal/reload failures; G5 asserts `sidecar_calls > 0`, component calls recorded, `errors.jsonl` empty |
| 5 | accounting double count + wrong call counts | instrumentation now only dumps requests + accumulates per-episode usage; the runner is the single recorder, with `calls=usage["calls"]`; usage windows reset in `before_episode` |
| 6 | EmbodiSkill manual rendering | worker calls the official `get_active_manual_text()` (fallback renders official `sections[].items`) |

Verification this round: `tests/test_worldmind_timing.py` PASS (sidecar ran once
pre-step, `observation=state_after` + `state_before` kwarg asserted, accountant
single-authority), fail-closed run refused a non-empty output root, 8/8 unit
tests PASS.

## 0. v0.5 review round — all 12 items addressed

| # | Item | Fix |
|---|---|---|
| 1 | clone wrote into live state | `clone_ctx` puts temp `state_root` last + hard path assertions; test asserts clone/live/snapshot dirs differ |
| 2 | `last_update_usage` name collision | renamed to `_last_update_usage` + `get_last_update_usage()`; runner uses the accessor |
| 3 | update failures swallowed by gates | runner and G5/G6/G7 count `res.status != PASS`; any `rsi_update_error` fails the run |
| 4 | WorldMind actual state was a task id | environment passes `task_instruction` + `state_before` + `state_after`; process module compares predicted vs real post-action state |
| 5 | G3 SpatialWorld old tool name | validator now calls `Rotate {direction:right, degrees:90}` |
| 6 | SpatialWorld `EndTask` didn't terminate | terminal branch returns `terminated=True, action_success=True` (verified) |
| 7 | EB-Habitat alignment still broken | join is now pickle-authoritative (release index → official pickle episode → full signature incl. pose → unique dataset index); 4/4 previously failing episodes now pass |
| 8 | release status self-contradiction | `rsi_smoke` accepts PASS-or-BLOCKED; exit code accepts `FULL_PILOT_PASS`/`PIPELINE_PASS_WITH_BLOCKER` |
| 9 | cost accounting wrong/double counted | ACE reads official `prompt_num_tokens`/`response_num_tokens` and sends `max_tokens` (provider != "openai"); WorldMind sidecar no longer double counted; official `LLMClient._call_api` instrumented → discriminator/reflector/refiner tokens + request dumps captured (verified: sidecar 467 tok, components 3256 tok, roles in dump) |
| 10 | request audit was planner-only | ACE updater client wrapped; WorldMind sidecar and components recorded; `private_reference_values` now high-precision (ids + canonical private JSON + identifier-like strings, no bare numbers) |
| 11 | EmbodiSkill PoC constructor wrong | official dataclass fields (`namespace/global_config/llm_model/embedding_func`) + DeepSeek `LLMCallable`; manual guidance renders official `sections[].items` |
| 12 | skill-family analysis placeholder | `PILOT_RESULTS_BY_SKILL.csv` aggregates registry skill labels per method/role/checkpoint |

Deterministic regression (v0.5, all PASS): none exp+probe, raw_memory exp+retrieval,
WorldMind 1 experience (sidecar+components accounted), SpatialWorld `EndTask`,
EB-Habitat 4 previously-failing episodes. Unit tests: 7/7 PASS.

## 1. Gate summary (post-fix, gates NOT yet re-run)

| Gate | Status | Notes |
|---|---|---|
| G0 Dataset | PASS | `outputs/preflight/DATASET_AUDIT.json` |
| G1 OpenETA Freeze | PASS | `outputs/preflight/OPENETA_FREEZE.json` (pin, hashes, native SI off) |
| G2 Model | PASS | `outputs/api_smoke/deepseek_flash.json` (text/vision/backend; no `chat_template_kwargs`) |
| G3 Adapter | **must re-run** | Previous run: 16/100 crash. EB-Habitat join fixed and verified on 4 previously-failing episodes; validator now also fails on empty schema. Re-run `scripts/04_validate_adapters.py` |
| G4 Baseline | **must re-run** | Now runs through the upstream episode runner; per-source budgets + accountant; re-run `scripts/06_smoke_openeta.py` |
| G5 RSI Smoke | **must re-run** | Three-state gate; blocked method is BLOCKED, never PASS |
| G6 Snapshot | PASS (unit) | `tests/test_snapshot_reload.py` under the new new-instance clone semantics |
| G7 Pilot-150 | not started | manifest rebuilt (balanced); still gated behind G3/G4/G5 |

## 2. Review items addressed in this revision

### P0
1. **Canonical OpenETA episode loop** — `benchmark/openeta_bridge/benchmark_environment.py`
   implements the upstream `EpisodeEnvironment` protocol; `episode_runner.py` now
   delegates the closed loop to the pinned `OpenEtaEpisodeRunner` (completion,
   tool/token budgets, timeout, environment receipts, recovery). Verified with a
   real TVR episode: `status=PASS`, upstream `stop_reason=max_turns`,
   `runner_tokens=29922`, planner accounting populated, leakage 0.
2. **WorldMind prediction timing** — `predict_sidecar()` runs inside
   `BenchmarkEpisodeEnvironment.step()` *after* the action is locked and *before*
   `adapter.step()`; `after_step()` then feeds the real feedback into the
   official `process_single_step()`. Sidecar output is never returned to the
   planner and its tokens are accounted separately.
3. **EB-Habitat action schema** — two-phase environment preparation
   (`env.prepare()` → live schema → `build_runtime`) removes the
   build-before-reset bug; the worker now raises on an empty/miscounted schema
   (expected 70 skills) and exposes a schema hash.
4. **Probe snapshot clone** — `RSIMethod.clone_from_snapshot()` creates a **new
   instance** operating on a temporary copy; `load_snapshot()` copies into the
   instance's own state dir and calls `_reload_state()` (official modules /
   OpenAI clients are never deep-copied). Unit-tested that probes cannot mutate
   the snapshot.
5. **Gate correctness** — `benchmark/runner/gates.py` gives PASS/FAIL/BLOCKED
   semantics; `run_episode` errors count as infrastructure errors in G4/G6/G7;
   G5 records BLOCKED as BLOCKED; release audit emits two-level status
   (`FULL_PILOT_PASS` / `PIPELINE_PASS_WITH_BLOCKER` / `FAIL`).
6. **Retention semantics** — removed the ad-hoc "retention anchor → experience
   anchor" pairing (`scripts/05_build_pilot150.py`); retention anchors are fixed
   held-out tasks re-measured at checkpoints.

### P1
7. **Pilot balance** — deficit filling levels all non-anchor sources; final
   distribution: TVR 23 (mandatory pair anchors), SpatialWorld 13, Asgard 13,
   EB-ALFRED 13, EB-Habitat 13 (was 37/15/5/10/8).
8. **SpatialWorld action semantics** — worker now exposes the source abstraction
   `Move / Rotate / Tilt / ChangePosture / Pick / Place / ChangeState /
   Manipulate / EndTask` mapped to AI2-THOR primitives.
9. **Observation allowlist** — `BenchmarkEpisodeEnvironment` only forwards
   `image_roles` (+ optional per-source object-type list, default off);
   simulator diagnostics stay out of planner metadata.
10. **Request-level leakage audit** — every planner request is stored redacted in
    `public_context_dump.jsonl` and scanned for forbidden keys/private values;
    per-episode `LEAKAGE_AUDIT.json` is written by the runner.
11. **Budgets** — OpenETA turns/tool calls (upstream runner) and environment
    steps (environment counter → truncation) are separate, config-driven budgets
    (`configs/pilot150.yaml`).
12. **Cost accounting** — `benchmark/runner/accounting.py` centralizes planner,
    injection, updater, sidecar, embedding and simulator-step accounting into
    `token_usage.json`; `metrics.json` aggregates it per method.
13. **ACE episode index** — the pilot runner calls `method.advance_step(index,
    total)` before every experience episode.
14. **EmbodiSkill** — re-evaluated: the official
    `init_task_context → move_skill_state → save_task_context → reflect_episode →
    revise_manual` path makes a thin adapter feasible; converter implemented in
    `benchmark/adapters/workers/embodiskill_worker.py` (≤200 lines, no upstream
    edits) with status **POC_READY_UNVERIFIED** — C4 stays out of the pilot table
    until the PoC runs in its dependency env.
15. **TVR target image** — the immutable target view is returned in every step
    observation, not only at reset.
16. **Validator isolation** — G3 creates a fresh worker per task.

## 3. Remaining problems / next actions (in order)

1. **Re-run G3** (`scripts/04_validate_adapters.py`): expect the EB-Habitat
   alignment failure to be replaced by the explicit episode-lookup error if the
   id/instruction join still fails; the worker now resolves by instruction then
   episode_id and fails loudly otherwise. If it still fails, add the
   physical-signature join (scene + sampled_entities + start pose).
2. **Re-run G4** (`scripts/06_smoke_openeta.py`): RGB delivery threshold 90 %;
   EB-Habitat was the dominant miss.
3. **Re-run G5** (`scripts/07_smoke_rsi.py`) and then the pilot
   (`scripts/08_run_pilot150.py --method none|raw_memory|ace_context|worldmind`).
   Expect multi-hour wall time per method (DeepSeek thinking + simulators).
4. **EmbodiSkill PoC execution** — provision the EmbodiSkill env
   (`langchain-chroma`, `sentence-transformers`, `finch`) and run
   `benchmark/adapters/workers/embodiskill_worker.py`; verify the four official
   reflection categories appear, then decide C4 inclusion.
5. **Cost per method** — after the pilot, extend `09_analyze_pilot.py` by-skill
   analysis with registry skill labels (currently emitted only for source).
6. **Portability** — worker python paths and the data root are still
   machine-specific; move to a `config.yaml` path map before Full Core.

## 4. Verified working (so problems stay in context)

- Upstream-runner episode: TVR + `none`, 3 turns, 3 env steps, PASS, leakage 0,
  accounting populated (`/tmp/opencode/new_runner_test.log` evidence in-session).
- Unit tests: freeze, gate status, snapshot reload, probe readonly, RSI state
  isolation, observation allowlist, adapter action equivalence — all PASS.
- Freeze invariants: pinned commit, clean tree, critical hashes, native SI
  disabled and asserted each run; planner prompt hash fixed; benchmark-only tool
  registry.
