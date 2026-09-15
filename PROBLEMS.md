# Current Problems & Status — Embodied RSI OpenETA Pilot (v0.4)

Date: 2026-09-15 · Project: `embodied_rsi/` · Model: DeepSeek `deepseek-flash`

Consolidates the post-review state. The review's P0/P1 items were implemented;
the gates must be re-run before the pilot. Evidence paths are relative to
`embodied_rsi/`.

## 1. Gate summary (post-fix, gates NOT yet re-run)

| Gate | Status | Notes |
|---|---|---|
| G0 Dataset | PASS | `outputs/preflight/DATASET_AUDIT.json` |
| G1 OpenETA Freeze | PASS | `outputs/preflight/OPENETA_FREEZE.json` (pin, hashes, native SI off) |
| G2 Model | PASS | `outputs/api_smoke/deepseek_flash.json` (text/vision/backend; no `chat_template_kwargs`) |
| G3 Adapter | **must re-run** | Previous run: 16/100 crash (EB-Habitat alignment + TVR timeouts). Fixes below; re-run `scripts/04_validate_adapters.py` |
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
