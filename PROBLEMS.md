# Current Problems & Status — Embodied RSI OpenETA Pilot (v0.3)

Date: 2026-09-15 · Project: `embodied_rsi/` · Dataset: Core v1.0 · Model: DeepSeek `deepseek-flash`

This document consolidates **what works, what is blocked, and why**.
Evidence paths are relative to `embodied_rsi/`.

## 1. Gate summary

| Gate | Status | Evidence |
|---|---|---|
| G0 Dataset | **PASS** | `outputs/preflight/DATASET_AUDIT.json` (8046 raw / 2200 core / 1500-300-300-100 / disjoint) |
| G1 OpenETA Freeze | **PASS** | `outputs/preflight/OPENETA_FREEZE.json`, `manifests/openeta_freeze_manifest.json` (pin, clean tree, 11 hashes, native SI disabled + asserted) |
| G2 Model | **PASS** | `outputs/api_smoke/deepseek_flash.json` (text/vision OK; pinned backend returns valid `tool_call`; no `chat_template_kwargs`) |
| G3 Adapter | **FAIL** | `outputs/preflight/ADAPTER_VALIDATION.json` — 100 tasks attempted, 84 clean, 16 crashes (16%), leakage 0 |
| G4 OpenETA Baseline | **FAIL** | `outputs/preflight/OPENETA_BASELINE.json` — 25 tasks, 0 infra crashes, invalid-action rate 0.0, 0 native-SI writes, but RGB delivery 19/25 (< 90%) |
| G5 RSI Smoke | not run | blocked behind G3/G4 by the guide's no-skip rule |
| G6 Snapshot | unit-tested | `tests/test_snapshot_reload.py` PASS (all implemented methods) |
| G7 Pilot-150 | **not started** | manifest ready; multi-hour simulator+API budget |

## 2. P1 — EB-Habitat episode alignment (dominant G3 failure, 14/20 samples)

**Symptom (G3).** EB-Habitat samples crash during `reset`:
- `RuntimeError: EB-Habitat episode mismatch: dataset index 41 is episode 32, expected 74`
- after switching the join key to instruction: `episode mismatch after reset: got 35, expected 16`
- some resets crash the worker process outright (`worker closed stdout on reset`).

**Root cause.** `EBHabEnv(eval_set=...)` (EmbodiedBench) builds its own episode
ordering *and its own `episode_id` values*. Neither the release's
`source_entry_index` (pickle order) nor `source_task_id` (pickle `episode_id`)
addresses the episode that EmbodiedBench's env actually loads:

```
release  : source_entry_index = 41, episode_id = 74
EBHabEnv : dataset[41].episode_id = 32, instruction-match loads episode 35
```

The instruction is not unique enough (rewrites share templates), so one-to-one
mapping is unsafe.

**Impact.** 100% of EB-Habitat G4 samples deliver no image (its resets fail
silently into short `ask_human` episodes), which is the entire reason G4 is
below threshold (19/25 vs 21/25 needed).

**Fix plan (next session).**
1. In `benchmark/adapters/workers/eb_habitat_worker.py`, build the dataset index
   from the physical signature already stored in the release:
   `(basename(scene_id), canonical_json(sampled_entities), start_position[4dp])`
   → dataset index; assert uniqueness; keep instruction only as a tie-breaker.
2. Re-run `scripts/04_validate_adapters.py` → target crash rate ≤ 2%.
3. Re-run `scripts/06_smoke_openeta.py` → expect 24/25 image delivery (96%).

## 3. P2 — transient AI2-THOR timeouts under GPU contention (2/20 TVR samples)

Two TVRBench resets hit `TimeoutError: Reading from AI2-THOR backend timed out`
and one 300 s worker timeout while all four GPUs were simultaneously saturated
by unrelated processes (28–44 GB used per card).

**Mitigation already applied.** `scripts/04_validate_adapters.py` retries
transient timeout failures once with a fresh worker.
**Recommendation.** Run gates when the GPUs are not saturated; keep one worker
per source (already the case) rather than parallel workers on the same GPU.

## 4. P3 — EmbodiSkill (Condition E) is BLOCKED

Per the guide's own stop condition (sections 2.5 / 21.7): the official
implementation is ALFWorld/agentkit-specific (`StateChain`, reflection epoch
files and manual injection are owned by the agentkit executor). Wiring an
external OpenETA trajectory would rewrite reflection/update semantics (>200
lines). Full call-path audit: `outputs/preflight/EMBODISKILL_CALL_PATH.md`.
**No pseudo-EmbodiSkill was shipped.** Pilot primary table therefore covers
C0–C3 (`none`, `raw_memory`, `ace_context`, `worldmind`).

## 5. P4 — EB-ALFRED requires the machine's physical X server

The 2018 AI2-THOR 2.1.0 build renders through NVIDIA Vulkan and cannot create a
swapchain for a private Xvfb (SIGABRT). It works on the console X server
(this machine `:1`) with `-force-vulkan`. Mitigations already in place:
`scripts_render/00_pick_display.sh` auto-picks a working display; the adapter
reports display failures as infrastructure errors (never as method failures).
Server moves must re-check `DISPLAY`.

## 6. P5 — token / cost accounting is partial

Per the guide (§33) the pilot must report planner input/output tokens, RSI
injection/update tokens, sidecar tokens, embedding calls and env steps.
Currently: planner calls per episode are counted (`audit_requests`), injection
tokens are budget-checked and logged, WorldMind sidecar usage is captured per
call — but nothing is aggregated into `metrics.json` yet, and exact planner
token totals need the upstream rollout recorder wired into the runner.
`PILOT_COST.csv` is therefore emitted with the fields it can fill.

## 7. P6 — environment portability

- Workers hardcode machine paths (`/media/moxc/...` data root, conda env pythons
  in `benchmark/adapters/__init__.py`). Only the data root honors an env var
  (`EMBODIED_RSI_DATA`). A `config.yaml`-driven path map is the next cleanup.
- `external/OpenETA/.venv` needs `pyarrow` and `openai` added after `uv sync`
  (the pinned repo itself is untouched; freeze manifest covers repo code only).
- Upstream repos are not vendored (by design); `scripts/01_clone_pin_external.sh`
  recreates them at the exact commits recorded in `manifests/external_sources.json`.

## 8. What is verified working (so the problems above stay in context)

- Full canonical episode loop on the pinned agent: fresh session per episode,
  benchmark-only tool registry, RGB frames materialized into the planner's
  `image_artifacts` slot, tool calls executed in real simulators, private
  evaluators from each source, `chat_template_kwargs` never sent, native
  self-improvement disabled and asserted every run.
- RSI methods: `none` (zero state), `raw_memory` (BM25 top-4 + deterministic
  truncation), `ace_context` (official ACE Reflector/Curator — verified writing
  playbook bullets through DeepSeek), `worldmind` (official plugin modules +
  isolated prediction sidecar).
- Frozen-probe protocol unit-verified (`tests/test_probe_readonly.py`,
  `test_snapshot_reload.py`, `test_rsi_state_isolation.py`).
- Zero private leakage in adapter observations and planner payloads
  (`tests/test_no_private_leakage.py`, G3 leakage count = 0).

## 9. Next actions (in order)

1. Fix P1 (EB-Habitat physical-signature join) → re-run G3 until ≤ 2%.
2. Re-run G4 to PASS; then G5 (`scripts/07_smoke_rsi.py`).
3. Wire token aggregation (P5) into `episode_runner`/`metrics.json`.
4. Run Pilot-150 per method (`scripts/08_run_pilot150.py --method <m>`), then
   `scripts/09_analyze_pilot.py` and `scripts/10_audit_release.py`.
5. Optional: parameterize worker paths (P6) before scaling to Full Core v1.0.
