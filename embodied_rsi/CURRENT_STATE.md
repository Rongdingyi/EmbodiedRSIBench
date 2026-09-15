# Current State

## Last completed gate
G2 Model Gate (DeepSeek `deepseek-flash`)

## Status
G0/G1/G2 PASS. **G3 Adapter Gate FAIL** (16/100 crashes: EB-Habitat 14 due to
dataset index/id misalignment, TVRBench 2 transient simulator timeouts under GPU
load; leakage = 0). **G4 OpenETA Baseline Gate FAIL**: all 25 samples ran with 0 infrastructure
crashes and 0 native-SI writes, and the invalid-action rate was within budget,
but the RGB-delivery check failed at 19/25 (threshold 90%).
Per-source delivery: AsgardBench 5/5, EB-ALFRED 5/5, SpatialWorld 5/5,
TVRBench 4/5, EB-Habitat 0/5 — the miss is dominated by B3 (EB-Habitat episode
alignment); fixing B3 should bring the gate to 24/25 >= 90%.
Per the guide, G3+B3 and the G4 image check must be fixed before Phase F/G runs.

## Evidence
- **G0 Dataset Gate** — `outputs/preflight/DATASET_AUDIT.json` = PASS
  (raw 8046, Core 2200, roles 1500/300/300/100, role physical sets disjoint,
  source quotas 1408/410/104/137/141).
- **G1 OpenETA Freeze Gate** — `outputs/preflight/OPENETA_FREEZE.json` = PASS
  (HEAD == pinned `7d4a0a1522ba8ebbd362bde880bad81d2a98f15e`, clean worktree, no
  diff vs pin, 11 critical file hashes recorded, planner prompt hash recorded,
  native self-improvement disabled + auto_applier cleared by assertion).
  Manifest: `manifests/openeta_freeze_manifest.json`.
- **G2 Model Gate** — `outputs/api_smoke/deepseek_flash.json` = PASS
  (text `DEEPSEEK_FLASH_TEXT_OK`; vision described a real simulator frame;
  pinned OpenETA backend returned a valid `tool_call` decision; request audit:
  URL `https://api.deepseek.com/v1/chat/completions`, model `deepseek-flash`,
  image in the user message, **no `chat_template_kwargs`**).
- **External pins** — `manifests/external_sources.json`: OpenETA / ACE /
  WorldMind / EmbodiSkill all at the guide's commits, worktrees clean.
- **Adapters (Phase D)** — all five sources drive their official simulators
  through sandboxed worker processes (RoboTwin = AI2-THOR 5 CloudRendering;
  bench_avp = AI2-THOR 2.1.0 + Vulkan + physical X; embench = Habitat-Sim 0.3.0).
  Each adapter returns public observations (RGB + official feedback) and uses
  the source's own evaluator privately (TVR metrics, ALFRED goal satisfaction,
  EB-Habitat predicate success, SpatialWorld condition predicate, Asgard goals).
  `outputs/preflight/ADAPTER_VALIDATION.json` records the G3 result:
  100/100 tasks attempted, 84 clean, 16 crashes (see BLOCKERS.md B3).
- **OpenETA bridge (Phase E)** — `benchmark/openeta_bridge/` builds a pinned
  runtime per episode with a benchmark-only tool registry, the canonical planner
  prompt, DeepSeek backend (`enable_thinking=None`), per-episode fresh session,
  and RGB frames materialised into the planner's `image_artifacts` slot.
  Verified: planner emits `tool_call` actions that the host executes in the
  simulator (`RotateLeft`, ...), 0 infrastructure errors, 2 audited API calls.
- **RSI (Phase F)** — `none`, `raw_memory` (BM25 top-4, deterministic
  truncation), `ace_context` (official ACE Reflector/Curator + playbook ops;
  verified writing bullets via DeepSeek), `worldmind` (official plugin modules +
  independent prediction sidecar). `embodiskill` = BLOCKED (see `BLOCKERS.md`).
- **Tests** — `tests/test_openeta_frozen.py`, `test_no_private_leakage.py`,
  `test_probe_readonly.py`, `test_rsi_state_isolation.py`,
  `test_snapshot_reload.py`, `test_adapter_action_equivalence.py` all PASS.
- **Pilot-150 manifest** — `manifests/pilot150.json`,
  `outputs/preflight/PILOT150_AUDIT.json` = PASS (75/25/30/20, physical sets
  disjoint; stream order fixed by `sha256("20260915|experience_stream|" + id)`).

## Known issues
- EmbodiSkill integration blocked (ALFWorld/agentkit-specific) — audit in
  `outputs/preflight/EMBODISKILL_CALL_PATH.md`.
- EB-ALFRED rendering requires the machine's physical X display (`:1`);
  see `BLOCKERS.md` B2 and `scripts_render/00_pick_display.sh`.
- Exact per-episode token accounting is not yet wired to the rollout recorder;
  planner calls are counted per episode (`audit_requests`).
- `scripts/04_validate_adapters.py` and `scripts/06_smoke_openeta.py` were still
  running when this file was last updated; re-check their logs under
  `outputs/preflight/`.

## Next step
0. Fix B3 (EB-Habitat join key) and re-run `scripts/04_validate_adapters.py`
   until crash rate <= 2%; re-run G4 to completion.
2. Run G5: `python scripts/07_smoke_rsi.py` (~5 experience + 3 probes per method).
3. Run the pilot per method (order: none, raw_memory, ace_context, worldmind):
   `external/OpenETA/.venv/bin/python scripts/08_run_pilot150.py --method <m>`.
   Expect multi-hour wall time per method (simulator + DeepSeek latency).
4. `python scripts/09_analyze_pilot.py` then `python scripts/10_audit_release.py`
   for `PILOT_REPORT.md`, the result CSVs and FINAL_STATUS.
