# Current State

## Last completed milestone
Post-review refactor (v0.4): canonical OpenETA episode runner, WorldMind timing,
snapshot probe semantics, gate vocabulary, adapters, pilot manifest.

## Status
G0/G1/G2 PASS. G3/G4/G5 must be RE-RUN with the refactored stack (the previous
G3/G4 failures predate the fixes). G7 (Pilot-150) is still gated behind those.

## Evidence
- **G0 Dataset** `outputs/preflight/DATASET_AUDIT.json` = PASS
- **G1 Freeze** `outputs/preflight/OPENETA_FREEZE.json` = PASS;
  `manifests/openeta_freeze_manifest.json`
- **G2 Model** `outputs/api_smoke/deepseek_flash.json` = PASS
- **Upstream runner** verified on a real TVR episode: status=PASS, 3 turns,
  3 env steps, upstream `stop_reason=max_turns`, `runner_tokens=29922`,
  planner accounting populated (40.5k in / 45.2k out over 7 calls), leakage=0.
- **Pilot-150 manifest** `manifests/pilot150.json` rebuilt:
  experience TVR 23 / Spatial 13 / Asgard 13 / ALFRED 13 / Habitat 13;
  ID 25 / Transfer 30 / Retention 20; physical sets disjoint.
- **Tests** all PASS: freeze, gate status, snapshot reload (new-instance clone),
  probe readonly, RSI isolation, observation allowlist, adapter equivalence.

## Key changes (review P0/P1)
1. `benchmark/openeta_bridge/benchmark_environment.py` implements upstream
   `EpisodeEnvironment`; `episode_runner.py` delegates the loop to
   `OpenEtaEpisodeRunner`. Planner/prompts/pipeline remain upstream-unmodified.
2. WorldMind sidecar runs after action lock, before `env.step`; process update
   consumes the real feedback; sidecar never re-enters the planner.
3. Two-phase environment preparation gives a live action schema before runtime
   assembly (fixes EB-Habitat empty registry; worker fails loud on schema drift).
4. `clone_from_snapshot` = new instance + temp copy + `update_enabled=False`.
5. Gates are PASS/FAIL/BLOCKED; BLOCKED is never coerced to PASS; release status
   is `FULL_PILOT_PASS` / `PIPELINE_PASS_WITH_BLOCKER` / `FAIL`.
6. Retention anchors have no oracle pairing; pilot experience is deficit-filled
   toward 15/source.
7. SpatialWorld exposes the source abstraction `Move/Rotate/Tilt/ChangePosture/
   Pick/Place/ChangeState/Manipulate/EndTask`.
8. Strict observation allowlist + redacted `public_context_dump.jsonl` per
   episode with leakage scan.
9. Central accounting (`benchmark/runner/accounting.py`) → `token_usage.json`,
   aggregated in `metrics.json` (planner/injection/updater/sidecar/env steps).
10. EmbodiSkill: thin adapter implemented
    (`benchmark/adapters/workers/embodiskill_worker.py`, official API path);
    status POC_READY_UNVERIFIED, excluded from the pilot table until executed.

## Known issues
- G3/G4/G5 need a full re-run (hours); EB-Habitat episode lookup is the known
  risk (see `BLOCKERS.md` B3: instruction/id join, physical-signature fallback
  documented).
- EmbodiSkill PoC needs its dependency environment (langchain-chroma /
  sentence-transformers / finch).
- Worker paths and data root are still machine-specific.

## Next step
1. `external/OpenETA/.venv/bin/python scripts/04_validate_adapters.py`   (G3)
2. `external/OpenETA/.venv/bin/python scripts/06_smoke_openeta.py`       (G4)
3. `external/OpenETA/.venv/bin/python scripts/07_smoke_rsi.py`           (G5)
4. `external/OpenETA/.venv/bin/python scripts/08_run_pilot150.py --method none` etc.
5. `scripts/09_analyze_pilot.py`, `scripts/10_audit_release.py`
