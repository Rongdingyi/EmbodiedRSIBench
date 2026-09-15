# embodied_rsi — OpenETA + RSI Pilot (Pilot v0.3, DeepSeek-Flash)

Implementation of `../Embodied_RSI_OpenETA_Pilot_Execution_Guide_DeepSeekFlash.md`:
a frozen public embodied agent (OpenETA Stage-2) driven over the Embodied RSI
Benchmark Core v1.0 with swappable self-evolution (RSI) methods, frozen probes
and full leakage/freeze auditing.

## Fixed factors / variable factor

| Fixed | Value |
|---|---|
| Tasks | Core v1.0 (2200 physical tasks; Pilot-150 subset) |
| Agent | OpenETA Stage-2 @ `7d4a0a1522ba8ebbd362bde880bad81d2a98f15e` (unmodified) |
| Model | DeepSeek official API `deepseek-flash` (planner + all updaters) |
| Simulators | each source's own runtime (AI2-THOR 5, AI2-THOR 2.1.0, Habitat-Sim 0.3.0) |
| Probes | frozen (`UpdateEnabled=false`, state hash asserted unchanged) |
| **Variable** | only how experience becomes persistent state (`benchmark/rsi/`) |

## Layout

```
embodied_rsi/
├── external/                 cloned upstreams at pinned commits (not committed)
├── benchmark/
│   ├── registry/loader.py    joins role parquet + JSONL private metadata
│   ├── adapters/             host adapters + workers/ (one per source, own env)
│   ├── openeta_bridge/       freeze, build_runtime, context_injection,
│   │                         episode_runner, api_smoke
│   └── rsi/                  none / raw_memory / ace_context / worldmind / embodiskill
├── configs/{agent,model,rsi,pilot150.yaml}
├── manifests/                external_sources.json, openeta_freeze_manifest.json,
│                             pilot150.json
├── scripts/00..10            the guide's execution order (gate by gate)
├── tests/                    freeze / leakage / probe-readonly / isolation /
│                             snapshot-reload / action-equivalence
├── outputs/                  gate evidence (JSON + logs + API smoke frame)
├── CURRENT_STATE.md          phase-by-phase status (authoritative)
└── BLOCKERS.md               B1/B2/B3 — open problems and fix plans
```

Guide-expected modules `openeta_bridge/tool_registry.py` and `runner/*` are
implemented inside `openeta_bridge/build_runtime.py` and `scripts/04..10`
respectively (same responsibilities, fewer indirections).

## Environments (reused, simulator-isolated)

| Role | Env | What it runs |
|---|---|---|
| OpenETA runtime / RSI updaters | `external/OpenETA/.venv` (uv) | pinned agent + this repo's bridge (pyarrow, openai added) |
| EB-ALFRED worker | `bench_avp` | AI2-THOR 2.1.0 (`-force-vulkan`, physical X) + EmbodiedBench EBAlfEnv |
| EB-Habitat worker | `embench` (py3.9) | Habitat-Sim 0.3.0 + habitat-lab 0.3.0 + ReplicaCAD assets |
| SpatialWorld / AsgardBench / TVRBench workers | `RoboTwin` | AI2-THOR 5 CloudRendering |

Every worker runs in its own subprocess and speaks newline-JSON with base64 PNG
frames; the host never imports simulator code.

## Run order (never skip a gate)

```bash
export DEEPSEEK_API_KEY=...            # also see .env.example
python scripts/00_verify_dataset.py                                   # G0
bash   scripts/01_clone_pin_external.sh
external/OpenETA/.venv/bin/python scripts/03_verify_openeta_freeze.py  # G1
bash   scripts/02_check_deepseek_api.sh                               # G2
external/OpenETA/.venv/bin/python scripts/04_validate_adapters.py      # G3 (see PROBLEMS)
external/OpenETA/.venv/bin/python scripts/06_smoke_openeta.py          # G4 (see PROBLEMS)
external/OpenETA/.venv/bin/python scripts/07_smoke_rsi.py              # G5
external/OpenETA/.venv/bin/python scripts/05_build_pilot150.py         # manifest
external/OpenETA/.venv/bin/python scripts/08_run_pilot150.py --method none
external/OpenETA/.venv/bin/python scripts/09_analyze_pilot.py
python scripts/10_audit_release.py
```

## Current status (v0.4, 2026-09-15)

- G0/G1/G2 **PASS**.
- Episode loop now runs on the pinned **upstream `OpenEtaEpisodeRunner`**
  (environment protocol implementation in `benchmark/openeta_bridge/`), verified
  on a real TVR episode (PASS, upstream budgets/receipts, leakage 0).
- G3/G4/G5 must be **re-run** with the refactored stack (previous failures
  predate the fixes); see `../PROBLEMS.md` and `BLOCKERS.md` B3.
- C4 `embodiskill`: thin adapter implemented, **POC_READY_UNVERIFIED** (B1).
- Pilot-150 manifest rebuilt and balanced (TVR 23 / others 13); not yet run.

See `../PROBLEMS.md` (problems/next actions) and `CURRENT_STATE.md` (evidence).
