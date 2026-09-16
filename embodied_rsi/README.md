# embodied_rsi — OpenETA + RSI Pilot (Pilot v0.3, DeepSeek-Flash)

Implementation of `../Embodied_RSI_OpenETA_Pilot_Execution_Guide_DeepSeekFlash.md`:
a frozen public embodied agent (OpenETA Stage-2) driven over the Embodied RSI
Benchmark Core v1.0 with swappable self-evolution (RSI) methods, frozen probes
and full leakage/freeze auditing.

## Fixed factors / variable factor

| Fixed | Value |
|---|---|
| Tasks | Core v1.0 (2200 physical tasks; Pilot-60 subset) |
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
├── configs/{agent,model,rsi,pilot60.yaml}
├── manifests/                external_sources.json, openeta_freeze_manifest.json,
│                             pilot60.json
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
external/OpenETA/.venv/bin/python scripts/05b_build_pilot60.py          # manifest
external/OpenETA/.venv/bin/python scripts/08_run_pilot60.py --method none
external/OpenETA/.venv/bin/python scripts/09_analyze_pilot.py
python scripts/10_audit_release.py
```

## Current status (v0.7 Pilot-60, 2026-09-16)

- G0-G6 **PASS** on the relay endpoint (see `CURRENT_STATE.md`).
- Pilot protocol is now **Pilot-60** (v0.7): 30 experience + 10 ID + 10 transfer
  + 10 retention, checkpoints S000/S030, 90 episodes per method, 360 total.
- C4 `embodiskill`: thin adapter implemented, **POC_READY_UNVERIFIED** (B1).
- G7 Pilot-60 (four primary conditions) not yet run.

See `../PROBLEMS.md` (problems/next actions) and `CURRENT_STATE.md` (evidence).
