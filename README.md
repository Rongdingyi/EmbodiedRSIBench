# EmbodiedRSIBench

Research workspace for **Embodied Agent RSI / Experience Learning** benchmarks:
a normalized, leakage-audited task corpus (Core v1.0) plus a pilot harness that
runs a frozen public embodied agent (OpenETA) with swappable self-evolution
methods over real simulators.

## Contents

| Path | What it is |
|---|---|
| `Embodied_RSI_Benchmark_Data_Preparation_Guide.md` | Task book 1: dataset download / normalization / dedup / pairing / validation spec |
| `Embodied_RSI_OpenETA_Pilot_Execution_Guide_DeepSeekFlash.md` | Task book 2: OpenETA + RSI pilot protocol (frozen agent, DeepSeek-Flash) |
| `embodied_rsi/` | The pilot project: adapters, OpenETA bridge, RSI methods, gates, tests |
| `scripts_render/` | Verified rendering recipes for the five source simulators |
| `PROBLEMS.md` | **Current status, problems, root causes and fix plans** |
| `data` (symlink, not committed) | Core v1.0 release artifacts (registry + splits + pairs) |

## Core v1.0 (built per task book 1)

- **8,046 raw entries** → **2,200 mutually-exclusive physical tasks**
  (Experience 1500 / ID 300 / Transfer 300 / Retention 100),
  300 ID pairs, 300 Transfer pairs, 100 retention anchors, 12 skill families,
  13 canonical primitive classes.
- Sources: EB-ALFRED, EB-Habitat, SpatialWorld (single-agent), AsgardBench,
  TVRBench — pinned upstream commits, leakage checks passing
  (`embodied_rsi_data/release/VALIDATION_REPORT.md` in the data workspace).

## Pilot project (built per task book 2)

- OpenETA Stage-2 pinned at `7d4a0a1522ba8ebbd362bde880bad81d2a98f15e`,
  frozen and audited (critical-file hashes, planner-prompt hash, native
  self-improvement hard-disabled, benchmark-only tool registry).
- Five simulator adapters (AI2-THOR 5 / AI2-THOR 2.1.0 / Habitat-Sim 0.3.0)
  behind a subprocess JSON protocol; private evaluators stay server-side.
- RSI methods: `none`, `raw_memory`, `ace_context` (official ACE
  Reflector/Curator), `worldmind` (official WorldMind plugin), `embodiskill`
  (**BLOCKED** — audit included, no pseudo-implementation shipped).

## Status

G0/G1/G2 gates **PASS**; G3/G4 currently **FAIL** with a single dominant root
cause (EB-Habitat episode alignment) plus one transient GPU-contention issue.
Full write-up, evidence and next actions: **`PROBLEMS.md`**, with per-phase
details in `embodied_rsi/CURRENT_STATE.md` and `embodied_rsi/BLOCKERS.md`.

## Quick start

```bash
# 1) data: point to the Core v1.0 release (parquet/jsonl, ~2 MB; sources ~2 GB)
export EMBODIED_RSI_DATA=/path/to/embodied_rsi_data/release

# 2) external code at pinned commits
bash embodied_rsi/scripts/01_clone_pin_external.sh
cd embodied_rsi/external/OpenETA && uv sync --extra dev && cd -

# 3) secrets (never committed)
cp embodied_rsi/.env.example embodied_rsi/.env   # then fill DEEPSEEK_API_KEY

# 4) gates, in order (never skip)
python embodied_rsi/scripts/00_verify_dataset.py
embodied_rsi/external/OpenETA/.venv/bin/python embodied_rsi/scripts/03_verify_openeta_freeze.py
bash embodied_rsi/scripts/02_check_deepseek_api.sh
# ... see embodied_rsi/README.md for the full list
```

Simulator environments (`RoboTwin`, `bench_avp`, `embench`) and their packages
are documented in `scripts_render/README.md` and `embodied_rsi/README.md`.

## Notes

- Upstream method repositories, model weights and datasets are **not vendored**;
  exact commits are pinned in `embodied_rsi/manifests/external_sources.json`.
- Secrets: only `embodied_rsi/.env` (git-ignored) holds the DeepSeek key.
