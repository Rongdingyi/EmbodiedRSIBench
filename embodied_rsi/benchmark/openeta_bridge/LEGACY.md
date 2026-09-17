# LEGACY: OpenETA substrate (historical track only)

This package reproduces the historical OpenETA substrate used by Pilot-60 v0.7
and its gate evidence. It is **no longer the canonical agent** for the main RSI
benchmark after protocol v0.8.

Rules:
- Do not import this package from `benchmark/agent/` (canonical ReAct agent) or
  from canonical script paths.
- Do not delete or rename it: historical results and provenance depend on it.
- `context_injection.py` is kept as a compatibility shim that re-exports
  `benchmark.rsi.context` so the legacy track stays reproducible.

Historical result framing (for any writeup):
> Historical Substrate Study: OpenETA Stage-2 adaptation. The OpenETA-based
> substrate produced severe floor effects under the benchmark-native
> source-action adaptation, motivating the canonical planner-executor redesign.
Do not mix historical OpenETA rows with canonical-agent rows in one leaderboard.
