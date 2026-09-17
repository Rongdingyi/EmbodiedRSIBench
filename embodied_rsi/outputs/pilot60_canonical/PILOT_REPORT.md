# Pilot-60 Report

| Method | Init ID | Final ID | ID Gain | Init Transfer | Final Transfer | Transfer Gain | Retention | Probe state OK |
|---|---:|---:|---:|---:|---:|---:|---:|---|

## Paired transitions (S000 -> Sfinal)

Same frozen tasks at both checkpoints; a task counts only when both runs produced a scored outcome.

| Method | Role | n | fail->success | success->fail | success->success | fail->fail | Net gain |
|---|---|---:|---:|---:|---:|---:|---:|

## Excluded conditions
- `embodiskill`: BLOCKED (see BLOCKERS.md / EMBODISKILL_CALL_PATH.md).

## Notes
- Success rates use the official per-source private evaluators.
- Probe checkpoints re-run the same frozen tasks with updates disabled (Pilot-60: S000 and S030).
- Token accounting: planner calls are counted per episode; exact token totals are emitted by the rollout recorder when enabled.
