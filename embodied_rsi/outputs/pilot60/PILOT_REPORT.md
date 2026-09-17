# Pilot-60 Report

| Method | Init ID | Final ID | ID Gain | Init Transfer | Final Transfer | Transfer Gain | Retention | Probe state OK |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ace_context | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True |
| none | 0.100 | 0.000 | -0.100 | 0.000 | 0.000 | 0.000 | 0.000 | True |
| raw_memory | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.100 | True |
| worldmind | 0.000 | 0.100 | 0.100 | 0.000 | 0.000 | 0.000 | 0.100 | True |

## Paired transitions (S000 -> Sfinal)

Same frozen tasks at both checkpoints; a task counts only when both runs produced a scored outcome.

| Method | Role | n | fail->success | success->fail | success->success | fail->fail | Net gain |
|---|---|---:|---:|---:|---:|---:|---:|
| ace_context | id | 10 | 0 | 0 | 0 | 10 | +0 |
| ace_context | transfer | 10 | 0 | 0 | 0 | 10 | +0 |
| ace_context | retention | 10 | 0 | 1 | 0 | 9 | -1 |
| none | id | 10 | 0 | 1 | 0 | 9 | -1 |
| none | transfer | 10 | 0 | 0 | 0 | 10 | +0 |
| none | retention | 10 | 0 | 1 | 0 | 9 | -1 |
| raw_memory | id | 10 | 0 | 0 | 0 | 10 | +0 |
| raw_memory | transfer | 10 | 0 | 0 | 0 | 10 | +0 |
| raw_memory | retention | 10 | 0 | 1 | 1 | 8 | -1 |
| worldmind | id | 10 | 1 | 0 | 0 | 9 | +1 |
| worldmind | transfer | 10 | 0 | 0 | 0 | 10 | +0 |
| worldmind | retention | 10 | 1 | 0 | 0 | 9 | +1 |

## Excluded conditions
- `embodiskill`: BLOCKED (see BLOCKERS.md / EMBODISKILL_CALL_PATH.md).

## Notes
- Success rates use the official per-source private evaluators.
- Probe checkpoints re-run the same frozen tasks with updates disabled (Pilot-60: S000 and S030).
- Token accounting: planner calls are counted per episode; exact token totals are emitted by the rollout recorder when enabled.
