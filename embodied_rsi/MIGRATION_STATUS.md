# Migration Status

Canonical Multimodal ReAct Agent migration (taskbook v0.8), branch
`refactor/canonical-react-agent`.

## Current commit

M8 commit (this commit) on `refactor/canonical-react-agent`.

Base: `a238c8b` (current `main`). The taskbook names `56ad3aa` as the migration
base; this branch was cut from `main` **including** the four post-base fixes
that the canonical track depends on (WorldMind goal-extraction signature,
Asgard missing-parameter handling, infra-failed-episode learning skip,
TVR target-image delivery). Deviation is recorded here deliberately; no other
lineage change was made.

## Completed phases

- [x] M1 `32f0bf2` chore: pin EmbodiedBench canonical-agent reference
- [x] M2 `12758a6` refactor: decouple RSI context types from OpenETA
- [x] M3 `e8a1fdc` feat: canonical multimodal ReAct agent core
- [x] M4 `ad3d61f` feat: canonical one-action episode runner
- [x] M5 `f545797` refactor: attach none raw-memory ACE WorldMind to canonical agent
- [x] M6 `7457e3f` chore: harden EmbodiSkill adapter and keep C4 blocked
- [x] M7 `33b885c` feat: canonical agent configs gates and Pilot-60 runner
- [x] M8 (this commit) test: canonical agent deterministic acceptance suite

## Tests run

```bash
external/OpenETA/.venv/bin/python -m pytest tests/ -q
# 33 passed (11 warnings), includes:
#   - 11 new canonical-agent suites (20 test cases)
#   - legacy adapter / snapshot / read-only / leakage suites
external/OpenETA/.venv/bin/python scripts/03b_verify_canonical_agent.py
# CANONICAL AGENT FREEZE: PASS
```

Appendix B greps (all clean):

```text
benchmark/rsi   -> no benchmark.openeta_bridge imports
benchmark/agent -> no OpenETA / agent.runtime imports
benchmark/agent -> no random fallback (random.choice / np.random)
benchmark/agent -> no tool_batch / executable_plan multi-action execution
```

## Phase highlights

- Phase A: `benchmark/rsi/context.py` is the canonical injection channel
  (tiktoken cl100k or char/4 estimate, honestly recorded); the historical
  `context_injection.py` is a compatibility shim re-exporting it.
- Phase I/J: `decide()` never touches the environment; the runner enforces
  `1 planner decision = 1 adapter.step()`; success comes only from
  `adapter.private_evaluate()`.
- RSI hook timing: sidecar runs after action validation and before `step()`;
  `after_step` sees the real post-action state; infra-failed experience
  episodes are not learning signals and are skipped.
- WorldMind official modules unchanged (process/goal/retrieval); ACE official
  reflector/curator unchanged.
- Stop reasons are structured: `planner_protocol_failure`,
  `simulator_infrastructure_error`, `max_agent_actions`, `source_terminated`,
  `source_truncated`, `episode_timeout` (`rsi_update_error` recorded separately).
- Accounting adds `planner_validation_retries`, `planner_invalid_json_count`,
  `planner_invalid_action_count`; the runner wires injected agents/backends
  into its accountant so counters cannot disappear.

## Known blockers

- C4 `embodiskill` stays **BLOCKED / POC_READY_UNVERIFIED**: the worker was
  hardened (path derivation, canonical namespace, per-episode usage +
  request records) but the dependency environment (langchain-chroma +
  sentence-transformers + finch) is not provisioned, so the PoC could not run.
  Per the taskbook, `BLOCKED=False` is NOT set.
- `06b` canonical API smoke and the 10-episode `none` capability calibration
  are **paid** and were NOT run (taskbook §45: stop at M8 and wait for human
  confirmation).

## Files changed

```text
benchmark/agent/{__init__,types,config,prompt,vision,backend,action_schema,
                 working_memory,react_agent,episode_runner,freeze}.py
benchmark/rsi/context.py + all benchmark/rsi/*.py imports
benchmark/openeta_bridge/{context_injection.py shim, LEGACY.md}
benchmark/adapters/workers/embodiskill_worker.py (path/namespace/usage)
benchmark/rsi/embodiskill.py (worker client; still blocked)
benchmark/runner/accounting.py (three planner counters)
configs/agent/canonical_react.yaml, configs/pilot60_canonical.yaml
manifests/{canonical_agent_reference.json, external_sources.json}
scripts/{03b,06b,07b,08b,10b}_*.py, scripts/09_analyze_pilot.py (--root/--protocol)
tests/canonical_fakes.py + 11 new test files
```

## Expensive actions NOT run

- no real API calls in any new test;
- no canonical API smoke (06b);
- no 10-episode `none` capability calibration;
- no canonical Pilot-60 run.

## Next (requires human confirmation)

1. Run `scripts/06b_smoke_canonical_agent.py` (one episode, paid).
2. Run the 10-episode `none` capability calibration (paid, the only bring-up);
   if overall success <= 10%, STOP and diagnose model/agent/budget first.
3. Only then decide on the canonical Pilot-60 run.
