# Migration Status

Canonical Multimodal ReAct Agent migration (taskbook v0.8), branch
`refactor/canonical-react-agent`.

## Current commit

M9 review-fix commit (this commit) on `refactor/canonical-react-agent`
(head before this commit: `594de82`).

### Provenance (three independent fields)

```text
branch fork base / origin main at migration start: 56ad3aa
    (origin/main is STILL 56ad3aa; the historical track was not merged)
pre-M1 fixes included from local main:
    3de6f7e  WorldMind goal-extraction call signature
    7bfe78c  Asgard worker rejects missing-parameter actions as failed steps
    f09a444  infra-failed episodes are not learning signals + 0-step retry
    a238c8b  TVR target image delivered to the planner (guide 11.1)
canonical branch head: this commit (M9)
```

## Completed phases

- [x] M1 `32f0bf2` chore: pin EmbodiedBench canonical-agent reference
- [x] M2 `12758a6` refactor: decouple RSI context types from OpenETA
- [x] M3 `e8a1fdc` feat: canonical multimodal ReAct agent core
- [x] M4 `ad3d61f` feat: canonical one-action episode runner
- [x] M5 `f545797` refactor: attach none raw-memory ACE WorldMind to canonical agent
- [x] M6 `7457e3f` chore: harden EmbodiSkill adapter and keep C4 blocked
- [x] M7 `33b885c` feat: canonical agent configs gates and Pilot-60 runner
- [x] M8 (this commit) test: canonical agent deterministic acceptance suite

## Review v0.9 fixes (M9)

| Item | Fix |
|---|---|
| P0 RSI transaction | experience/probes take a pre-episode snapshot; any `after_step`/`after_episode`/sidecar exception stops learning immediately, restores the snapshot, sets `stop_reason=rsi_update_error` and never calls `after_episode` afterwards |
| P0 runner lifecycle | whole adapter lifecycle guarded; `adapter.close()` always runs; reset/tool-spec failures return a canonical result with `simulator_infrastructure_error`; no evaluator call when reset failed |
| P0 delete-existing-run | `08b --delete-existing-run` really `rmtree`s the previous root; `--overwrite` is a deprecated alias; non-empty roots abort otherwise |
| P0 TVR roles | fixed `## Visual inputs` prompt section (always present, identical for all methods) lists allowlisted `image_roles` only |
| P1 WorldMind budget | `truncate_to_token_budget()` (binary search over the same counter) used by WorldMind and ACE; the budget assert can no longer fire |
| P1 WorldMind state | `_state_text()` reads allowlisted fields from `public_metadata` (image_roles now actually visible) |
| P1 analysis | success-rate/paired-transition denominators require `status == PASS`; `PILOT_EXCLUDED.csv` + report section list protocol/infra failures |
| P1 release audit | infra crash rates reported with two honest denominators: per experience episode and per executed attempt (includes probe retries) |
| P1 backend | `DEEPSEEK_BASE_URL` falls back to the official host; missing `DEEPSEEK_API_KEY` / unsupported `json_mode` fail fast at startup |
| P1 frozen config | every YAML switch is now consumed or rejected (`validate_agent_config`); no fake knobs remain |
| P2 freeze | artifact carries `static_checks_status` + deterministic `invariant_tests` (hashes + PASS/FAIL); overall PASS requires both |
| P2 provenance | this section (three fields) |
| P2 resolved config | `config_resolved.json` merges protocol + agent + CLI + resolved budgets + hashes |

## Tests run

```bash
external/OpenETA/.venv/bin/python -m pytest tests/ -q
# 48 passed, includes:
#   - 19 canonical suites (new: transaction rollback, reset failure,
#     always-closed, delete-existing-run, TVR image roles, WorldMind budget,
#     analysis exclusion, release infra denominators)
#   - legacy adapter / snapshot / read-only / leakage suites
external/OpenETA/.venv/bin/python scripts/03b_verify_canonical_agent.py
# CANONICAL AGENT FREEZE: PASS (static checks + invariant tests)
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
