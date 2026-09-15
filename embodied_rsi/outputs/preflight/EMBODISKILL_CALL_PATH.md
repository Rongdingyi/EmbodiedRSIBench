# EmbodiSkill Call-Path Audit (guide section 21.2)

Repo: `external/EmbodiSkill` @ `760126030eab1d33ec6a6f30988f0f1fb58df3a7`
Audited files: `tasks/run_epochs.py`, `agentkit/skill/common.py`,
`agentkit/skill/embodiskill_skill/{EmbodiSkill,prompt,skill_base}.py`.

## How the official pipeline works

1. **Task init** — `tasks/run_epochs.py` builds an MS-Agent `Task`/env through
   `agentkit` (`envs/`, `configs.yaml`); ALFWorld is the shipped environment.
2. **StateChain** — `common.py::MASSkillBase` owns `StateChain`, populated by the
   agentkit executor while it runs the episode (the executor *is* the trajectory
   owner). `EmbodiSkill` subclasses `MASSkillBase`.
3. **Trajectory → skill module** — the skill hooks (`build_instruction`,
   `get_system_prompt`, `save_reflection`, `update_manual_state`) are called *by
   the agentkit loop* at points that assume agentkit's own action/observation
   objects and its `StateChain` layout.
4. **Reflection trigger** — at epoch boundaries inside `run_epochs.py`
   (`_reflection_epoch_file`, `_load_reflections_for_epoch`); reflection records
   carry agentkit-specific fields (`step_id`, `task_id`, ALFWorld env ids).
5. **manual_state update** — `EmbodiSkill.update_manual_state` /
   `save_reflection` mutate `skill/current.json` + `skill/versions/`.
6. **Skill retrieval** — `EmbodiSkill.get_system_prompt` injects the current
   manual text into the agentkit prompt (its own privileged prompt channel).

## ALFWorld-specific assumptions

- env/action/observation objects and ids from `agentkit/envs/alfworld`;
- `StateChain` populated by the agentkit executor (not an external trajectory);
- epoch-labeled reflection files tied to `run_epochs.py`;
- manual/skill text injected through the agentkit system-prompt path.

## Update (review re-evaluation): thin adapter IS feasible

The official base class exposes the exact integration points needed:

```text
EmbodiSkill.init_task_context(task_main, task_description)
EmbodiSkill.move_skill_state(action, observation)          # per step
EmbodiSkill.save_task_context(label, feedback) -> MASMessage
EmbodiSkill.reflect_episode(mas_message)                   # S_NEW/S_BETTER/FAIL_SKILL/FAIL_EXECUTION
EmbodiSkill.revise_manual(epoch_id, success_rate)          # manual_state update
```

`StateChain` itself is just a NetworkX DiGraph container, so an external OpenETA
trajectory can be replayed into it without touching upstream code. The converter
is implemented in `benchmark/adapters/workers/embodiskill_worker.py` (<<200
lines) and runs in an isolated worker env because of the dependency weight
(`langchain-chroma`, `sentence-transformers`, `finch`). Status is therefore
**POC_READY_UNVERIFIED**, not a permanent BLOCK: C4 stays out of the pilot table
until the PoC is executed and the four official reflection categories appear.

## Original assessment (superseded)

To feed an external OpenETA trajectory we would have to (a) rebuild
`StateChain` with a foreign schema, (b) bypass the agentkit loop that triggers
reflection, and (c) re-route the manual injection through our shared
`context_text` channel. That rewrites reflection/update semantics and exceeds
the guide's ~200-line integration limit, so per sections 2.5 / 21.7 the method
is recorded as **BLOCKED** (no pseudo-EmbodiSkill is shipped).
