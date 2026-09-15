# BLOCKERS

## B1 — EmbodiSkill integration (Condition E)

Status: **BLOCKED** for Pilot v0.3 (guide sections 2.5 / 21.7).

The official EmbodiSkill implementation is ALFWorld/agentkit-specific: its
`StateChain`, reflection epoch files and manual-injection path are owned by the
agentkit executor. Feeding an external OpenETA trajectory requires rewriting
reflection/update semantics (>200 lines of core changes). Audit:
`outputs/preflight/EMBODISKILL_CALL_PATH.md`.

Consequence: Pilot v0.3 runs conditions C0 (`none`), C1 (`raw_memory`),
C2 (`ace_context`), C3 (`worldmind`); C4 remains BLOCKED and is excluded from
the primary table until a canonical integration is available.

## B2 — EB-ALFRED rendering depends on the machine's physical X display

The 2018 AI2-THOR build presents through NVIDIA Vulkan, which cannot create a
swapchain for a private Xvfb. EB-ALFRED episodes therefore require a real X
server (this machine: `:1`). Mitigation: `scripts_render/00_pick_display.sh`;
failures are reported as infrastructure errors, never as method failures.

## B3 — EB-Habitat dataset index/id alignment (G3 FAIL root cause)

Status: **OPEN** (found by G3 adapter validation: 14/20 EB-Habitat samples crashed).

`EBHabEnv(eval_set=...)` builds its own dataset ordering and its own
`episode_id` values: neither the release's `source_entry_index` (pickle order)
nor the release `source_task_id` (pickle episode_id) resolves to the right
episode — e.g. instruction-matched episode 35 loaded where the release expects
16. Impact: EB-Habitat task selection must join on the physical signature
(scene_id + sampled_entities + start pose) that the release already records.

Fix plan (next session):
1. In `eb_habitat_worker.reset`, build a lookup from
   `(scene_id basename, canonical_json(sampled_entities))` to dataset index and
   use it as the primary key (fall back to instruction only when unique).
2. Re-run `scripts/04_validate_adapters.py`; target crash rate <= 2%.

Second G3 contributor: 2/20 TVRBench samples hit transient AI2-THOR backend
timeouts while all four GPUs were heavily loaded (external processes). The
validator now retries transient timeouts once; keep GPU contention in mind when
running gates.
