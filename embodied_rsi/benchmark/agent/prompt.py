"""Fixed canonical agent prompt skeleton (taskbook v0.8, Phase D).

The system prompt is a frozen code constant. The turn prompt has fixed slots in
a fixed order; the only method-dependent content is the RSI guidance slot.
No method-specific prompt mutation is allowed anywhere.
"""
from __future__ import annotations

import json

from benchmark.rsi.context import CLOSE_TAG, OPEN_TAG, RSIInjection

CANONICAL_SYSTEM_PROMPT = r"""
You are the planner of a closed-loop embodied agent.

Your job is to complete the user's physical task by repeatedly observing the
current environment, reasoning about the next useful step, and selecting exactly
ONE action from the provided legal action set.

Hard rules:
1. Select exactly one environment action per turn.
2. Use only action names and parameters that appear in the legal action schema.
3. Never invent hidden object ids, simulator state, success predicates, or goal state.
4. Use the current visual observation, public environment feedback, and current-episode
   action history to recover from failed actions and replan.
5. Persistent cross-episode guidance is advisory experience, not ground truth.
6. Do not claim benchmark success yourself. The environment decides termination and the
   private evaluator decides task success.
7. If the previous action failed, use the public feedback to choose a different useful
   next action when appropriate.
8. Return JSON only. Do not return markdown fences.

Required JSON schema:
{
  "thought": "brief task-relevant reasoning for the next action",
  "action": {
    "name": "one legal action name",
    "parameters": {}
  }
}
""".strip()

NONE_GUIDANCE = "(none)"
NO_IMAGES_TEXT = "(no image inputs)"


def render_visual_inputs(roles: list[str] | None) -> str:
    """Roles come from the source's allowlisted `image_roles` only."""
    cleaned = [str(role) for role in (roles or []) if str(role).strip()]
    if not cleaned:
        return NO_IMAGES_TEXT
    return "\n".join(f"Image {index}: {role}" for index, role in enumerate(cleaned, 1))


def render_rsi_slot(injection: RSIInjection | None) -> str:
    """One shared slot for every method; `none` keeps the section with (none)."""
    text = (injection.context_text if injection is not None else "").strip()
    if not text:
        return f"{OPEN_TAG}\n{NONE_GUIDANCE}\n{CLOSE_TAG}"
    return text


def build_turn_prompt(
    *,
    instruction: str,
    tool_specs: list[dict],
    working_memory_text: str,
    public_feedback: str,
    rsi_injection: RSIInjection | None,
    stuck_warning: str = "",
    visual_inputs_text: str = "",
) -> str:
    """Fixed-order user prompt; identical skeleton for every RSI method."""
    schema = json.dumps(tool_specs, ensure_ascii=False, indent=2)
    return "\n".join([
        "## Task",
        instruction or "",
        "",
        "## Visual inputs",
        visual_inputs_text or NO_IMAGES_TEXT,
        "",
        "## Persistent cross-episode guidance",
        render_rsi_slot(rsi_injection),
        "",
        "## Current public environment feedback",
        public_feedback or "(none)",
        "",
        "## Current-episode action history",
        working_memory_text or "(no actions yet)",
        "",
        "## Legal actions",
        schema,
        "",
        "## Optional recovery note",
        stuck_warning or "(none)",
        "",
        "Select exactly one next action and return JSON only.",
    ])


def build_invalid_response_note(errors: list[str]) -> str:
    bullet = "\n".join(f"- {e}" for e in errors[-4:])
    return ("Your previous response was invalid for the following protocol reason:\n"
            f"{bullet}\n"
            "Return a corrected JSON object with exactly one legal action.")
