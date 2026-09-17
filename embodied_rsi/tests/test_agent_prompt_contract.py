"""Test 4: the prompt skeleton is identical for every RSI method (one shared slot)."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.agent.prompt import NONE_GUIDANCE, build_turn_prompt  # noqa: E402
from benchmark.rsi.context import OPEN_TAG, CLOSE_TAG, make_injection  # noqa: E402

SLOT_ORDER = ["## Task", "## Persistent cross-episode guidance",
              "## Current public environment feedback", "## Current-episode action history",
              "## Legal actions", "## Optional recovery note"]


def _prompt(injection):
    return build_turn_prompt(instruction="do it", tool_specs=[{"name": "A", "parameters": {}}],
                             working_memory_text="", public_feedback="",
                             rsi_injection=injection)


def test_none_and_raw_memory_share_the_skeleton():
    none_prompt = _prompt(make_injection(""))
    other_prompt = _prompt(make_injection("remember: rotate before moving"))
    for section in SLOT_ORDER:
        assert section in none_prompt and section in other_prompt
    pos_none = [none_prompt.index(s) for s in SLOT_ORDER]
    pos_other = [other_prompt.index(s) for s in SLOT_ORDER]
    assert pos_none == sorted(pos_none) and pos_other == sorted(pos_other)
    assert f"{OPEN_TAG}\n{NONE_GUIDANCE}\n{CLOSE_TAG}" in none_prompt
    assert "remember: rotate before moving" in other_prompt
    # skeleton outside the slot is identical
    left = none_prompt.split(OPEN_TAG)[0]
    right = other_prompt.split(OPEN_TAG)[0]
    assert left == right


def test_none_keeps_the_section():
    prompt = _prompt(make_injection(""))
    assert OPEN_TAG in prompt and CLOSE_TAG in prompt
    assert NONE_GUIDANCE in prompt
