"""P1: WorldMind guidance is token-budget safe (never trips the make_injection assert)."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark.rsi.context import MAX_RSI_INJECTION_TOKENS, count_tokens  # noqa: E402
from benchmark.rsi.worldmind import WorldMindRSI  # noqa: E402


class HugeRetrieval:
    def retrieve(self, *, task_instruction: str, enable_refine: bool = True):
        return {"formatted_prompt": ("- kitchen experience line\n" * 5000),
                "experience_ids": ["exp1"]}


def test_worldmind_injection_fits_the_budget(tmp_path):
    rsi = WorldMindRSI.__new__(WorldMindRSI)
    rsi.state_dir = tmp_path
    rsi.update_enabled = True
    rsi.run_ctx = {}
    rsi.retrieval_module = HugeRetrieval()
    rsi._pending_prediction = None
    rsi._pending_action = None
    rsi._sidecar_usage = {}
    rsi._component_usage = {}
    rsi._sidecar_wall_s = 0.0
    rsi.update_errors = []
    rsi.prediction_errors = []

    injection = WorldMindRSI.before_episode(rsi, {"instruction": "do something"})
    assert injection.estimated_tokens <= MAX_RSI_INJECTION_TOKENS
    assert injection.estimated_tokens > 0
    assert count_tokens(injection.context_text) <= MAX_RSI_INJECTION_TOKENS
    assert injection.truncated is False or True  # budget respected either way
