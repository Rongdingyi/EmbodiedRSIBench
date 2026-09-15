"""Gate: nothing crosses episodes except RSIInjection (fresh session rule)."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "external" / "OpenETA"))

from benchmark.rsi.none import NoneRSI  # noqa: E402
from benchmark.rsi.raw_memory import RawMemoryRSI  # noqa: E402


def test_none_isolation(tmp=Path("/tmp/opencode/rsi_isolation_test")):
    method = NoneRSI()
    method.init_run({"state_root": str(tmp / "none"), "run_id": "t"})
    # episode 1 as a normal session would record working memory
    from agent.runtime.runtime import OpenEtaAgentRuntime
    from benchmark.openeta_bridge.build_runtime import BenchmarkRuntime, build_benchmark_tools
    from agent.runtime.skills import SkillRegistry

    tools = build_benchmark_tools([{"name": "move", "description": "x",
                                    "parameters": {"type": "object"}}])
    r1 = BenchmarkRuntime(tools=tools, skills=SkillRegistry(), rollout_enabled=False)
    r1.start_session(task="task one", metadata={})
    r1.memory.save_fact("episode1_note", {"x": 1}, source="test")
    # episode 2 gets a brand-new runtime; nothing from episode 1 is visible
    r2 = BenchmarkRuntime(tools=build_benchmark_tools(
        [{"name": "move", "description": "x", "parameters": {"type": "object"}}]),
        skills=SkillRegistry(), rollout_enabled=False)
    r2.start_session(task="task two", metadata={})
    assert "episode1_note" not in r2.memory.facts
    injection = method.before_episode({"instruction": "task two"})
    assert injection.context_text == ""
    return True


def test_raw_memory_is_the_only_channel(tmp=Path("/tmp/opencode/rsi_isolation_test")):
    method = RawMemoryRSI()
    method.init_run({"state_root": str(tmp / "raw"), "run_id": "t"})
    method.after_episode({"episode_id": "e1", "instruction": "wash the mug",
                          "actions": [{"action": "pick", "public_feedback": "ok"}],
                          "success": True}, {"success": True, "terminated": True, "num_steps": 1})
    inj = method.before_episode({"instruction": "wash the mug"})
    assert "wash the mug" in inj.context_text and inj.estimated_tokens > 0
    return True


if __name__ == "__main__":
    assert test_none_isolation() and test_raw_memory_is_the_only_channel()
    print("test_rsi_state_isolation: PASS")
