"""Gate: OpenETA repo is pinned, clean, and native SI is disabled."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "external" / "OpenETA"))


def test_frozen():
    from benchmark.openeta_bridge import freeze
    assert freeze.head_commit() == freeze.OPENETA_PIN
    assert freeze.git_status_clean()
    assert freeze.diff_against_pin() == ""
    hashes = freeze.critical_hashes()
    assert len(hashes) == len(freeze.CRITICAL_FILES)
    manifest = freeze.freeze_manifest()
    assert manifest["native_self_improvement"] is False
    assert manifest["native_auto_apply"] is False
    assert manifest["python_exec_enabled"] is False
    assert manifest["web_tools_enabled"] is False


def test_runtime_si_disabled():
    from benchmark.openeta_bridge.build_runtime import build_benchmark_tools, BenchmarkRuntime
    from benchmark.openeta_bridge.freeze import disable_native_self_improvement
    from agent.runtime.skills import SkillRegistry

    runtime = BenchmarkRuntime(tools=build_benchmark_tools(
        [{"name": "move_ahead", "description": "x", "parameters": {"type": "object"}}]),
        skills=SkillRegistry(), rollout_enabled=False)
    disable_native_self_improvement(runtime)
    reviewer = runtime.self_improvement_reviewer
    assert reviewer.config.enabled is False
    assert reviewer.config.auto_apply_reviewed is False
    assert reviewer.auto_applier is None
    names = {t.name for t in runtime.tools.list()}
    assert "python_exec" not in names and "web_search" not in names


if __name__ == "__main__":
    test_frozen(); test_runtime_si_disabled(); print("test_openeta_frozen: PASS")
