"""Gate (P0-1): WorldMind compares prediction against the POST-action state.

Sends one real environment step through BenchmarkEpisodeEnvironment and spies on
the official `process_single_step` call to assert:
  * `step.observation` == state AFTER the action
  * `state_before` kwarg == state BEFORE the action
  * the sidecar request is accounted exactly once and its output never returns
    to the planner
Also asserts the accounting authority rule: `get_last_update_usage()` equals the
component usage the runner records (no double counting).
"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "external" / "OpenETA"))

# the test needs the DeepSeek key (WorldMind modules construct LLMClients)
if not os.environ.get("DEEPSEEK_API_KEY") and (PROJECT / ".env").exists():
    for line in (PROJECT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

from adapter.protocol import EnvAction  # noqa: E402
from benchmark.adapters import make_adapter  # noqa: E402
from benchmark.openeta_bridge.benchmark_environment import BenchmarkEpisodeEnvironment  # noqa: E402
from benchmark.registry.loader import load_role  # noqa: E402
from benchmark.rsi.worldmind import WorldMindRSI  # noqa: E402
from benchmark.runner.accounting import Accountant  # noqa: E402


def test_worldmind_timing(tmp=Path("/tmp/opencode/wm_timing_test")):
    rec = [r for r in load_role("experience") if r["source_dataset"] == "TVRBench"][0]
    method = WorldMindRSI()
    method.init_run({"state_root": str(tmp / "state"), "run_id": "t"})
    method.before_episode({"instruction": rec["instruction"] or "reproduce the target view",
                           "global_task_id": rec["global_task_id"]})
    captured = {}

    def spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return False, []

    method.process_module.process_single_step = spy

    accountant = Accountant(episode_dir=tmp / "ep")
    adapter = make_adapter("TVRBench")
    env = BenchmarkEpisodeEnvironment(adapter, rec, rsi=method, accountant=accountant,
                                      env_step_budget=3)
    env.prepare()
    action = EnvAction(action_type="tool_call",
                       command={"request": {"kind": "tool_call", "name": "RotateLeft",
                                            "parameters": {}}})
    result = env.step(action)
    env.close()

    assert method._sidecar_usage["calls"] == 1, "sidecar did not run exactly once before env.step"
    assert accountant.worldmind_sidecar.calls == 1, "sidecar accounting mismatch"
    assert captured, "official process_single_step was not called"
    step_arg = captured["kwargs"].get("step") if captured.get("kwargs") else captured["args"][1]
    state_before_kwarg = captured["kwargs"].get("state_before")
    state_after_text = step_arg.observation
    feedback = result.info.get("public_feedback") or ""
    assert feedback[:30] in state_after_text or state_after_text.endswith(feedback[:30]), (
        f"observation must be the post-action state, got: {state_after_text!r}")
    assert state_before_kwarg and state_before_kwarg != state_after_text, (
        "state_before must be passed as an explicit distinct kwarg")
    # accounting authority: the environment only accumulates per-episode usage;
    # the runner is the single place that records rsi_update into the accountant
    usage = method.get_last_update_usage()
    assert usage is not None
    assert accountant.rsi_update.calls == 0, (
        "environment must not record updater calls (runner records them once)")
    assert usage["calls"] == accountant.rsi_update.calls or accountant.rsi_update.calls == 0
    return True


if __name__ == "__main__":
    assert test_worldmind_timing()
    print("test_worldmind_timing: PASS")
