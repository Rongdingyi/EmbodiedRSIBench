"""Tests 15/16: source termination stops the loop; action budget truncates deterministically."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))

from benchmark.agent.config import AgentConfig  # noqa: E402
from benchmark.agent.episode_runner import run_episode  # noqa: E402
from benchmark.agent.react_agent import CanonicalMultimodalReActAgent  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from canonical_fakes import FakeAdapter, FakeBackend, make_protocol, make_task, valid_response  # noqa: E402


def test_terminated_stops_planner(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=True, feedback="done", success=True)])
    backend = FakeBackend([valid_response("MoveAhead")])
    agent = CanonicalMultimodalReActAgent(backend=backend, config=AgentConfig())
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    res = run_episode(make_task(), adapter, rsi, role="id", config=make_protocol(),
                      output_dir=tmp_path / "ep", agent=agent)
    assert res.stop_reason == "source_terminated"
    assert len(backend.requests) == 1
    assert len(adapter.step_calls) == 1


def test_max_action_budget_truncates(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=False, feedback="step", success=False)] * 10)
    backend = FakeBackend([valid_response("MoveAhead")] * 10)
    agent = CanonicalMultimodalReActAgent(backend=backend, config=AgentConfig())
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    res = run_episode(make_task(), adapter, rsi, role="id", config=make_protocol(max_actions=3),
                      output_dir=tmp_path / "ep", agent=agent)
    assert res.env_steps == 3
    assert res.stop_reason == "max_agent_actions"
    assert res.outcome["truncated"] is True
