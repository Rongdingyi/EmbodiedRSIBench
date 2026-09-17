"""Test 1/2: one planner decision -> exactly one environment action; multi-action rejected."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))

from benchmark.agent.episode_runner import run_episode  # noqa: E402
from canonical_fakes import FakeAdapter, FakeBackend, make_protocol, make_task, valid_response  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from benchmark.runner import gates  # noqa: E402


def test_one_action_per_turn(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
    backend = FakeBackend([valid_response("MoveAhead")])
    from benchmark.agent.react_agent import CanonicalMultimodalReActAgent
    from benchmark.agent.config import AgentConfig

    agent = CanonicalMultimodalReActAgent(backend=backend, config=AgentConfig())
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    res = run_episode(make_task(), adapter, rsi, role="id", config=make_protocol(),
                      output_dir=tmp_path / "ep", agent=agent)
    assert res.status == gates.PASS
    assert len(adapter.step_calls) == 1
    assert adapter.step_calls[0][0] == "MoveAhead"
    assert len(backend.requests) == 1


def test_multi_action_is_rejected(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
    backend = FakeBackend(['{"thought":"x","actions":[{"name":"MoveAhead","parameters":{}}]}',
                           valid_response("MoveAhead")])
    from benchmark.agent.react_agent import CanonicalMultimodalReActAgent
    from benchmark.agent.config import AgentConfig

    agent = CanonicalMultimodalReActAgent(backend=backend, config=AgentConfig())
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    res = run_episode(make_task(), adapter, rsi, role="id", config=make_protocol(),
                      output_dir=tmp_path / "ep", agent=agent)
    # first response rejected -> retry -> exactly one executed action
    assert len(adapter.step_calls) == 1
    assert len(backend.requests) == 2
    assert res.accounting["planner_invalid_action_count"] == 1
    assert res.accounting["planner_validation_retries"] == 1
