"""P0: adapter lifecycle robustness — reset failures return results, close always runs."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))

from benchmark.agent.config import AgentConfig  # noqa: E402
from benchmark.agent.episode_runner import run_episode  # noqa: E402
from benchmark.agent.react_agent import CanonicalMultimodalReActAgent  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from canonical_fakes import (FakeAdapter, FakeBackend, make_protocol, make_task,  # noqa: E402
                            make_test_agent, valid_response)


class ResetFailAdapter(FakeAdapter):
    def reset(self, task):
        self.reset_calls += 1
        raise RuntimeError("simulated reset failure")


class StepFailAdapter(FakeAdapter):
    def step(self, name, parameters):
        self.step_calls.append((name, dict(parameters)))
        raise RuntimeError("simulated step failure")


def _rsi(tmp_path):
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    return rsi


def test_reset_failure_returns_canonical_result(tmp_path):
    adapter = ResetFailAdapter()
    res = run_episode(make_task(), adapter, _rsi(tmp_path), role="id",
                      config=make_protocol(), output_dir=tmp_path / "ep",
                      agent=make_test_agent([]))
    assert res.status == "FAIL"
    assert res.stop_reason == "simulator_infrastructure_error"
    assert "simulated reset failure" in (res.error or "")
    assert res.public_trajectory == []
    assert res.env_steps == 0
    assert res.private_evaluation.get("success") is None


def test_reset_failure_closes_adapter(tmp_path):
    adapter = ResetFailAdapter()
    run_episode(make_task(), adapter, _rsi(tmp_path), role="id",
                config=make_protocol(), output_dir=tmp_path / "ep",
                agent=make_test_agent([]))
    assert adapter.closed is True


def test_step_failure_closes_adapter_and_records_infra(tmp_path):
    adapter = StepFailAdapter()
    agent = CanonicalMultimodalReActAgent(backend=FakeBackend([valid_response("MoveAhead")]),
                                          config=AgentConfig())
    res = run_episode(make_task(), adapter, _rsi(tmp_path), role="id",
                      config=make_protocol(), output_dir=tmp_path / "ep", agent=agent)
    assert adapter.closed is True
    assert res.stop_reason == "simulator_infrastructure_error"
    assert res.status == "FAIL"


def test_planner_protocol_failure_closes_adapter(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
    agent = CanonicalMultimodalReActAgent(backend=FakeBackend(["bad", "bad", "bad"]),
                                          config=AgentConfig())
    res = run_episode(make_task(), adapter, _rsi(tmp_path), role="id",
                      config=make_protocol(), output_dir=tmp_path / "ep", agent=agent)
    assert adapter.closed is True
    assert res.stop_reason == "planner_protocol_failure"
    assert res.env_steps == 0
