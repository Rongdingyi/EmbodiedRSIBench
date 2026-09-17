"""P0: adapter.close() runs for every termination path (via the other suite's fixtures)."""
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


def _rsi(tmp_path):
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    return rsi


def _closed(adapter, *, agent=None, tmp_path, role="id"):
    run_episode(make_task(), adapter, _rsi(tmp_path), role=role,
                config=make_protocol(), output_dir=tmp_path / "ep", agent=agent)
    return adapter.closed


def test_close_on_normal_termination(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
    agent = make_test_agent([valid_response("MoveAhead")])
    assert _closed(adapter, agent=agent, tmp_path=tmp_path) is True


def test_close_on_budget_truncation(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=False, feedback="s")] * 5)
    agent = make_test_agent([valid_response("MoveAhead")] * 5)
    assert _closed(adapter, agent=agent, tmp_path=tmp_path) is True


def test_close_on_evaluator_exception(tmp_path):
    class BadEvalAdapter(FakeAdapter):
        def private_evaluate(self):
            raise RuntimeError("evaluator exploded")

    adapter = BadEvalAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
    agent = make_test_agent([valid_response("MoveAhead")])
    assert _closed(adapter, agent=agent, tmp_path=tmp_path) is True
