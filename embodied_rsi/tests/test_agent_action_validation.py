"""Tests 6/7/8: deterministic recovery, invalid action names, no random fallback."""
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))

from benchmark.agent.backend import PlannerProtocolError  # noqa: E402
from benchmark.agent.config import AgentConfig  # noqa: E402
from benchmark.agent.episode_runner import run_episode  # noqa: E402
from benchmark.agent.react_agent import CanonicalMultimodalReActAgent  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from canonical_fakes import FakeAdapter, FakeBackend, make_protocol, make_task, valid_response  # noqa: E402


def test_invalid_json_recovery_counts(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
    agent = CanonicalMultimodalReActAgent(
        backend=FakeBackend(["not json at all", valid_response("MoveAhead")]),
        config=AgentConfig())
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    res = run_episode(make_task(), adapter, rsi, role="id", config=make_protocol(),
                      output_dir=tmp_path / "ep", agent=agent)
    assert len(adapter.step_calls) == 1
    assert res.planner_calls == 2
    assert res.accounting["planner_invalid_json_count"] == 1
    assert res.accounting["planner_validation_retries"] == 1


def test_invalid_action_name_never_touches_env(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
    agent = CanonicalMultimodalReActAgent(
        backend=FakeBackend([valid_response("Teleport"),
                             valid_response("MoveAhead")]),
        config=AgentConfig())
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    res = run_episode(make_task(), adapter, rsi, role="id", config=make_protocol(),
                      output_dir=tmp_path / "ep", agent=agent)
    assert [call[0] for call in adapter.step_calls] == ["MoveAhead"]
    assert res.accounting["planner_invalid_action_count"] == 1


def test_no_random_fallback_after_three_failures(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
    agent = CanonicalMultimodalReActAgent(
        backend=FakeBackend(["bad", "still bad", "hopeless"]), config=AgentConfig())
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    res = run_episode(make_task(), adapter, rsi, role="id", config=make_protocol(),
                      output_dir=tmp_path / "ep", agent=agent)
    assert adapter.step_calls == []
    assert res.error and "PlannerProtocolError" in res.error
    assert res.stop_reason == "planner_protocol_failure"
    assert res.status == "FAIL"
