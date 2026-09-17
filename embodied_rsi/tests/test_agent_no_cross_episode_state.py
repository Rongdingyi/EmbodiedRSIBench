"""Test 3/13: the agent keeps no cross-episode state; memory clears each episode."""
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


def test_second_episode_prompt_has_no_first_episode_content(tmp_path):
    agent = CanonicalMultimodalReActAgent(
        backend=FakeBackend([valid_response("MoveAhead")]),
        config=AgentConfig())
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})

    adapter1 = FakeAdapter(outcomes=[dict(terminated=True, feedback="first_feedback_marker",
                                          success=True)])
    run_episode(make_task("task_0001"), adapter1, rsi, role="id", config=make_protocol(),
                output_dir=tmp_path / "ep1", agent=agent)

    backend2 = FakeBackend([valid_response("MoveAhead")])
    agent.backend = backend2
    adapter2 = FakeAdapter(outcomes=[dict(terminated=True, feedback="second", success=True)])
    run_episode(make_task("task_0002"), adapter2, rsi, role="id", config=make_protocol(),
                output_dir=tmp_path / "ep2", agent=agent)

    second_prompt = backend2.requests[0]["user_text"]
    assert "first_feedback_marker" not in second_prompt
    assert agent.memory.steps == []


def test_agent_is_fresh_instance_per_episode_in_runner(tmp_path, monkeypatch):
    """The runner builds its own agent per run; no shared memory objects."""
    import benchmark.agent.episode_runner as er

    created = []
    real_cls = er.CanonicalMultimodalReActAgent

    class TrackingAgent(real_cls):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

    monkeypatch.setattr(er, "CanonicalMultimodalReActAgent", TrackingAgent)
    monkeypatch.setattr(er, "CanonicalVLMBackend",
                        lambda *a, **k: FakeBackend([valid_response("MoveAhead")] * 4))
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    for index in (1, 2):
        adapter = FakeAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
        res = run_episode(make_task(f"task_{index:04d}"), adapter, rsi, role="id",
                          config=make_protocol(), output_dir=tmp_path / f"ep{index}")
        assert res.status == "PASS"
    assert len(created) == 2
    assert created[0] is not created[1]
    assert all(agent.memory.steps == [] for agent in created)
