"""Test 10: private observation fields never reach the planner request."""
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


class LeakyAdapter(FakeAdapter):
    def reset(self, task):
        obs = super().reset(task)
        obs.public_metadata.update({
            "private_eval_metadata": {"golden_actions": ["Open"]},
            "target_pose": [1.0, 2.0, 3.0],
            "physical_task_id": "abc123def456",
        })
        return obs


def test_private_observation_fields_are_never_sent(tmp_path):
    adapter = LeakyAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
    backend = FakeBackend([valid_response("MoveAhead")])
    agent = CanonicalMultimodalReActAgent(backend=backend, config=AgentConfig())
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    task = make_task("leak_sentinel_001122")
    task["physical_task_id"] = "abc123def456"
    res = run_episode(task, adapter, rsi, role="id", config=make_protocol(),
                      output_dir=tmp_path / "ep", agent=agent)
    prompt = backend.requests[0]["user_text"]
    assert "golden_actions" not in prompt
    assert "target_pose" not in prompt
    assert "abc123def456" not in prompt
    # the leak sentinel fires if a forbidden private value reaches a request
    assert res.leakage_violations == []
