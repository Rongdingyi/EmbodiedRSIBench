"""Test 9: the runner resets the environment BEFORE reading dynamic tool specs (EB-Habitat)."""
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


def test_reset_before_dynamic_tool_specs(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)],
                          reset_returns_specs=True)
    backend = FakeBackend([valid_response("Dynamic")])
    agent = CanonicalMultimodalReActAgent(backend=backend, config=AgentConfig())
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    res = run_episode(make_task(), adapter, rsi, role="id", config=make_protocol(),
                      output_dir=tmp_path / "ep", agent=agent)
    assert adapter.reset_calls == 1
    assert [call[0] for call in adapter.step_calls] == ["Dynamic"]
    assert "Dynamic" in backend.requests[0]["user_text"]
