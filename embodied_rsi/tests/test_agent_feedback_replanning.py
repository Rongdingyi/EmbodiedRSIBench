"""Test 5: public failure feedback conditions the next planner request."""
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


def test_failure_feedback_reaches_next_request(tmp_path):
    specs = [
        {"name": "MoveAhead", "description": "move",
         "parameters": {"type": "object", "properties": {},
                        "required": [], "additionalProperties": False}},
        {"name": "RotateLeft", "description": "rotate",
         "parameters": {"type": "object", "properties": {},
                        "required": [], "additionalProperties": False}},
    ]
    adapter = FakeAdapter(tool_specs=specs, outcomes=[
        dict(terminated=False, truncated=False, action_success=False,
             feedback="MoveAhead failed: blocked by table"),
        dict(terminated=True, truncated=False, action_success=True, feedback="ok", success=True),
    ])
    backend = FakeBackend([valid_response("MoveAhead"), valid_response("RotateLeft")])
    agent = CanonicalMultimodalReActAgent(backend=backend, config=AgentConfig())
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    res = run_episode(make_task(), adapter, rsi, role="id", config=make_protocol(),
                      output_dir=tmp_path / "ep", agent=agent)
    assert res.env_steps == 2
    assert "blocked by table" in backend.requests[1]["user_text"]
    assert "MoveAhead" in backend.requests[1]["user_text"]   # action history slot
    assert len(backend.requests) == 2
