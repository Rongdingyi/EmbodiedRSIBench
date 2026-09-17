"""Test 14: every planner request is audited; usage categories are counted."""
import json
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


def test_request_audit_roles_and_usage(tmp_path):
    adapter = FakeAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
    from benchmark.runner.accounting import Accountant

    accountant = Accountant(episode_dir=tmp_path / "ep")
    backend = FakeBackend([valid_response("MoveAhead")], accountant=accountant)
    agent = CanonicalMultimodalReActAgent(backend=backend, config=AgentConfig(),
                                          accountant=accountant)
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    res = run_episode(make_task(), adapter, rsi, role="id", config=make_protocol(),
                      output_dir=tmp_path / "ep", agent=agent, accountant=accountant)

    assert res.planner_calls == 1
    assert res.env_steps == 1
    assert res.accounting["planner_input_tokens"] == 7
    assert res.accounting["planner_output_tokens"] == 3
    dumps = [json.loads(line) for line in
             (tmp_path / "ep" / "public_context_dump.jsonl").read_text().splitlines() if line.strip()]
    assert dumps and dumps[0]["role"] == "canonical_planner"
    assert dumps[0]["has_image"] is True
