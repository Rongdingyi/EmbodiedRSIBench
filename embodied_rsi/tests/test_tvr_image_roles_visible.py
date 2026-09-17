"""P0: the planner prompt always states which image is current_view / target_view."""
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))

from benchmark.adapters.base import PublicObservation  # noqa: E402
from benchmark.agent.config import AgentConfig  # noqa: E402
from benchmark.agent.episode_runner import run_episode  # noqa: E402
from benchmark.agent.react_agent import CanonicalMultimodalReActAgent  # noqa: E402
from benchmark.rsi.none import NoneRSI  # noqa: E402
from canonical_fakes import FakeAdapter, make_protocol, make_task, make_test_agent, valid_response  # noqa: E402


class DualViewAdapter(FakeAdapter):
    """TVR-style observation: current + target frame with public roles."""

    def reset(self, task):
        self.reset_calls += 1
        images = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]
        return PublicObservation("reproduce the target viewpoint", images, "started",
                                 {"image_roles": ["current_view", "target_view"],
                                  "target_pose": [9, 9, 9]})


def test_image_roles_are_visible_in_prompt(tmp_path):
    adapter = DualViewAdapter(outcomes=[dict(terminated=True, feedback="ok", success=True)])
    backend_records = []
    agent = make_test_agent([valid_response("MoveAhead")])
    original_complete = agent.backend.complete

    def spy_complete(**kwargs):
        backend_records.append(kwargs)
        return original_complete(**kwargs)

    agent.backend.complete = spy_complete
    rsi = NoneRSI()
    rsi.init_run({"state_root": str(tmp_path / "rsi")})
    res = run_episode(make_task(), adapter, rsi, role="id", config=make_protocol(),
                      output_dir=tmp_path / "ep", agent=agent)

    prompt = backend_records[0]["user_text"]
    assert "## Visual inputs" in prompt
    assert "Image 1: current_view" in prompt
    assert "Image 2: target_view" in prompt
    assert len(backend_records[0]["images"]) == 2
    # private metadata must not ride along in the role section
    assert "target_pose" not in prompt
    assert res.status == "PASS"
