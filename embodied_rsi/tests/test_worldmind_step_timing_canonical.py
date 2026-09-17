"""Test 11: WorldMind sidecar timing on the canonical runner.

predict_sidecar must run AFTER the agent selected/validated the action and
BEFORE adapter.step; after_step must see the real post-action state.
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))

from benchmark.agent.config import AgentConfig  # noqa: E402
from benchmark.agent.episode_runner import run_episode  # noqa: E402
from benchmark.agent.react_agent import CanonicalMultimodalReActAgent  # noqa: E402
from canonical_fakes import FakeAdapter, FakeBackend, make_protocol, make_task, valid_response  # noqa: E402


class SpyRSI:
    name = "spy"

    def __init__(self):
        self.events: list[str] = []
        self.step_records: list[dict] = []
        self.update_enabled = True
        self.run_ctx: dict = {}
        self._last_update_usage = {"prompt_tokens": 1, "completion_tokens": 1, "calls": 1}
        self._last_update_wall_s = 0.0

    def before_episode(self, task_public_view):
        from benchmark.rsi.context import make_injection

        self.events.append("before_episode")
        return make_injection("")

    def predict_sidecar(self, *, public_observation, action, accountant=None):
        from benchmark.adapters.base import PublicObservation
        import numpy as np

        self.events.append("predict_sidecar")
        assert isinstance(public_observation, dict)
        self.predict_observation = public_observation
        return "predicted"

    def after_step(self, step_public_record):
        self.events.append("after_step")
        self.step_records.append(step_public_record)

    def after_episode(self, trajectory_public, outcome_public):
        self.events.append("after_episode")

    def get_last_update_usage(self):
        return self._last_update_usage

    def get_last_update_wall_s(self):
        return self._last_update_wall_s

    def advance_step(self, index, total):
        self.run_ctx["step"] = index

    def set_update_enabled(self, enabled):
        self.update_enabled = enabled


class TimingAdapter(FakeAdapter):
    def step(self, name, parameters):
        self.events = getattr(self, "events", [])
        self.events.append("adapter_step")
        return super().step(name, parameters)


def test_worldmind_timing_and_state_after(tmp_path):
    adapter = TimingAdapter(outcomes=[
        dict(terminated=False, feedback="step one", action_success=True, success=None),
        dict(terminated=True, feedback="done", action_success=True, success=True),
    ])
    spy = SpyRSI()
    # attach the spy events to the adapter so ordering is observable in one list
    adapter.events = spy.events
    backend = FakeBackend([valid_response("MoveAhead"), valid_response("MoveAhead")])
    agent = CanonicalMultimodalReActAgent(backend=backend, config=AgentConfig())
    res = run_episode(make_task(), adapter, spy, role="experience",
                      config=make_protocol(), output_dir=tmp_path / "ep", agent=agent)

    assert res.env_steps == 2
    assert spy.events.index("predict_sidecar") < spy.events.index("adapter_step")
    assert spy.events.index("adapter_step") < spy.events.index("after_step")
    assert spy.events.index("after_step") < spy.events.index("after_episode")
    for record in spy.step_records:
        assert record["state_after"] == {
            "instruction": "do the thing", "text_feedback": record["public_feedback"],
            "public_metadata": {"image_roles": ["current_view"]}}
        assert record["predicted_state"] == "predicted"
        assert set(record) >= {"task_instruction", "role", "action", "predicted_state",
                               "state_before", "state_after", "public_feedback",
                               "action_success", "terminated", "truncated"}
