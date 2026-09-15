"""Gate: model-visible observation carries only allowlisted metadata (P1-3)."""
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "external" / "OpenETA"))

from benchmark.adapters.base import PublicObservation  # noqa: E402
from benchmark.openeta_bridge.benchmark_environment import BenchmarkEpisodeEnvironment  # noqa: E402


class _FakeAdapter:
    source = "fake"

    def build_tool_specs(self, task):
        return [{"name": "Move", "description": "x", "parameters": {"type": "object"}}]


def test_allowlist():
    env = BenchmarkEpisodeEnvironment(_FakeAdapter(), {"global_task_id": "t"})
    public = PublicObservation(
        instruction="do it",
        images=[np.zeros((4, 4, 3), dtype=np.uint8)],
        text_feedback="started",
        public_metadata={"image_roles": ["current_view"], "visible_objects": [{"id": "Mug|x"}],
                         "target_pose": [1, 2, 3], "success_conditions": [{"x": 1}]},
    )
    obs = env._to_env_observation(public, step_idx=0)
    assert "visible_objects" not in obs.metadata, "object ids leaked into planner metadata"
    assert "target_pose" not in obs.metadata
    assert "success_conditions" not in obs.metadata
    assert obs.metadata["image_roles"] == ["current_view"]
    assert obs.metadata["image_artifacts"] == []      # artifact_dir not set in this unit test
    return True


if __name__ == "__main__":
    assert test_allowlist()
    print("test_observation_allowlist: PASS")
