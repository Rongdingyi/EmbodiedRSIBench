"""Gate: no private field may reach a planner request or public observation."""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "external" / "OpenETA"))

FORBIDDEN_KEYS = {
    "golden_actions", "goal_preds", "success_conditions", "target_pose",
    "physical_task_id", "transfer_pair_id", "retention_anchor_id",
    "private_eval_metadata", "target",
}


def _scan(node, path, hits):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in FORBIDDEN_KEYS:
                hits.append(f"{path}/{key}")
            _scan(value, f"{path}/{key}", hits)
    elif isinstance(node, list):
        for i, value in enumerate(node[:10]):
            _scan(value, f"{path}[{i}]", hits)


def test_observation_and_requests_clean():
    from benchmark.registry.loader import load_role
    from benchmark.adapters import make_adapter
    from benchmark.openeta_bridge.build_runtime import build_runtime
    from benchmark.openeta_bridge.episode_runner import to_env_observation

    rec = [r for r in load_role("id") if r["source_dataset"] == "TVRBench"][0]
    adapter = make_adapter("TVRBench")
    obs = adapter.reset(rec)
    hits = []
    _scan(obs.public_metadata, "public_metadata", hits)
    assert not hits, f"private keys in public observation: {hits}"
    # tool specs + env observation metadata are the only planner-visible payload
    tools = adapter.build_tool_specs(rec)
    env_obs = to_env_observation(obs, step_idx=0)
    hits = []
    _scan(json.loads(json.dumps(env_obs.to_dict(), default=str)), "env_obs", hits)
    assert not hits, f"private keys in planner observation: {hits}"
    adapter.close()
    return True


if __name__ == "__main__":
    assert test_observation_and_requests_clean()
    print("test_no_private_leakage: PASS")
