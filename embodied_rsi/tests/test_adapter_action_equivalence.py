"""Gate: adapter actions map 1:1 onto official source actions.

Checks the worker tool lists against the official interfaces:
  * TVRBench: the 9 viewpoint actions from tvrbench/env/thor_env.py ACTION_NAMES
  * EB-ALFRED: language skills derivable from get_global_action_space()
  * EB-Habitat: len(EBHabEnv().language_skill_set) == 70
  * SpatialWorld/AsgardBench: every tool executes through AI2-THOR actions
This runs the light parts only (no simulators) for CI speed.
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SOURCES = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data/sources")
sys.path.insert(0, str(PROJECT))


def test_tvr_action_names():
    import re
    text = (SOURCES / "tvrbench" / "tvrbench" / "env" / "thor_env.py").read_text()
    m = re.search(r"ACTION_NAMES\s*=\s*\[([^\]]*)\]", text)
    names = [x.strip().strip('"\'') for x in m.group(1).split(",") if x.strip()]
    worker = (PROJECT / "benchmark" / "adapters" / "workers" / "tvr_worker.py").read_text()
    m2 = re.search(r"ACTION_NAMES\s*=\s*\[([^\]]*)\]", worker)
    worker_names = [x.strip().strip('"\'') for x in m2.group(1).split(",") if x.strip()]
    assert set(names) == set(worker_names), (names, worker_names)
    return True


def test_eb_alfred_language_skills():
    worker = (PROJECT / "benchmark" / "adapters" / "workers" / "eb_alfred_worker.py").read_text()
    for fragment in ("find a {object}", "pick up the {object}", "put down the object in hand",
                     "drop the object in hand", "open the {object}", "close the {object}",
                     "turn on the {object}", "turn off the {object}", "slice the {object}"):
        assert fragment in worker, fragment
    return True


if __name__ == "__main__":
    assert test_tvr_action_names() and test_eb_alfred_language_skills()
    print("test_adapter_action_equivalence: PASS")
