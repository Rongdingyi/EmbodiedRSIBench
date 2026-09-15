#!/usr/bin/env python
"""TVRBench render smoke test (iTHOR + ProcTHOR).

Env: RoboTwin (ai2thor 5.0.0 + CloudRendering). No DISPLAY needed.
Run: /data/users/rongdingyi/miniconda3/envs/RoboTwin/bin/python tvrbench_smoke.py
"""
import gzip
import json
import sys
import time
from pathlib import Path

SOURCES = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data/sources")
sys.path.insert(0, str(SOURCES / "tvrbench"))

from tvrbench.env.thor_env import ThorEnv  # noqa: E402

TASKS = SOURCES / "tvrbench" / "data" / "tasks" / "sft.json"
PROCTHOR = SOURCES / "tvrbench" / "data" / "procthor-10k" / "train.jsonl.gz"


def load_house(index: int) -> dict:
    with gzip.open(PROCTHOR, "rt") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise KeyError(index)


def main() -> None:
    tasks = json.loads(TASKS.read_text())

    task = next(t for t in tasks if t["dataset"] == "ithor")
    t0 = time.time()
    env = ThorEnv(scene=task["scene"], width=640, height=480, quality="Medium")
    frame = env.reset(position=task["start"]["position"], rotation_y=task["start"]["rotation_y"],
                      horizon=task["start"]["horizon"])
    print(f"[TVRBench/ithor]    {task['task_id']:32s} frame={frame.shape} {frame.dtype} "
          f"({time.time()-t0:.1f}s)")
    env.controller.stop()

    task = next(t for t in tasks if t["dataset"] == "procthor")
    t0 = time.time()
    house = load_house(int(task["scene"].split("_")[1]))
    env = ThorEnv(scene="FloorPlan1", width=640, height=480, quality="Medium")
    env.load_scene(house)
    frame = env.reset(position=task["start"]["position"], rotation_y=task["start"]["rotation_y"],
                      horizon=task["start"]["horizon"])
    print(f"[TVRBench/procthor] {task['task_id']:32s} frame={frame.shape} {frame.dtype} "
          f"({time.time()-t0:.1f}s)")
    env.controller.stop()
    print("TVRBench render OK")


if __name__ == "__main__":
    main()
