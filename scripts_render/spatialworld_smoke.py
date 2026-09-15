#!/usr/bin/env python
"""SpatialWorld render smoke test (ai2thor + ProcTHOR single-agent).

Env: RoboTwin (ai2thor 5.0.0 + CloudRendering). No DISPLAY needed.
Run: /data/users/rongdingyi/miniconda3/envs/RoboTwin/bin/python spatialworld_smoke.py
"""
import gzip
import json
import time
from pathlib import Path

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

ROOT = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data")
SPATIAL = ROOT / "sources" / "spatialworld" / "data"
PROCTHOR = ROOT / "sources" / "tvrbench" / "data" / "procthor-10k" / "train.jsonl.gz"


def load_house(index: int) -> dict:
    with gzip.open(PROCTHOR, "rt") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise KeyError(index)


def main() -> None:
    # --- ai2thor single-agent task
    tj = json.loads((SPATIAL / "ai2thor" / "tasks" / "ai2thor00000" / "task.json").read_text())
    ij = json.loads((SPATIAL / "ai2thor" / "tasks" / "ai2thor00000" / "init.json").read_text())
    t0 = time.time()
    c = Controller(scene=ij["scene"], platform=CloudRendering, width=640, height=480, quality="Medium")
    for a in ij["actions"]:
        if a == "Done":
            break
        c.step(a)
    ev = c.step("Pass")
    print(f"[SpatialWorld/ai2thor] {tj['task_id']} scene={ij['scene']} "
          f"frame={ev.frame.shape} objects={len(ev.metadata['objects'])} ({time.time()-t0:.1f}s)")
    c.stop()

    # --- procthor single-agent task (scene_index -> procthor-10k house)
    tj = json.loads((SPATIAL / "procthor" / "tasks" / "procthor109" / "task.json").read_text())
    house = load_house(int(tj["scene_index"]))
    t0 = time.time()
    c = Controller(scene="FloorPlan1", platform=CloudRendering, width=640, height=480, quality="Medium")
    c.reset(scene=house)
    ev = c.step("Pass")
    print(f"[SpatialWorld/procthor] {tj['task_id']} scene_index={tj['scene_index']} "
          f"frame={ev.frame.shape} objects={len(ev.metadata['objects'])} ({time.time()-t0:.1f}s)")
    c.stop()
    print("SpatialWorld render OK")


if __name__ == "__main__":
    main()
