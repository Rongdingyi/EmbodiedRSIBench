#!/usr/bin/env python
"""AsgardBench render smoke test (one magt_benchmark plan).

Env: RoboTwin (ai2thor 5.0.0). The repo's Scenario hardcodes platform=Linux64 (X11);
we swap it for CloudRendering at runtime so no X server is needed.
Run: /data/users/rongdingyi/miniconda3/envs/RoboTwin/bin/python asgardbench_smoke.py
"""
import json
import os
import sys
import time
from pathlib import Path

SOURCES = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data/sources")
REPO = SOURCES / "asgardbench"
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # AsgardBench writes Generated/status.json relative to CWD

import AsgardBench.scenario as scenario_mod  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402

scenario_mod.Linux64 = CloudRendering  # headless Vulkan instead of X11

from AsgardBench.plan import Goal, ObjectSetup, PlanType, Randomization, SetupAction  # noqa: E402
from AsgardBench.scenario import Scenario  # noqa: E402


def main() -> None:
    bench = REPO / "Generated" / "magt_benchmark"
    plan_dir = bench / sorted(p.name for p in bench.iterdir() if p.is_dir())[0]
    plan = json.loads((plan_dir / "plan.json").read_text())

    data_folder = Path("/tmp/asgardbench_smoke")
    data_folder.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sc = Scenario(
        task=plan["task_description"],
        scene=plan["scene"],
        name=plan["name"],
        plan_type=PlanType(plan["plan_type"]),
        data_folder=str(data_folder),
        setup_actions=[SetupAction.from_dict(x) for x in plan["setup_actions"]],
        object_setup=ObjectSetup.from_dict(plan["object_setup"]),
        randomization=Randomization.from_dict(plan["randomization"]),
        goal=Goal.from_dict(plan["goal"]),
        initial_pose=plan["initial_pose"],
    )
    frame = sc.controller.last_event.frame
    print(f"[AsgardBench] {plan['name']} scene={plan['scene']} frame={frame.shape} "
          f"({time.time()-t0:.1f}s)")
    sc.controller.stop()
    print("AsgardBench render OK")


if __name__ == "__main__":
    main()
