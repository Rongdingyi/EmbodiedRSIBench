#!/usr/bin/env python
"""EB-Habitat render smoke test (Habitat-Sim 0.3.0 + ReplicaCAD, one episode).

Env: embench (python 3.9, habitat-sim 0.3.0 withbullet headless, habitat-lab 0.3.0,
             torch CPU, transformers)
Assets: runtime_assets are placed at
        sources/embodiedbench/embodiedbench/envs/eb_habitat/data/
Run: /data/users/rongdingyi/miniconda3/envs/embench/bin/python eb_habitat_smoke.py
"""
import os
import sys
from pathlib import Path

SOURCES = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data/sources")
EB = SOURCES / "embodiedbench"

sys.path.insert(0, str(EB))
os.chdir(EB / "embodiedbench" / "envs" / "eb_habitat")

from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv  # noqa: E402


def main() -> None:
    env = EBHabEnv(eval_set="base", resolution=320)
    obs = env.reset()
    frame = obs["head_rgb"]
    episode = env.env.current_episode if hasattr(env.env, "current_episode") else None
    print(f"[EB-Habitat] episode={getattr(episode, 'episode_id', '?')} "
          f"frame={frame.shape} {frame.dtype} | action_space={env.action_space}")
    print("EB-Habitat render OK")


if __name__ == "__main__":
    main()
