#!/usr/bin/env python
"""EB-ALFRED render smoke test (AI2-THOR 2.1.0, real split task).

Env: bench_avp  (ai2thor==2.1.0, numpy==1.23.5, opencv-python==4.10.0.84,
                flask==1.1.2, werkzeug==1.0.1, gym, networkx, pandas, scipy, Pillow)
Display: private Xvfb (:99) started by 00_start_xvfb.sh
GPU: the 2018.3 Unity build needs `-force-vulkan` to use the NVIDIA Vulkan ICD
     (without it, Unity picks llvmpipe software OpenGL and spins).

Run:
  bash scripts_render/00_start_xvfb.sh 99
  DISPLAY=:99 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    /data/users/rongdingyi/miniconda3/envs/bench_avp/bin/python eb_alfred_smoke.py
"""
import json
import os
import shlex
import sys
from pathlib import Path

ROOT = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data")
SOURCES = ROOT / "sources"
sys.path.insert(0, str(SOURCES / "embodiedbench"))

import ai2thor.controller as ac  # noqa: E402

_orig_unity_command = ac.Controller.unity_command


def patched_unity_command(self, width, height, headless):
    return shlex.split(" ".join(_orig_unity_command(self, width, height, headless)) + " -force-vulkan")


ac.Controller.unity_command = patched_unity_command

import embodiedbench.envs.eb_alfred.gen.constants as constants  # noqa: E402

constants.X_DISPLAY = os.environ.get("DISPLAY", ":0").lstrip(":") or "0"
from embodiedbench.envs.eb_alfred.env.thor_env import ThorEnv  # noqa: E402


def main() -> None:
    splits = json.loads((SOURCES / "embodiedbench" / "embodiedbench" / "envs" / "eb_alfred"
                         / "data" / "splits" / "splits.json").read_text())
    entry = splits["base"][0]
    ann = json.loads((SOURCES / "eb_alfred_hf" / entry["task"] / "pp"
                      / f"ann_{entry['repeat_idx']}.json").read_text())

    env = ThorEnv()
    env.reset(ann["scene"]["floor_plan"])
    env.restore_scene(ann["scene"]["object_poses"], ann["scene"]["object_toggles"],
                      ann["scene"]["dirty_and_empty"])
    ev = env.step(dict(action="Pass"))
    objs = {o["objectType"] for o in ev.metadata["objects"]}
    target = ann["pddl_params"]["object_target"]
    print(f"[EB-ALFRED] scene={ann['scene']['floor_plan']} frame={ev.frame.shape} "
          f"objects={len(objs)} target={target} present={target in objs}")
    print(f"[EB-ALFRED] instruction: {entry['instruction']}")
    print("EB-ALFRED render OK")


if __name__ == "__main__":
    main()
