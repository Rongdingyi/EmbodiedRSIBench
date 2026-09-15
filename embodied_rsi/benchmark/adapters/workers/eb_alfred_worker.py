#!/usr/bin/env python
"""EB-ALFRED simulator worker (bench_avp env: AI2-THOR 2.1.0 + Vulkan).

Runs the official EmbodiedBench EBAlfEnv (language-skill interface). Requires:
  - DISPLAY pointing at the machine's physical X server
  - NVIDIA Vulkan ICD (VK_ICD_FILENAMES)
  - the 2018 Unity build forced to Vulkan via the unity_command patch
  - flask==1.1.2 / werkzeug==1.0.1 in the env (old raw-socket HTTP client)
"""
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_DIR))
from worker_base import Worker, encode_frame, serve  # noqa: E402

ROOT = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data")
EB = ROOT / "sources" / "embodiedbench"
sys.path.insert(0, str(EB))

import ai2thor.controller as ac  # noqa: E402

_orig_unity_command = ac.Controller.unity_command


def _patched_unity_command(self, width, height, headless):
    return shlex.split(" ".join(_orig_unity_command(self, width, height, headless)) + " -force-vulkan")


ac.Controller.unity_command = _patched_unity_command

import embodiedbench.envs.eb_alfred.gen.constants as constants  # noqa: E402

constants.X_DISPLAY = (os.environ.get("DISPLAY", ":0") or ":0").lstrip(":") or "0"

from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv  # noqa: E402

MAX_STEPS = 30

TOOLS = [
    ("find", "Locate an object type.", "object", "find a {object}"),
    ("pick_up", "Pick up an object.", "object", "pick up the {object}"),
    ("put_down", "Put the held object down on a nearby surface.", None, "put down the object in hand"),
    ("drop", "Drop the held object.", None, "drop the object in hand"),
    ("open", "Open a receptacle.", "object", "open the {object}"),
    ("close", "Close a receptacle.", "object", "close the {object}"),
    ("turn_on", "Turn on a device.", "object", "turn on the {object}"),
    ("turn_off", "Turn off a device.", "object", "turn off the {object}"),
    ("slice", "Slice a sliceable object.", "object", "slice the {object}"),
]


class EBAlfredWorker(Worker):
    source = "eb_alfred"

    def __init__(self) -> None:
        self.env = None
        self.task = None
        self.steps = 0

    def actions(self) -> list[dict]:
        tools = []
        for name, desc, param, _tmpl in TOOLS:
            props = {}
            if param:
                props = {"object": {"type": "string", "description": "Object type name."}}
            tools.append({
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props,
                               "required": list(props), "additionalProperties": False},
            })
        return tools

    def _lang(self, action: str, parameters: dict) -> str:
        tmpl = {t[0]: t[3] for t in TOOLS}.get(action)
        if tmpl is None:
            raise ValueError(f"illegal EB-ALFRED action {action!r}")
        return tmpl.format(object=parameters.get("object", ""))

    def _observation(self, info: dict | None, frame) -> dict:
        feedback = (info or {}).get("env_feedback", "Episode started.")
        visible = ((info or {}).get("object_states") or {}).get("visible_objs") or []
        return {
            "instruction": self.task["instruction"],
            "images": [encode_frame(frame)],
            "image_roles": ["current_view"],
            "text_feedback": feedback,
            "public_metadata": {"visible_objects": sorted(set(visible))},
        }

    def reset(self, task_record: dict) -> dict:
        self.task = task_record
        self.steps = 0
        split = task_record["source_split"]
        index = int(task_record["source_entry_index"])
        self.env = EBAlfEnv(eval_set=split, selected_indexes=[index], resolution=500)
        obs = self.env.reset()
        return self._observation(None, obs["head_rgb"])

    def step(self, action: str, parameters: dict) -> dict:
        self.steps += 1
        lang = self._lang(action, parameters)
        obs, _reward, done, info = self.env.step(lang, reasoning="")
        terminated = bool(done)
        truncated = self.steps >= MAX_STEPS and not terminated
        return {
            "observation": self._observation(info, obs["head_rgb"]),
            "action_success": bool(info.get("last_action_success", 0.0)),
            "reward_public": None,
            "terminated": terminated,
            "truncated": truncated,
            "public_feedback": info.get("env_feedback", ""),
        }

    def private_evaluate(self) -> dict:
        success = bool(self.env.env.get_goal_satisfied())
        return {"success": success, "physical_task_id": self.task.get("physical_task_id")}

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.env.controller.stop()
            except Exception:
                pass
            self.env = None


if __name__ == "__main__":
    serve(EBAlfredWorker())
