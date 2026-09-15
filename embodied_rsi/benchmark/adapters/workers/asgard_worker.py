#!/usr/bin/env python
"""AsgardBench simulator worker (RoboTwin env: AI2-THOR 5).

Uses the official `Scenario.do(...)` skill API with `Specifier` (same call path
as `AsgardBench/player.py`), and the official goal evaluator
`goal.evaluate_goals(scenario)` for private evaluation.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_DIR))
from worker_base import Worker, encode_frame, serve  # noqa: E402

ROOT = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data")
REPO = ROOT / "sources" / "asgardbench"
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # AsgardBench writes Generated/status.json relative to CWD

import AsgardBench.scenario as scenario_mod  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402

scenario_mod.Linux64 = CloudRendering  # headless Vulkan instead of X11

from AsgardBench import constants as c  # noqa: E402
from AsgardBench.plan import Goal, ObjectSetup, PlanType, Randomization, SetupAction  # noqa: E402
from AsgardBench.scenario import Scenario  # noqa: E402
from AsgardBench.specifier import Specifier  # noqa: E402

MAX_STEPS = 60

SKILLS = [
    ("find", "Locate an object type in the scene.", "object"),
    ("pickup", "Pick up an object of the given type.", "object"),
    ("put", "Put the held object into a receptacle type.", "receptacle"),
    ("open", "Open an object (fridge, cabinet, microwave).", "object"),
    ("close", "Close an object.", "object"),
    ("toggle_on", "Turn on a device.", "object"),
    ("toggle_off", "Turn off a device.", "object"),
    ("slice", "Slice an object.", "object"),
    ("clean", "Clean a dirty object.", "object"),
    ("spray", "Spray a surface (mirror).", "object"),
    ("drink", "Drink the contents of the held container.", None),
    ("put_away", "Store the given object away.", "object"),
    ("empty_hand", "Drop whatever is in hand.", None),
]
ACTION_MAP = {
    "find": c.Action.FIND, "pickup": c.Action.PICKUP, "put": c.Action.PUT,
    "open": c.Action.OPEN, "close": c.Action.CLOSE, "toggle_on": c.Action.TOGGLE_ON,
    "toggle_off": c.Action.TOGGLE_OFF, "slice": c.Action.SLICE, "clean": c.Action.CLEAN,
    "spray": c.Action.SPRAY, "drink": c.Action.DRINK, "put_away": c.Action.PUT_AWAY,
    "empty_hand": c.Action.EMPTY_HAND,
}


class AsgardWorker(Worker):
    source = "asgard"

    def __init__(self) -> None:
        self.scenario = None
        self.plan = None
        self.steps = 0
        self.done = False
        self.tmpdir = None

    def actions(self) -> list[dict]:
        tools = []
        for name, desc, param in SKILLS:
            props = {}
            if param:
                props = {"object": {"type": "string",
                                    "description": f"{param.title()} type from the scene."}}
            tools.append({
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props,
                               "required": list(props), "additionalProperties": False},
            })
        tools.append({"name": "done", "description": "Finish the episode (goals are then scored).",
                      "parameters": {"type": "object", "properties": {},
                                     "required": [], "additionalProperties": False}})
        return tools

    def _public_objects(self) -> list[str]:
        try:
            objs = self.scenario.controller.last_event.metadata.get("objects", [])
            return sorted({o["objectType"] for o in objs if o.get("visible")})
        except Exception:
            return []

    def _observation(self, feedback: str) -> dict:
        frame = self.scenario.controller.last_event.frame
        return {
            "instruction": self.plan["task_description"],
            "images": [encode_frame(frame)],
            "image_roles": ["current_view"],
            "text_feedback": feedback,
            "public_metadata": {"visible_object_types": self._public_objects()},
        }

    def reset(self, task_record: dict) -> dict:
        self.plan = task_record
        self.steps = 0
        self.done = False
        self.tmpdir = tempfile.mkdtemp(prefix="asgard_worker_")
        self.scenario = Scenario(
            task=task_record["task_description"],
            scene=task_record["scene"],
            name=task_record["name"],
            plan_type=PlanType(task_record["plan_type"]),
            data_folder=self.tmpdir,
            setup_actions=[SetupAction.from_dict(x) for x in (task_record.get("setup_actions") or [])],
            object_setup=ObjectSetup.from_dict(task_record.get("object_setup") or {}),
            randomization=Randomization.from_dict(task_record.get("randomization") or {"seed": 0}),
            goal=Goal.from_dict(task_record.get("goal") or {}),
            initial_pose=task_record.get("initial_pose"),
        )
        return self._observation("Episode started. Use the skills to complete the instruction.")

    def step(self, action: str, parameters: dict) -> dict:
        if self.done:
            raise RuntimeError("episode already finished")
        self.steps += 1
        if action == "done":
            self.done = True
            return {
                "observation": self._observation("Episode finished by agent."),
                "action_success": True, "reward_public": None,
                "terminated": True, "truncated": False,
                "public_feedback": "Episode finished by agent.",
            }
        if action not in ACTION_MAP:
            raise ValueError(f"illegal Asgard action {action!r}")
        object_type = str(parameters.get("object") or "")
        specifier = Specifier(types=[object_type]) if object_type else Specifier()
        try:
            self.scenario.do(ACTION_MAP[action], specifier, "", [])
            ok = self.scenario.step_error is None
            err = self.scenario.step_error
        except Exception as exc:  # noqa: BLE001 — source skill failures are not infra crashes
            ok = False
            err = f"{type(exc).__name__}: {exc}"
        feedback = f"{action} {object_type} -> {'ok' if ok else 'failed'}" + (f" ({err})" if err else "")
        return {
            "observation": self._observation(feedback),
            "action_success": ok,
            "reward_public": None,
            "terminated": False,
            "truncated": self.steps >= MAX_STEPS,
            "public_feedback": feedback,
        }

    def private_evaluate(self) -> dict:
        try:
            goal = self.scenario.raw_plan.goal
            success = bool(goal.evaluate_goals(self.scenario))
        except Exception as exc:  # noqa: BLE001
            return {"success": None, "detail": f"evaluator error: {exc}",
                    "physical_task_id": self.plan.get("physical_task_id")}
        return {"success": success, "physical_task_id": self.plan.get("physical_task_id")}

    def close(self) -> None:
        if self.scenario is not None:
            try:
                self.scenario.controller.stop()
            except Exception:
                pass
            self.scenario = None


if __name__ == "__main__":
    serve(AsgardWorker())
