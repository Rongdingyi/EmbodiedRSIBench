#!/usr/bin/env python
"""SpatialWorld simulator worker (RoboTwin env: AI2-THOR 5 CloudRendering).

Single-agent ai2thor + ProcTHOR tasks only (Core v1.0 scope).
The private evaluator mirrors SpatialWorld's own
`mllm_base_agent/environments/ai2thor/wrapper.py::_condition_satisfied`
(object_state / object_in_receptacle / object_in_hand; other condition types are
False in the upstream implementation as well).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_DIR))
from worker_base import Worker, encode_frame, serve  # noqa: E402

from ai2thor.controller import Controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402

ROOT = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data")
PROCTHOR_JSONL = ROOT / "sources" / "tvrbench" / "data" / "procthor-10k" / "train.jsonl.gz"

MOVE_ACTIONS = {
    "move_ahead": "MoveAhead", "move_back": "MoveBack",
    "move_left": "MoveLeft", "move_right": "MoveRight",
    "rotate_left": "RotateLeft", "rotate_right": "RotateRight",
    "look_up": "LookUp", "look_down": "LookDown",
}
INTERACT_ACTIONS = {
    "pick_up": "PickupObject", "put": "PutObject", "open": "OpenObject",
    "close": "CloseObject", "toggle_on": "ToggleObjectOn", "toggle_off": "ToggleObjectOff",
    "slice": "SliceObject", "clean": "CleanObject", "fill_with_liquid": "FillObjectWithLiquid",
}
MAX_STEPS = 60


def load_house(index: int) -> dict:
    with gzip.open(PROCTHOR_JSONL, "rt") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise KeyError(index)


def public_objects(metadata: dict) -> list[dict]:
    objs = []
    for o in metadata.get("objects", []):
        if not o.get("visible"):
            continue
        objs.append({
            "id": o["objectId"], "type": o["objectType"],
            "isPickedUp": bool(o.get("isPickedUp", False)),
            "distance": round(float(o.get("distance", 0.0)), 2),
        })
    for o in metadata.get("inventoryObjects") or []:
        objs.append({"id": o["objectId"], "type": o["objectType"],
                     "isPickedUp": True, "distance": 0.0})
    return objs


class SpatialWorldWorker(Worker):
    source = "spatialworld"

    def __init__(self) -> None:
        self.c = None
        self.task = None
        self.steps = 0
        self.done = False
        # cache of every object seen so far (public state accumulates in AI2-THOR)
        self.seen: dict[str, dict] = {}

    def actions(self) -> list[dict]:
        def tool(name, desc, props=None, required=None):
            return {
                "name": name, "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props or {},
                    "required": required or [],
                    "additionalProperties": False,
                },
            }
        obj_prop = {"object": {"type": "string", "description": "Object name or id from the visible objects list."}}
        rec_prop = {"receptacle": {"type": "string", "description": "Target receptacle name or id."}}
        tools = [tool(n, f"{n.replace('_', ' ')} (grid move/rotate)") for n in MOVE_ACTIONS]
        tools += [tool("pick_up", "Pick up an object.", obj_prop, ["object"])]
        tools += [tool("put", "Put the held object into a receptacle.",
                       {**obj_prop, **rec_prop}, ["receptacle"])]
        tools += [tool("open", "Open an object (e.g. fridge, cabinet).", obj_prop, ["object"])]
        tools += [tool("close", "Close an object.", obj_prop, ["object"])]
        tools += [tool("toggle_on", "Turn on a device (e.g. lamp, TV).", obj_prop, ["object"])]
        tools += [tool("toggle_off", "Turn off a device.", obj_prop, ["object"])]
        tools += [tool("slice", "Slice a sliceable object.", obj_prop, ["object"])]
        tools += [tool("clean", "Clean a dirty object.", obj_prop, ["object"])]
        tools += [tool("fill_with_liquid", "Fill a container with liquid.", obj_prop, ["object"])]
        tools += [tool("done", "Finish the episode (task believes complete).")]
        return tools

    # --------------------------------------------------------------- helpers
    def _resolve(self, name_or_id: str) -> str:
        name_or_id = str(name_or_id)
        if name_or_id in self.seen:
            return name_or_id
        for oid, o in self.seen.items():
            if o["objectType"].lower() == name_or_id.lower():
                return oid
        for oid, o in self.seen.items():
            if o["objectType"].lower().startswith(name_or_id.lower()):
                return oid
        # short names like "Fridge|+01.2|..." prefix match
        for oid in self.seen:
            if oid.split("|")[0].lower() == name_or_id.split("|")[0].lower():
                return oid
        return name_or_id

    def _observation(self, event, feedback: str) -> dict:
        md = event.metadata
        for o in md.get("objects", []):
            self.seen[o["objectId"]] = {"objectType": o["objectType"],
                                        "visible": o.get("visible", False)}
        for o in md.get("inventoryObjects") or []:
            self.seen[o["objectId"]] = {"objectType": o["objectType"], "visible": True}
        return {
            "instruction": self.task["instruction"],
            "images": [encode_frame(event.frame)],
            "image_roles": ["current_view"],
            "text_feedback": feedback,
            "public_metadata": {
                "visible_objects": public_objects(md),
                "last_action_success": bool(md.get("lastActionSuccess", False)),
            },
        }

    # --------------------------------------------------------------- episode
    def reset(self, task_record: dict) -> dict:
        self.task = task_record
        self.steps = 0
        self.done = False
        self.seen = {}
        env = task_record["source_backend"]
        if env == "ai2thor":
            scene = task_record["scene_id_raw"]
        else:
            scene = "FloorPlan1"
        self.c = Controller(scene=scene, platform=CloudRendering,
                            width=640, height=480, quality="Medium")
        if env == "procthor":
            self.c.reset(scene=load_house(int(task_record["scene_id_raw"])))
        init_actions = task_record.get("init_actions") or []
        if not init_actions and task_record.get("source_path"):
            init_path = ROOT / task_record["source_path"]
            init_path = init_path.parent / "init.json"
            if init_path.exists():
                init_actions = (json.loads(init_path.read_text()) or {}).get("actions") or []
        for a in init_actions:
            if a == "Done":
                break
            self.c.step(a)
        event = self.c.step("Pass")
        return self._observation(event, "Episode started. Use the tools to complete the instruction.")

    def step(self, action: str, parameters: dict) -> dict:
        if self.done:
            raise RuntimeError("episode already finished")
        self.steps += 1
        ok = True
        if action in MOVE_ACTIONS:
            event = self.c.step(MOVE_ACTIONS[action])
        elif action == "done":
            self.done = True
            event = self.c.last_event
            return {
                "observation": self._observation(event, "Episode finished by agent."),
                "action_success": True, "reward_public": None,
                "terminated": True, "truncated": False,
                "public_feedback": "Episode finished by agent.",
            }
        elif action in INTERACT_ACTIONS:
            oid = self._resolve(parameters.get("object", ""))
            kwargs = {"objectId": oid}
            if action == "put":
                kwargs = {"receptacleObjectId": self._resolve(parameters.get("receptacle", ""))}
            if action == "fill_with_liquid":
                kwargs["fillLiquid"] = "water"
            event = self.c.step(dict(action=INTERACT_ACTIONS[action], **kwargs))
        else:
            raise ValueError(f"illegal SpatialWorld action {action!r}")
        ok = bool(event.metadata.get("lastActionSuccess", False))
        err = event.metadata.get("errorMessage") or ""
        feedback = f"{action} -> {'ok' if ok else 'failed'}" + (f" ({err})" if err and not ok else "")
        terminated = False
        truncated = self.steps >= MAX_STEPS
        return {
            "observation": self._observation(event, feedback),
            "action_success": ok,
            "reward_public": None,
            "terminated": terminated,
            "truncated": truncated,
            "public_feedback": feedback,
        }

    # ------------------------------------------------------------- evaluator
    @staticmethod
    def _type_matches(actual: str, expected: str) -> bool:
        if not expected:
            return False
        return actual == expected or actual.startswith(expected + "|")

    def _condition_satisfied(self, condition: dict, metadata: dict) -> bool:
        ctype = condition.get("type", "object_state")
        target_types = self.task.get("target_object_types") or []
        if ctype == "object_state":
            object_type = condition.get("object_type")
            field = condition.get("field") or condition.get("state") or "isOpen"
            target_value = condition.get("value", True)
            for obj in metadata.get("objects", []):
                if object_type and not self._type_matches(obj.get("objectType", ""), object_type):
                    continue
                if not object_type and obj.get("objectType") not in target_types:
                    continue
                if obj.get(field, False) == target_value:
                    return True
            return False
        if ctype == "object_in_receptacle":
            object_type = condition.get("object_type")
            receptacle_type = condition.get("receptacle_type")
            if not object_type or not receptacle_type:
                return False
            for obj in metadata.get("objects", []):
                if not self._type_matches(obj.get("objectType", ""), object_type):
                    continue
                for parent_id in obj.get("parentReceptacles") or []:
                    parent_type = parent_id.split("|")[0] if "|" in parent_id else parent_id
                    if parent_type == receptacle_type:
                        return True
            return False
        if ctype == "object_in_hand":
            target_type = condition.get("object_type")
            for item in metadata.get("inventoryObjects") or []:
                if self._type_matches(item.get("objectType", ""), target_type):
                    return True
            return False
        return False

    def private_evaluate(self) -> dict:
        md = self.c.last_event.metadata
        conditions = [c for c in (self.task.get("success_conditions") or []) if isinstance(c, dict)]
        results = [self._condition_satisfied(c, md) for c in conditions]
        logic = self.task.get("success_logic") or "OR"
        success = bool(any(results)) if logic == "OR" else bool(all(results) and results)
        return {
            "success": success,
            "success_logic": logic,
            "condition_results": results,
            "physical_task_id": self.task.get("physical_task_id"),
        }

    def close(self) -> None:
        if self.c is not None:
            try:
                self.c.stop()
            except Exception:
                pass
            self.c = None


if __name__ == "__main__":
    serve(SpatialWorldWorker())
