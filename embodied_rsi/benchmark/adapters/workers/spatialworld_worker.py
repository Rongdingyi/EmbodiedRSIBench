#!/usr/bin/env python
"""SpatialWorld simulator worker (RoboTwin env: AI2-THOR 5 CloudRendering).

Single-agent ai2thor + ProcTHOR tasks only (Core v1.0 scope).

Public action interface = the source's unified abstraction (P1-2):

    Move(direction, distance)   -> MoveAhead/Back/Left/Right (0.25 m grid)
    Rotate(direction, degrees)  -> RotateLeft/RotateRight (90 deg steps)
    Tilt(direction, degrees)    -> LookUp/LookDown (30 deg steps)
    ChangePosture(posture)      -> Stand / Crouch
    Pick(object)                -> PickupObject
    Place(receptacle)           -> PutObject
    ChangeState(object, state)  -> Open/Close/ToggleOn/ToggleOff/Slice/Clean/Fill
    Manipulate(action, object)  -> whitelisted generic AI2-THOR action
    EndTask()                   -> finish the episode

Private evaluator mirrors SpatialWorld's own
`mllm_base_agent/environments/ai2thor/wrapper.py::_condition_satisfied`.
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

MAX_STEPS = 60
GRID = 0.25
ROTATE_STEP = 90
TILT_STEP = 30
MOVE_LIMIT = 4          # max grid cells per Move call
ROTATE_LIMIT = 2        # max 90-deg steps per Rotate call

STATE_ACTIONS = {
    "open": "OpenObject",
    "close": "CloseObject",
    "on": "ToggleObjectOn",
    "off": "ToggleObjectOff",
    "toggle": "ToggleObjectOn",
    "slice": "SliceObject",
    "clean": "CleanObject",
    "fill": "FillObjectWithLiquid",
    "empty": "EmptyLiquidFromObject",
    "break": "BreakObject",
    "dirty": "DirtyObject",
    "cook": "CookObject",
}
MANIPULATE_WHITELIST = {
    "DropHandObject", "PickupObject", "PutObject", "OpenObject", "CloseObject",
    "ToggleObjectOn", "ToggleObjectOff", "SliceObject", "CleanObject",
    "FillObjectWithLiquid", "EmptyLiquidFromObject",
}


def load_house(index: int) -> dict:
    with gzip.open(PROCTHOR_JSONL, "rt") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise KeyError(index)


class SpatialWorldWorker(Worker):
    source = "spatialworld"

    def __init__(self) -> None:
        self.c = None
        self.task = None
        self.steps = 0
        self.done = False
        self.objects_by_id: dict[str, dict] = {}

    # ------------------------------------------------------------- tool schema
    def actions(self) -> list[dict]:
        def tool(name, desc, props=None, required=None):
            return {"name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props or {},
                                   "required": required or [],
                                   "additionalProperties": False}}
        obj = {"object": {"type": "string", "description": "Object type name (e.g. Mug, Lettuce)."}}
        rec = {"receptacle": {"type": "string", "description": "Receptacle type name."}}
        return [
            tool("Move", "Move on the grid.",
                 {"direction": {"type": "string", "enum": ["ahead", "back", "left", "right"]},
                  "distance": {"type": "number", "description": "Meters (0.25 per cell, max 1.0)."}},
                 ["direction"]),
            tool("Rotate", "Rotate the view horizontally.",
                 {"direction": {"type": "string", "enum": ["left", "right"]},
                  "degrees": {"type": "number", "description": "90 or 180."}}, ["direction"]),
            tool("Tilt", "Tilt the camera vertically.",
                 {"direction": {"type": "string", "enum": ["up", "down"]},
                  "degrees": {"type": "number", "description": "30 per step."}}, ["direction"]),
            tool("ChangePosture", "Stand up or crouch.",
                 {"posture": {"type": "string", "enum": ["stand", "crouch"]}}, ["posture"]),
            tool("Pick", "Pick up an object.", obj, ["object"]),
            tool("Place", "Put the held object into a receptacle.", rec, ["receptacle"]),
            tool("ChangeState", "Change an object's state.",
                 {**obj, "state": {"type": "string",
                                   "enum": ["open", "close", "on", "off", "slice", "clean",
                                            "fill", "empty", "break"]}},
                 ["object", "state"]),
            tool("Manipulate", "Apply a low-level interaction to an object.",
                 {**obj, "action": {"type": "string", "description": "AI2-THOR action name."}},
                 ["action", "object"]),
            tool("EndTask", "Finish the episode (task believes complete)."),
        ]

    # --------------------------------------------------------------- helpers
    def _index_objects(self, metadata: dict) -> None:
        self.objects_by_id = {}
        for o in metadata.get("objects", []):
            self.objects_by_id[o["objectId"]] = o
        for o in metadata.get("inventoryObjects") or []:
            self.objects_by_id[o["objectId"]] = o

    def _resolve(self, name_or_id: str) -> str:
        text = str(name_or_id or "").strip()
        if not text:
            return ""
        if text in self.objects_by_id:
            return text
        lowered = text.lower()
        exact = [oid for oid, o in self.objects_by_id.items()
                 if str(o.get("objectType", "")).lower() == lowered]
        if exact:
            return sorted(exact, key=lambda oid: float(self.objects_by_id[oid].get("distance") or 1e9))[0]
        prefix = [oid for oid, o in self.objects_by_id.items()
                  if str(o.get("objectType", "")).lower().startswith(lowered)]
        if prefix:
            return sorted(prefix, key=lambda oid: float(self.objects_by_id[oid].get("distance") or 1e9))[0]
        return text

    def _observation(self, event, feedback: str) -> dict:
        return {
            "instruction": self.task["instruction"],
            "images": [encode_frame(event.frame)],
            "image_roles": ["current_view"],
            "text_feedback": feedback,
            "public_metadata": {"last_action_success": bool(event.metadata.get("lastActionSuccess", False))},
        }

    def _step_thor(self, action: dict):
        return self.c.step(action)

    def _repeat(self, base_action: str, times: int, **kwargs):
        event = None
        for _ in range(max(0, times)):
            event = self._step_thor(dict(action=base_action, **kwargs))
            if not event.metadata.get("lastActionSuccess", False):
                break
        return event

    # --------------------------------------------------------------- episode
    def reset(self, task_record: dict) -> dict:
        self.task = task_record
        self.steps = 0
        self.done = False
        env = task_record["source_backend"]
        scene = task_record["scene_id_raw"] if env == "ai2thor" else "FloorPlan1"
        self.c = Controller(scene=scene, platform=CloudRendering,
                            width=640, height=480, quality="Medium")
        if env == "procthor":
            self.c.reset(scene=load_house(int(task_record["scene_id_raw"])))
        init_actions = task_record.get("init_actions") or []
        if not init_actions and task_record.get("source_path"):
            init_path = (ROOT / task_record["source_path"]).parent / "init.json"
            if init_path.exists():
                init_actions = (json.loads(init_path.read_text()) or {}).get("actions") or []
        for a in init_actions:
            if a == "Done":
                break
            self.c.step(a)
        event = self.c.step("Pass")
        self._index_objects(event.metadata)
        return self._observation(event, "Episode started. Use the tools to complete the instruction.")

    def step(self, action: str, parameters: dict) -> dict:
        if self.done:
            raise RuntimeError("episode already finished")
        self.steps += 1
        event = None
        error = ""
        try:
            event = self._dispatch(action, parameters)
        except ValueError as exc:
            error = str(exc)
        if event is None:
            event = self.c.last_event
            ok = False
            feedback = f"{action} -> failed ({error or 'no action executed'})"
        else:
            ok = bool(event.metadata.get("lastActionSuccess", False))
            err = event.metadata.get("errorMessage") or ""
            feedback = f"{action} -> {'ok' if ok else 'failed'}" + \
                       (f" ({err})" if err and not ok else "")
            self._index_objects(event.metadata)
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

    def _dispatch(self, action: str, parameters: dict):
        if action == "Move":
            direction = str(parameters.get("direction") or "").lower()
            base = {"ahead": "MoveAhead", "back": "MoveBack",
                    "left": "MoveLeft", "right": "MoveRight"}.get(direction)
            if base is None:
                raise ValueError(f"Move direction must be ahead/back/left/right, got {direction!r}")
            distance = float(parameters.get("distance") or GRID)
            cells = max(1, min(MOVE_LIMIT, round(distance / GRID)))
            return self._repeat(base, cells)
        if action == "Rotate":
            direction = str(parameters.get("direction") or "").lower()
            base = {"left": "RotateLeft", "right": "RotateRight"}.get(direction)
            if base is None:
                raise ValueError(f"Rotate direction must be left/right, got {direction!r}")
            degrees = float(parameters.get("degrees") or ROTATE_STEP)
            steps = max(1, min(ROTATE_LIMIT, round(degrees / ROTATE_STEP)))
            return self._repeat(base, steps)
        if action == "Tilt":
            direction = str(parameters.get("direction") or "").lower()
            base = {"up": "LookUp", "down": "LookDown"}.get(direction)
            if base is None:
                raise ValueError(f"Tilt direction must be up/down, got {direction!r}")
            degrees = float(parameters.get("degrees") or TILT_STEP)
            steps = max(1, min(3, round(degrees / TILT_STEP)))
            return self._repeat(base, steps)
        if action == "ChangePosture":
            posture = str(parameters.get("posture") or "").lower()
            if posture not in ("stand", "crouch"):
                raise ValueError(f"ChangePosture requires stand/crouch, got {posture!r}")
            return self._step_thor(dict(action=posture.capitalize()))
        if action == "Pick":
            return self._step_thor(dict(action="PickupObject",
                                        objectId=self._resolve(parameters.get("object", ""))))
        if action == "Place":
            return self._step_thor(dict(action="PutObject",
                                        receptacleObjectId=self._resolve(parameters.get("receptacle", ""))))
        if action == "ChangeState":
            state = str(parameters.get("state") or "").lower()
            thor_action = STATE_ACTIONS.get(state)
            if thor_action is None:
                raise ValueError(f"ChangeState state must be one of {sorted(STATE_ACTIONS)}, got {state!r}")
            kwargs = {"objectId": self._resolve(parameters.get("object", ""))}
            if state == "fill":
                kwargs["fillLiquid"] = "water"
            return self._step_thor(dict(action=thor_action, **kwargs))
        if action == "Manipulate":
            thor_action = str(parameters.get("action") or "")
            if thor_action not in MANIPULATE_WHITELIST:
                raise ValueError(f"Manipulate action {thor_action!r} is not in the allowed set")
            return self._step_thor(dict(action=thor_action,
                                        objectId=self._resolve(parameters.get("object", ""))))
        if action == "EndTask":
            self.done = True
            return None
        raise ValueError(f"unknown SpatialWorld action {action!r}")

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
        return {"success": success, "success_logic": logic,
                "condition_results": results,
                "physical_task_id": self.task.get("physical_task_id")}

    def close(self) -> None:
        if self.c is not None:
            try:
                self.c.stop()
            except Exception:
                pass
            self.c = None


if __name__ == "__main__":
    serve(SpatialWorldWorker())
