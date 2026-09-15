#!/usr/bin/env python
"""TVRBench simulator worker (runs in the RoboTwin env: AI2-THOR 5 CloudRendering).

Public input: current RGB + target RGB (TVR protocol).
Private (never returned): target pose, physical_task_id, pair ids.
Official evaluator: tvrbench/evaluation/metrics.py evaluate_episode().
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_DIR))
from worker_base import Worker, encode_frame, serve  # noqa: E402

ROOT = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data")
TVR = ROOT / "sources" / "tvrbench"
sys.path.insert(0, str(TVR))

from tvrbench.env.thor_env import ThorEnv  # noqa: E402
from tvrbench.evaluation.metrics import evaluate_episode  # noqa: E402

PROCTHOR_JSONL = TVR / "data" / "procthor-10k" / "train.jsonl.gz"

ACTION_NAMES = ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
                "RotateLeft", "RotateRight", "LookUp", "LookDown", "Stop"]

MAX_STEPS = 100


def load_house(index: int) -> dict:
    with gzip.open(PROCTHOR_JSONL, "rt") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise KeyError(index)


class TVRWorker(Worker):
    source = "tvr"

    def __init__(self) -> None:
        self.env = None
        self.task = None
        self.trajectory: list[dict] = []
        self.stopped = False
        self.steps = 0

    def actions(self) -> list[dict]:
        return [{
            "name": action,
            "description": f"TVRBench viewpoint action: {action}.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        } for action in ACTION_NAMES]

    # -------------------------------------------------------------- episode
    def reset(self, task_record: dict) -> dict:
        self.task = task_record
        self.stopped = False
        self.steps = 0
        dataset = task_record["dataset"]
        scene = task_record["scene"]
        start = task_record["start"]

        self.env = ThorEnv(scene=scene if dataset == "ithor" else "FloorPlan1",
                           width=640, height=480, quality="Medium")
        if dataset == "procthor":
            self.env.load_scene(load_house(int(scene.split("_")[1])))

        start_frame = self.env.reset(position=start["position"], rotation_y=start["rotation_y"],
                                     horizon=start["horizon"])

        # render the target viewpoint image (public per TVR protocol)
        target = task_record["target"]
        target_frame = self.env.controller.step(
            action="Teleport", position=target["position"],
            rotation={"x": 0, "y": target["rotation_y"], "z": 0},
            horizon=target["horizon"], standing=True, forceAction=True,
        ).frame
        # return the agent to the start pose for the actual episode
        self.env.reset(position=start["position"], rotation_y=start["rotation_y"],
                       horizon=start["horizon"])
        self.trajectory = [self.env.get_state()]

        return {
            "instruction": task_record.get("instruction") or
                           "Reproduce the target viewpoint: match the target image's position and camera angle.",
            "images": [encode_frame(start_frame), encode_frame(target_frame)],
            "image_roles": ["current_view", "target_view"],
            "text_feedback": "Episode started. Move and look to match the target image, then Stop.",
            "public_metadata": {"action_space": ACTION_NAMES},
        }

    def step(self, action: str, parameters: dict) -> dict:
        if action not in ACTION_NAMES:
            raise ValueError(f"illegal TVR action {action!r}")
        if self.stopped:
            raise RuntimeError("episode already stopped")
        frame, success = self.env.step(action)
        self.steps += 1
        if action == "Stop":
            self.stopped = True
        self.trajectory.append(self.env.get_state())
        terminated = self.stopped
        truncated = self.steps >= MAX_STEPS
        return {
            "observation": {
                "instruction": self.task.get("instruction") or
                               "Reproduce the target viewpoint: match the target image's position and camera angle.",
                "images": [encode_frame(frame)],
                "image_roles": ["current_view"],
                "text_feedback": f"{action} -> {'ok' if success else 'blocked'}",
                "public_metadata": {"action_space": ACTION_NAMES},
            },
            "action_success": bool(success),
            "reward_public": None,
            "terminated": terminated,
            "truncated": truncated,
            "public_feedback": f"{action} -> {'ok' if success else 'blocked'}",
        }

    def private_evaluate(self) -> dict:
        result = evaluate_episode(self.trajectory, self.task["target"], self.stopped)
        result["physical_task_id"] = self.task.get("physical_task_id")
        return result

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.controller.stop()
            except Exception:
                pass
            self.env = None


if __name__ == "__main__":
    serve(TVRWorker())
