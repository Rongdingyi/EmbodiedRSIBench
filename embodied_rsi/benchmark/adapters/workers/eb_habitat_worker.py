#!/usr/bin/env python
"""EB-Habitat simulator worker (embench env: Habitat-Sim 0.3.0 + habitat-lab 0.3.0).

Uses the official EmbodiedBench EBHabEnv with its 70 parameterized discrete
skills (`language_skill_set`). Runs with CWD inside the eb_habitat package so
that the ReplicaCAD `data/` assets resolve.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_DIR))
from worker_base import Worker, encode_frame, serve  # noqa: E402

ROOT = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data")
EB = ROOT / "sources" / "embodiedbench"
EB_HAB = EB / "embodiedbench" / "envs" / "eb_habitat"

sys.path.insert(0, str(EB))
os.chdir(EB_HAB)

from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv  # noqa: E402


def _slug(text: str) -> str:
    return "_".join(text.replace("/", " ").split())[:60]


class EBHabitatWorker(Worker):
    source = "eb_habitat"

    schema_hash: int = 0

    def __init__(self) -> None:
        self.env = None
        self.task = None
        self.steps = 0
        self.actions_text: list[str] = []
        self._tool_to_text: dict[str, str] = {}
        self._last_info: dict = {}

    def actions(self) -> list[dict]:
        if not self.actions_text:
            raise RuntimeError(
                "EB-Habitat action schema is not available yet: the worker must "
                "reset() an episode before exposing tools")
        tools = []
        self._tool_to_text = {}
        for i, text in enumerate(self.actions_text):
            name = f"skill_{i:02d}_{_slug(text)}"
            self._tool_to_text[name] = text
            tools.append({
                "name": name,
                "description": f"Habitat rearrangement skill: {text}",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            })
        return tools

    def _observation(self, obs, info: dict | None) -> dict:
        frame = obs["head_rgb"]
        success = None
        feedback = "Episode running."
        if info:
            ok = info.get("was_prev_action_invalid")
            feedback = info.get("env_feedback") or ("Action executed." if not ok else "Invalid action.")
        return {
            "instruction": self.task["instruction"],
            "images": [encode_frame(frame)],
            "image_roles": ["current_view"],
            "text_feedback": feedback,
            "public_metadata": {
                "available_skills": [t["description"] for t in self.actions()][:80],
            },
        }

    def reset(self, task_record: dict) -> dict:
        self.task = task_record
        self.steps = 0
        split = task_record["source_split"]
        expected_episode_id = str(task_record["source_task_id"])
        self.env = EBHabEnv(eval_set=split, resolution=320)
        # EBHabEnv re-orders/normalizes its internal dataset, so resolve the
        # episode by instruction text first (join key used by the release build),
        # then by episode_id.
        eps = self.env.dataset.episodes
        indices = []
        if task_record.get("instruction"):
            indices = [i for i, e in enumerate(eps)
                       if getattr(e, "instruction", None) == task_record["instruction"]]
        if not indices and expected_episode_id:
            ids = [str(getattr(ep, "episode_id", "")) for ep in eps]
            if expected_episode_id in ids:
                indices = [ids.index(expected_episode_id)]
        if not indices:
            raise RuntimeError(
                f"EB-Habitat episode not found in split {split!r}: "
                f"instruction={task_record.get('instruction')!r} "
                f"episode_id={expected_episode_id}")
        self.env._current_episode_num = indices[0]
        obs = self.env.reset()
        actual = str(getattr(self.env.episode_data, "episode_id", ""))
        if expected_episode_id and actual and actual != expected_episode_id:
            raise RuntimeError(
                f"EB-Habitat episode mismatch after reset: got {actual}, "
                f"expected {expected_episode_id}")
        self.actions_text = list(self.env.language_skill_set)
        if len(self.actions_text) != 70:
            raise RuntimeError(
                f"EB-Habitat canonical skill schema drift: expected 70 skills, "
                f"got {len(self.actions_text)}")
        self.schema_hash = hash(tuple(self.actions_text)) & 0xFFFFFFFF
        self._last_info = {}
        return self._observation(obs, None)

    def step(self, action: str, parameters: dict) -> dict:
        if action not in self._tool_to_text:
            raise ValueError(f"illegal EB-Habitat skill {action!r}")
        self.steps += 1
        text = self._tool_to_text[action]
        idx = self.env.language_skill_set.index(text)
        obs, _reward, done, info = self.env.step(idx)
        self._last_info = info or {}
        return {
            "observation": self._observation(obs, info),
            "action_success": not bool(self._last_info.get("was_prev_action_invalid", False)),
            "reward_public": None,
            "terminated": bool(done),
            "truncated": False,
            "public_feedback": self._last_info.get("env_feedback", ""),
        }

    def private_evaluate(self) -> dict:
        info = dict(self._last_info)
        success = None
        for key, value in info.items():
            if "success" in key.lower() and isinstance(value, (int, float, bool)):
                success = bool(value)
                break
        partial = None
        for key, value in info.items():
            if "percent" in key.lower() and isinstance(value, (int, float)):
                partial = float(value)
                break
        return {
            "success": success,
            "partial": partial,
            "measures": {k: v for k, v in info.items()
                         if isinstance(v, (int, float, bool, str))},
            "physical_task_id": self.task.get("physical_task_id"),
        }

    def close(self) -> None:
        self.env = None


if __name__ == "__main__":
    serve(EBHabitatWorker())
