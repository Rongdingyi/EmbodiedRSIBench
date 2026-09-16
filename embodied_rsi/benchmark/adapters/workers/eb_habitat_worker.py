#!/usr/bin/env python
"""EB-Habitat simulator worker (embench env: Habitat-Sim 0.3.0 + habitat-lab 0.3.0).

Uses the official EmbodiedBench EBHabEnv with its 70 parameterized discrete
skills (`language_skill_set`). Runs with CWD inside the eb_habitat package so
that the ReplicaCAD `data/` assets resolve.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_DIR))
from worker_base import Worker, encode_frame, serve  # noqa: E402

ROOT = Path("/media/moxc/data/datasets/embodied/embodied_rsi_data")
EB = ROOT / "sources" / "embodiedbench"
EB_HAB = EB / "embodiedbench" / "envs" / "eb_habitat"
PICKLE_DIR = EB_HAB / "datasets"

sys.path.insert(0, str(EB))
os.chdir(EB_HAB)

from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv  # noqa: E402


def _slug(text: str) -> str:
    return "_".join(text.replace("/", " ").split())[:60]


def _canonical(obj):
    """Order-stable JSON value for signature hashing."""
    import json as _json

    def walk(node):
        if isinstance(node, dict):
            return {str(k): walk(node[k]) for k in sorted(node.keys(), key=str)}
        if isinstance(node, (list, tuple)):
            return [walk(v) for v in node]
        if hasattr(node, "tolist"):
            return walk(node.tolist())
        if hasattr(node, "item"):
            return walk(node.item())
        return node

    return _json.dumps(walk(obj), sort_keys=True, ensure_ascii=False)


def _attr(episode, name, default=None):
    """Episode accessor that works for both pickle dicts and dataset objects."""
    if isinstance(episode, dict):
        return episode.get(name, default)
    return getattr(episode, name, default)


def _scene_key(scene_id) -> str:
    import os as _os

    scene = _os.path.basename(str(scene_id or ""))
    return scene.replace(".scene_instance.json", "")


def _episode_signature(episode, *, with_position: bool) -> tuple:
    """(scene basename, canonical sampled_entities[, rounded start position])."""
    scene = _scene_key(_attr(episode, "scene_id", ""))
    sampled = _canonical(_attr(episode, "sampled_entities", None) or {})
    if not with_position:
        return (scene, sampled)
    position = _attr(episode, "start_position", None)
    try:
        rounded = tuple(round(float(x), 4) for x in list(position or []))
    except Exception:
        rounded = ()
    return (scene, sampled, rounded)


class EBHabitatWorker(Worker):
    source = "eb_habitat"

    schema_hash: int = 0
    match_rule: str = ""

    @staticmethod
    def _task_keys(task_record: dict):
        raw = task_record.get("raw_metadata") or {}
        scene_key = _scene_key(task_record.get("scene_id_raw"))
        sampled = raw.get("sampled_entities")
        sampled_key = _canonical(sampled) if sampled else ""
        position = raw.get("start_position")
        try:
            position_key = tuple(round(float(x), 4) for x in list(position or []))
        except Exception:
            position_key = ()
        return scene_key, sampled_key, position_key

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
        # EBHabEnv re-orders/renames its internal dataset, so the ONLY reliable
        # join key is the physical signature already stored in the release:
        # (scene basename, canonical sampled_entities, rounded start position).
        eps = self.env.dataset.episodes
        # The release's source_entry_index addresses the OFFICIAL PICKLE order, so
        # the pickle episode is the authoritative definition of this task instance.
        index = int(task_record["source_entry_index"])
        with open(PICKLE_DIR / f"{split}.pickle", "rb") as handle:
            pickle_eps = pickle.load(handle)["all_eps"]
        if not 0 <= index < len(pickle_eps):
            raise RuntimeError(f"EB-Habitat pickle index {index} out of range for {split}")
        reference = pickle_eps[index]
        reference_id = str(_attr(reference, "episode_id"))
        target = _episode_signature(reference, with_position=True)
        indices = [i for i, e in enumerate(eps)
                   if _episode_signature(e, with_position=True) == target]
        match_rule = "pickle_signature+pose"
        if len(indices) != 1:
            # fall back to signature without pose only if it stays unique
            target_np = _episode_signature(reference, with_position=False)
            indices_np = [i for i, e in enumerate(eps)
                          if _episode_signature(e, with_position=False) == target_np]
            if len(indices_np) == 1:
                indices, match_rule = indices_np, "pickle_signature_no_pose"
        if len(indices) != 1:
            reference_np = _episode_signature(reference, with_position=False)
            candidates = [
                {"index": i, "episode_id": _attr(eps[i], "episode_id"),
                 "instruction": str(_attr(eps[i], "instruction", ""))[:60],
                 "same_entities": _episode_signature(eps[i], with_position=False) == reference_np}
                for i in range(len(eps))
            ]
            raise RuntimeError(
                f"EB-Habitat dataset match failed in split {split!r} "
                f"(reference_episode={reference_id}, index={index}, "
                f"matches={len(indices)}, candidates={[c for c in candidates if c['same_entities']][:3]})")
        self.env._current_episode_num = indices[0]
        self.match_rule = match_rule
        obs = self.env.reset()
        # sanity: the loaded episode must share the reference physical signature
        loaded_sig = _episode_signature(self.env.episode_data, with_position=False)
        reference_sig = _episode_signature(reference, with_position=False)
        if loaded_sig != reference_sig:
            raise RuntimeError(
                f"EB-Habitat post-reset signature mismatch: loaded "
                f"{_attr(self.env.episode_data, 'episode_id')} vs reference "
                f"{reference_id}")
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
