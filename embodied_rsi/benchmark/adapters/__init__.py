"""Host-side benchmark adapters backed by simulator worker subprocesses.

Each adapter knows its worker script, python interpreter and environment; it
converts worker JSON into PublicObservation / StepOutcome and never exposes
private fields (target poses, success conditions, golden actions) to callers.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from benchmark.adapters.base import BenchmarkEnvAdapter, PublicObservation, StepOutcome
from benchmark.adapters.worker_protocol import SimulatorWorker, decode_png_b64

PROJECT = Path(__file__).resolve().parents[2]
WORKERS = PROJECT / "benchmark" / "adapters" / "workers"
ROBOTWIN_PY = "/data/users/rongdingyi/miniconda3/envs/RoboTwin/bin/python"
BENCH_AVP_PY = "/data/users/rongdingyi/miniconda3/envs/bench_avp/bin/python"
EMBENCH_PY = "/data/users/rongdingyi/miniconda3/envs/embench/bin/python"
VK_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"


def _pick_display() -> str:
    """EB-ALFRED's 2018 Unity build must present through the physical X server."""
    current = os.environ.get("DISPLAY")
    for disp in ([current] if current else []) + [":0", ":1", ":2"]:
        if not disp:
            continue
        env = os.environ.copy()
        env["DISPLAY"] = disp
        try:
            if subprocess.call(["xdpyinfo"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, env=env, timeout=5) == 0:
                return disp
        except Exception:
            continue
    return current or ":1"


class SubprocessAdapter(BenchmarkEnvAdapter):
    """Shared plumbing for worker-backed adapters."""

    worker_script: str = ""
    worker_python: str = ""
    worker_env: dict = {}
    worker_cwd: Path | None = None

    op_timeout_s = 600.0

    def __init__(self) -> None:
        self.worker = SimulatorWorker(
            script=WORKERS / self.worker_script,
            python=self.worker_python,
            env=self.worker_env,
            cwd=self.worker_cwd,
            timeout_s=self.op_timeout_s,
        )
        self._tools: list[dict] = []
        self._task: dict | None = None

    # ------------------------------------------------------------------ hooks
    def _task_record_for_worker(self, task_record: dict) -> dict:
        return task_record

    # ------------------------------------------------------------- lifecycle
    def reset(self, task_record: dict) -> PublicObservation:
        self._task = task_record
        result = self.worker.reset(self._task_record_for_worker(task_record))
        self._tools = self.worker.call("actions")["actions"]
        return self._public(result)

    def build_tool_specs(self, task_record: dict) -> list[dict]:
        if not self._tools:
            self._tools = self.worker.call("actions")["actions"]
        return self._tools

    def step(self, action_name: str, parameters: dict) -> StepOutcome:
        result = self.worker.step(action_name, parameters)
        return StepOutcome(
            observation=self._public(result["observation"]),
            action_success=result.get("action_success"),
            reward_public=result.get("reward_public"),
            terminated=bool(result.get("terminated")),
            truncated=bool(result.get("truncated")),
            public_feedback=result.get("public_feedback", ""),
        )

    def private_evaluate(self) -> dict:
        return self.worker.private_evaluate()

    def close(self) -> None:
        self.worker.close()

    # ------------------------------------------------------------------ util
    @staticmethod
    def _public(payload: dict) -> PublicObservation:
        images = [decode_png_b64(b64) for b64 in payload.get("images") or []]
        metadata = dict(payload.get("public_metadata") or {})
        if payload.get("image_roles"):
            metadata["image_roles"] = list(payload["image_roles"])
        return PublicObservation(
            instruction=payload.get("instruction", ""),
            images=images,
            text_feedback=payload.get("text_feedback", ""),
            public_metadata=metadata,
        )


# --------------------------------------------------------------------- sources
class TVRAdapter(SubprocessAdapter):
    source = "tvr"
    worker_script = "tvr_worker.py"
    worker_python = ROBOTWIN_PY

    def _task_record_for_worker(self, task_record: dict) -> dict:
        private = task_record["private_eval_metadata"]
        return {
            "dataset": task_record["raw_metadata"]["dataset"],
            "scene": task_record["raw_metadata"]["scene"],
            "start": task_record["raw_metadata"]["start"],
            "target": private["target"],
            "instruction": task_record["instruction"],
            "physical_task_id": task_record["physical_task_id"],
        }


class SpatialWorldAdapter(SubprocessAdapter):
    source = "spatialworld"
    worker_script = "spatialworld_worker.py"
    worker_python = ROBOTWIN_PY

    def _task_record_for_worker(self, task_record: dict) -> dict:
        private = task_record["private_eval_metadata"]
        raw = task_record["raw_metadata"]
        return {
            "instruction": task_record["instruction"],
            "source_backend": task_record["source_backend"],
            "scene_id_raw": task_record["scene_id_raw"],
            "source_path": task_record["source_path"],
            "success_conditions": private.get("success_conditions") or [],
            "success_logic": raw.get("success_logic") or "OR",
            "target_object_types": raw.get("target_object_types") or [],
            "physical_task_id": task_record["physical_task_id"],
        }


class AsgardAdapter(SubprocessAdapter):
    source = "asgard"
    worker_script = "asgard_worker.py"
    worker_python = ROBOTWIN_PY

    def _task_record_for_worker(self, task_record: dict) -> dict:
        private = task_record["private_eval_metadata"]
        raw = task_record["raw_metadata"]
        return {
            "name": task_record["source_task_id"],
            "task_description": raw.get("task_description") or task_record["instruction"],
            "scene": raw.get("scene") or task_record["scene_id_raw"],
            "plan_type": raw.get("plan_type") or "generated",
            "task_failed": raw.get("task_failed", False),
            "initial_pose": raw.get("initial_pose"),
            "setup_actions": private.get("setup_actions") or raw.get("setup_actions") or [],
            "object_setup": private.get("object_setup") or raw.get("object_setup") or {},
            "randomization": raw.get("randomization") or {"seed": 0},
            "goal": private.get("goal") or raw.get("goal") or {},
            "physical_task_id": task_record["physical_task_id"],
        }


class EBAlfredAdapter(SubprocessAdapter):
    source = "eb_alfred"
    worker_script = "eb_alfred_worker.py"
    worker_python = BENCH_AVP_PY

    def __init__(self) -> None:
        self.worker_env = {"DISPLAY": _pick_display(), "VK_ICD_FILENAMES": VK_ICD}
        super().__init__()

    def _task_record_for_worker(self, task_record: dict) -> dict:
        return {
            "source_split": task_record["source_split"],
            "source_entry_index": task_record["source_entry_index"],
            "instruction": task_record["instruction"],
            "physical_task_id": task_record["physical_task_id"],
        }


class EBHabitatAdapter(SubprocessAdapter):
    source = "eb_habitat"
    worker_script = "eb_habitat_worker.py"
    worker_python = EMBENCH_PY

    def _task_record_for_worker(self, task_record: dict) -> dict:
        return {
            "source_split": task_record["source_split"],
            "source_entry_index": task_record["source_entry_index"],
            "source_task_id": task_record["source_task_id"],
            "instruction": task_record["instruction"],
            "physical_task_id": task_record["physical_task_id"],
            # physical-signature join fields (P0-7)
            "scene_id_raw": task_record.get("scene_id_raw"),
            "raw_metadata": task_record.get("raw_metadata") or {},
        }


ADAPTERS = {
    "EB-ALFRED": EBAlfredAdapter,
    "EB-Habitat": EBHabitatAdapter,
    "SpatialWorld": SpatialWorldAdapter,
    "AsgardBench": AsgardAdapter,
    "TVRBench": TVRAdapter,
}


def make_adapter(source_dataset: str) -> SubprocessAdapter:
    return ADAPTERS[source_dataset]()
