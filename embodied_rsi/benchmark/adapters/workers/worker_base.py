"""Shared JSON-line worker base for simulator subprocesses.

Workers run inside the simulator's own python environment and therefore only
use stdlib + numpy + PIL + the source benchmark code. They never import the
host project.

A worker subclass implements:

    actions() -> list[dict]           # public tool specs (name/description/parameters)
    reset(task_record) -> dict        # public observation
    step(action, parameters) -> dict  # step outcome
    private_evaluate() -> dict        # official evaluator result (never public)
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import traceback

# Keep the protocol channel clean: everything the source benchmark prints to
# stdout/stderr is redirected to the parent's stderr pipe, while JSON responses
# use a duplicate of the original fd 1 (the parent's stdout pipe).
_PROTOCOL_OUT = None


def _setup_protocol_channel() -> "io.TextIOWrapper":
    global _PROTOCOL_OUT
    if _PROTOCOL_OUT is None:
        saved_fd = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        _PROTOCOL_OUT = os.fdopen(saved_fd, "w", buffering=1)
    return _PROTOCOL_OUT


class Worker:
    source = "unknown"

    def actions(self) -> list[dict]:
        raise NotImplementedError

    def reset(self, task_record: dict) -> dict:
        raise NotImplementedError

    def step(self, action: str, parameters: dict) -> dict:
        raise NotImplementedError

    def private_evaluate(self) -> dict:
        return {"success": None, "detail": "no evaluator implemented"}

    def close(self) -> None:
        pass


def encode_frame(frame) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def decode_frame(b64: str):
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def serve(worker: Worker) -> None:
    out = _setup_protocol_channel()
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line.decode())
        rid = req.get("id")
        op = req.get("op")
        args = req.get("args") or {}
        try:
            if op == "ping":
                result = {"source": worker.source, "ok": True}
            elif op == "actions":
                result = {"actions": worker.actions()}
            elif op == "reset":
                result = worker.reset(args["task_record"])
            elif op == "step":
                result = worker.step(args["action"], args.get("parameters") or {})
            elif op == "private_evaluate":
                result = worker.private_evaluate()
            elif op == "close":
                worker.close()
                result = {}
            else:
                raise ValueError(f"unknown op {op!r}")
            response = {"id": rid, "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            response = {
                "id": rid,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
        out.write(json.dumps(response) + "\n")
        out.flush()
