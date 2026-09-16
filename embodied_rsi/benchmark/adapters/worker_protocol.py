"""Host side of the simulator-worker protocol.

Each source benchmark runs in its own conda/python environment (AI2-THOR 5,
AI2-THOR 2.1.0, Habitat-Sim) to avoid dependency conflicts; the host talks to it
over newline-delimited JSON on stdin/stdout. Images are sent as base64 PNG.

Protocol
--------
Request  (host -> worker):  {"id": int, "op": str, "args": {...}}
Response (worker -> host):  {"id": int, "ok": bool, "result": {...}, "error": str}

Ops: ping, reset, step, private_evaluate, close.
"""
from __future__ import annotations

import base64
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


class WorkerError(RuntimeError):
    pass


class SimulatorWorker:
    """Spawns and drives one simulator worker subprocess."""

    def __init__(self, script: Path, python: str, env: dict | None = None,
                 cwd: Path | None = None, timeout_s: float = 600.0):
        self.script = Path(script)
        self.python = python
        self.timeout_s = timeout_s
        full_env = os.environ.copy()
        full_env.update(env or {})
        self.proc = subprocess.Popen(
            [python, str(self.script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=False, cwd=str(cwd) if cwd else None, env=full_env, bufsize=0,
            # own process group so the simulator child (Unity) dies with the
            # worker even when the worker is killed on timeout (leak guard)
            start_new_session=True,
        )
        self._id = 0
        self._lines: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._read_lines, daemon=True)
        self._reader.start()

    def _read_lines(self) -> None:
        try:
            for line in self.proc.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)  # EOF marker

    # ------------------------------------------------------------------ rpc
    def call(self, op: str, timeout_s: float | None = None, **args) -> dict:
        if self.proc.poll() is not None:
            raise WorkerError(f"worker exited early (code {self.proc.returncode}): "
                              f"{self._drain_stderr()[:500]}")
        self._id += 1
        payload = json.dumps({"id": self._id, "op": op, "args": args}).encode() + b"\n"
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        deadline = time.monotonic() + (timeout_s or self.timeout_s)
        try:
            line = self._lines.get(timeout=max(0.1, deadline - time.monotonic()))
        except queue.Empty:
            self.kill()
            raise WorkerError(
                f"worker timed out after {timeout_s or self.timeout_s:.0f}s on {op}") from None
        if line is None:
            raise WorkerError(f"worker closed stdout on {op}: {self._drain_stderr()[:500]}")
        resp = json.loads(line.decode())
        if resp.get("id") != self._id:
            raise WorkerError(f"protocol id mismatch on {op}: {resp}")
        if not resp.get("ok"):
            raise WorkerError(f"{op} failed: {resp.get('error')}")
        return resp.get("result", {})

    def _kill_process_group(self, sig: int) -> None:
        import signal

        try:
            os.killpg(os.getpgid(self.proc.pid), sig)
        except Exception:
            try:
                self.proc.send_signal(signal.Signals(sig))
            except Exception:
                pass

    def kill(self) -> None:
        """Kill the worker AND its simulator children (Unity) via the group."""
        import signal

        self._kill_process_group(signal.SIGKILL)
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def _drain_stderr(self) -> str:
        try:
            if self.proc.stderr:
                return self.proc.stderr.read1(4096).decode(errors="ignore")  # type: ignore[attr-defined]
        except Exception:
            pass
        return ""

    # ---------------------------------------------------------------- helpers
    def ping(self) -> dict:
        return self.call("ping")

    def reset(self, task_record: dict) -> dict:
        return self.call("reset", task_record=task_record)

    def step(self, action: str, parameters: dict) -> dict:
        return self.call("step", action=action, parameters=parameters)

    def private_evaluate(self) -> dict:
        return self.call("private_evaluate")

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.call("close")
        except Exception:
            pass
        finally:
            import signal

            try:
                self._kill_process_group(signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self._kill_process_group(signal.SIGKILL)
                    self.proc.wait(timeout=5)
                except Exception:
                    pass


# ---------------------------------------------------------------- image codecs
def decode_png_b64(b64: str):
    """base64 PNG -> HxWx3 uint8 numpy array."""
    import numpy as np
    from PIL import Image

    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def encode_png_b64(frame) -> str:
    """HxWx3 uint8 numpy array -> base64 PNG."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
