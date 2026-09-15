"""Central per-episode cost/pathway accounting (P1-6, guide section 33).

Every model request goes through the audited transport; this module keeps the
redacted request dump (P1-4) and all token categories in one place so that
metrics.json / token_usage.json are complete instead of partial.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

FORBIDDEN_KEYS = {
    "golden_actions", "goal_preds", "success_conditions", "target_pose",
    "physical_task_id", "transfer_pair_id", "retention_anchor_id",
    "private_eval_metadata", "target",
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    wall_s: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.wall_s += other.wall_s

    def to_dict(self) -> dict:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "calls": self.calls, "wall_s": round(self.wall_s, 3)}


@dataclass
class Accountant:
    """One episode's cost ledger."""

    episode_dir: Path | None = None
    planner: Usage = field(default_factory=Usage)
    rsi_injection_tokens: int = 0
    rsi_update: Usage = field(default_factory=Usage)
    worldmind_sidecar: Usage = field(default_factory=Usage)
    embedding_calls: int = 0
    simulator_steps: int = 0
    request_dumps: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------- recording
    def record_planner_exchange(self, *, url: str, body: dict, response: dict | None,
                                wall_s: float = 0.0, role: str = "planner") -> None:
        usage = (response or {}).get("usage") or {}
        self.planner.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.planner.output_tokens += int(usage.get("completion_tokens") or 0)
        self.planner.calls += 1
        self.planner.wall_s += wall_s
        self.record_request_dump(url=url, body=body, role=role)

    def record_request_dump(self, *, url: str, body: dict, role: str) -> None:
        redacted = _redact_body(body)
        self.request_dumps.append({
            "role": role,
            "url": url,
            "model": body.get("model"),
            "has_image": _has_image(body),
            "message_count": len(body.get("messages") or []),
            "message_roles": [m.get("role") for m in (body.get("messages") or [])
                              if isinstance(m, dict)],
            "redacted_body": redacted,
        })

    def record_injection(self, tokens: int) -> None:
        self.rsi_injection_tokens += int(tokens or 0)

    def record_rsi_update(self, *, usage: dict | None, wall_s: float = 0.0,
                          calls: int = 1) -> None:
        usage = usage or {}
        self.rsi_update.input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.rsi_update.output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        self.rsi_update.calls += int(calls)
        self.rsi_update.wall_s += wall_s

    def record_sidecar(self, *, usage: dict | None, wall_s: float = 0.0) -> None:
        usage = usage or {}
        self.worldmind_sidecar.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.worldmind_sidecar.output_tokens += int(usage.get("completion_tokens") or 0)
        self.worldmind_sidecar.calls += 1
        self.worldmind_sidecar.wall_s += wall_s

    def record_env_step(self, n: int = 1) -> None:
        self.simulator_steps += int(n)

    # ---------------------------------------------------------------- output
    def leakage_violations(self, extra_forbidden_values: list[str] | None = None) -> list[str]:
        hits: list[str] = []
        forbidden_values = [v for v in (extra_forbidden_values or []) if v]
        for index, dump in enumerate(self.request_dumps):
            _scan_keys(dump["redacted_body"], f"request[{index}]", hits)
            text = json.dumps(dump["redacted_body"], ensure_ascii=False)
            for value in forbidden_values:
                if value in text:
                    hits.append(f"request[{index}]: private value {value[:40]!r}")
        return hits

    def dump_path(self) -> Path:
        assert self.episode_dir is not None
        return self.episode_dir / "public_context_dump.jsonl"

    def write_dump(self) -> None:
        if self.episode_dir is None or not self.request_dumps:
            return
        self.dump_path().parent.mkdir(parents=True, exist_ok=True)
        with open(self.dump_path(), "w") as f:
            for record in self.request_dumps:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def to_dict(self) -> dict:
        return {
            "planner_input_tokens": self.planner.input_tokens,
            "planner_output_tokens": self.planner.output_tokens,
            "planner_calls": self.planner.calls,
            "planner_wall_s": round(self.planner.wall_s, 3),
            "rsi_injection_tokens": self.rsi_injection_tokens,
            "rsi_update_input_tokens": self.rsi_update.input_tokens,
            "rsi_update_output_tokens": self.rsi_update.output_tokens,
            "rsi_update_calls": self.rsi_update.calls,
            "rsi_update_wall_s": round(self.rsi_update.wall_s, 3),
            "worldmind_prediction_sidecar_tokens":
                self.worldmind_sidecar.input_tokens + self.worldmind_sidecar.output_tokens,
            "worldmind_prediction_sidecar_calls": self.worldmind_sidecar.calls,
            "embedding_calls": self.embedding_calls,
            "simulator_steps": self.simulator_steps,
        }


def _redact_body(body: dict) -> dict:
    """Replace inline image payloads with a marker; keep structure and text."""
    def walk(node):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if key == "image_url" and isinstance(value, dict):
                    url = str(value.get("url") or "")
                    out[key] = {"url": "<redacted-image>" if url.startswith("data:") else url}
                else:
                    out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, str) and node.startswith("data:image"):
            return "<redacted-image>"
        return node

    return walk(body)


def _has_image(body: dict) -> bool:
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _scan_keys(node, path: str, hits: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in FORBIDDEN_KEYS:
                hits.append(f"{path}/{key}")
            _scan_keys(value, f"{path}/{key}", hits)
    elif isinstance(node, list):
        for i, value in enumerate(node[:20]):
            _scan_keys(value, f"{path}[{i}]", hits)
