"""Unified RSI context injection (guide sections 15/16).

Every RSI method must use this exact channel: a single runtime startup fact
named `persistent_experience_guidance` rendered inside the same wrapper block.
The planner system prompt, tool schemas and task instruction are untouched.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

OPEN_TAG = "[PERSISTENT EXPERIENCE GUIDANCE]"
CLOSE_TAG = "[/PERSISTENT EXPERIENCE GUIDANCE]"
MAX_RSI_INJECTION_TOKENS = 8192


@dataclass
class RSIInjection:
    context_text: str = ""
    provenance_ids: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "context_text": self.context_text,
            "provenance_ids": list(self.provenance_ids),
            "estimated_tokens": self.estimated_tokens,
            "truncated": self.truncated,
        }


def count_tokens(text: str) -> int:
    """Token count using the pinned OpenETA tokenizer when available."""
    try:
        from agent.runtime.token_counting import estimate_text_tokens

        return int(estimate_text_tokens(text))
    except Exception:
        return max(1, len(text) // 4)


def wrap_guidance(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return f"{OPEN_TAG}\n{text}\n{CLOSE_TAG}"


def make_injection(context_text: str, provenance_ids: list[str] | None = None,
                   *, truncated: bool = False) -> RSIInjection:
    wrapped = wrap_guidance(context_text)
    tokens = count_tokens(wrapped) if wrapped else 0
    assert tokens <= MAX_RSI_INJECTION_TOKENS, (
        f"RSI injection {tokens} tokens exceeds the shared {MAX_RSI_INJECTION_TOKENS} budget")
    return RSIInjection(context_text=wrapped, provenance_ids=list(provenance_ids or []),
                        estimated_tokens=tokens, truncated=truncated)


def injection_hash(injection: RSIInjection) -> str:
    return hashlib.sha256(injection.context_text.encode()).hexdigest()[:24]


def log_injection(injection: RSIInjection, path: Path, *, episode_id: str, method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "episode_id": episode_id,
        "method": method,
        "injection_sha256": injection_hash(injection),
        "estimated_tokens": injection.estimated_tokens,
        "truncated": injection.truncated,
        "context_text": injection.context_text,
        "provenance_ids": injection.provenance_ids,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
