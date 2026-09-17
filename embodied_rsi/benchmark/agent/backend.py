"""Thin, transparent OpenAI-compatible backend for the canonical agent.

The backend only does: HTTP/API calls, retry/backoff, token-usage extraction
and request auditing. It never selects actions and never uses provider-specific
function calling (the canonical action schema is prompt + validator controlled).
"""
from __future__ import annotations

import os
import time

import requests

from benchmark.utils import chat_completions_url, normalize_base_url


class PlannerProtocolError(RuntimeError):
    """Planner protocol failure (API failure or exhausted validation retries)."""


class CanonicalVLMBackend:
    def __init__(self, config, accountant=None, *, timeout_s: float = 180.0,
                 max_http_retries: int = 3, backoff_s: float = 2.0):
        self.config = config
        self.accountant = accountant
        self.timeout_s = timeout_s
        self.max_http_retries = max_http_retries
        self.backoff_s = backoff_s
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
        base = normalize_base_url(os.environ.get("DEEPSEEK_BASE_URL", ""))
        self.base_url = base
        self.url = chat_completions_url(base)
        self._api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    def complete(
        self,
        *,
        system_prompt: str,
        user_text: str,
        images: list[dict],
        role: str = "canonical_planner",
    ) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [*(images or []), {"type": "text", "text": user_text}],
            },
        ]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.max_output_tokens),
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_http_retries + 1):
            t0 = time.time()
            try:
                response = requests.post(self.url, headers=headers, json=body,
                                         timeout=self.timeout_s)
                wall_s = time.time() - t0
                if response.status_code >= 400:
                    raise PlannerProtocolError(
                        f"planner HTTP {response.status_code}: {response.text[:300]}")
                payload = response.json()
                if self.accountant is not None:
                    self.accountant.record_planner_exchange(
                        url=self.url, body=body, response=payload, wall_s=wall_s, role=role)
                choices = payload.get("choices") or []
                content = ""
                if choices and isinstance(choices[0], dict):
                    content = str((choices[0].get("message") or {}).get("content") or "")
                return {"content": content, "usage": payload.get("usage") or {},
                        "model": payload.get("model") or self.model}
            except PlannerProtocolError as exc:
                last_error = exc
                if "HTTP 4" in str(exc) and "HTTP 429" not in str(exc):
                    break  # client errors other than rate limiting do not heal by retrying
            except Exception as exc:  # noqa: BLE001 - transport errors are retried
                last_error = exc
            if attempt < self.max_http_retries:
                time.sleep(self.backoff_s * attempt)
        raise PlannerProtocolError(f"planner request failed: {last_error}")
