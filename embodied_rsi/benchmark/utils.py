"""Small shared helpers (URL composition for OpenAI-compatible endpoints)."""
from __future__ import annotations


def normalize_base_url(url: str) -> str:
    """Return the endpoint root without a trailing /v1 or slash.

    Accepts both `https://host` and `https://host/v1`; every client in this repo
    appends `/v1/...` itself.
    """
    url = (url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")].rstrip("/")
    return url


def chat_completions_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/v1/chat/completions"
