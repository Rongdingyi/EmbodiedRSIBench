"""M2 acceptance: RSI core is decoupled from the legacy OpenETA bridge."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

RSI_DIR = PROJECT / "benchmark" / "rsi"
AGENT_DIR = PROJECT / "benchmark" / "agent"


def test_rsi_package_has_no_openeta_bridge_imports():
    hits = []
    for path in sorted(RSI_DIR.glob("*.py")):
        text = path.read_text()
        if "openeta_bridge" in text:
            hits.append(path.name)
    assert hits == [], f"legacy coupling remains: {hits}"


def test_canonical_agent_has_no_legacy_imports():
    hits = []
    for path in sorted(AGENT_DIR.glob("*.py")):
        if path.name == "freeze.py":
            continue  # holds the marker list itself
        text = path.read_text()
        for marker in ("openeta_bridge", "agent.runtime"):
            if marker in text:
                hits.append(f"{path.name}:{marker}")
    assert hits == [], f"canonical agent coupling: {hits}"


def test_legacy_shim_still_re_exports_context():
    from benchmark.openeta_bridge.context_injection import (  # noqa: F401
        MAX_RSI_INJECTION_TOKENS,
        RSIInjection,
        count_tokens,
        make_injection,
        wrap_guidance,
    )
    from benchmark.rsi import context as canonical

    injection = make_injection("hello")
    assert injection.context_text == canonical.make_injection("hello").context_text
    assert MAX_RSI_INJECTION_TOKENS == canonical.MAX_RSI_INJECTION_TOKENS
    assert count_tokens("hello") == canonical.count_tokens("hello")


def test_rsi_token_counter_provenance_is_honest():
    from benchmark.rsi.context import count_tokens

    assert count_tokens("x" * 400) > 0  # tiktoken or char/4 estimate
