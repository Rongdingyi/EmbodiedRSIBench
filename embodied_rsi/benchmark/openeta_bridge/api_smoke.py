#!/usr/bin/env python
"""DeepSeek official API smoke: text, vision, and pinned-OpenETA backend compat.

Runs inside the OpenETA uv environment (imports pinned `agent.backends.planner`).
Every outgoing request body is captured and audited:
  - URL must be https://api.deepseek.com/v1/chat/completions
  - model must be deepseek-flash
  - chat_template_kwargs must NOT appear (guide section 3.2)
  - the vision request must carry the image inside a user message
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OPENETA = PROJECT / "external" / "OpenETA"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(OPENETA))

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
EXPECTED_URL = f"{BASE_URL.rstrip('/')}/v1/chat/completions"

AUDIT_BODIES: list[dict] = []


def _audited_transport(url, body, headers, timeout_s):
    from agent.backends.planner import _post_json

    AUDIT_BODIES.append({"url": url, "body": body})
    return _post_json(url, body, headers, timeout_s)


def _config():
    from agent.backends.planner import OpenAICompatiblePlannerBackendConfig

    return OpenAICompatiblePlannerBackendConfig(
        provider="openai-compatible",
        model=MODEL,
        api_base=BASE_URL,
        api_key=API_KEY,
        timeout_s=180,
        max_attempts=3,
        temperature=0.0,
        max_tokens=8192,
        use_json_response_format=False,
        enable_thinking=None,          # must stay None: no chat_template_kwargs
        enable_vision=True,
        max_vision_images=2,
    )


def _last_audit(index: int) -> dict:
    body = AUDIT_BODIES[index]["body"]
    url = AUDIT_BODIES[index]["url"]
    assert url == EXPECTED_URL, f"unexpected URL {url}"
    assert body.get("model") == MODEL, f"unexpected model {body.get('model')}"
    assert "chat_template_kwargs" not in body, "chat_template_kwargs leaked into DeepSeek request"
    return AUDIT_BODIES[index]


def text_smoke() -> dict:
    from agent.backends.planner import _post_json

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Return exactly: DEEPSEEK_FLASH_TEXT_OK"}],
        "max_tokens": 64,
    }
    response = _post_json(EXPECTED_URL,
                          body,
                          {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                          180)
    content = response["choices"][0]["message"]["content"]
    return {"requested_model": MODEL, "returned_model": response.get("model"),
            "content": content, "usage": response.get("usage", {})}


def vision_smoke(image_path: Path) -> dict:
    import base64

    from agent.backends.planner import _post_json

    b64 = base64.b64encode(image_path.read_bytes()).decode()
    body = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the visible scene in one short sentence."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            ],
        }],
        "max_tokens": 2048,
    }
    assert "chat_template_kwargs" not in body
    response = _post_json(EXPECTED_URL,
                          body,
                          {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                          180)
    message = response["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    if not content:  # thinking mode may exhaust the cap inside reasoning_content
        content = (message.get("reasoning_content") or "").strip()[-300:]
    return {"returned_model": response.get("model"), "content": content,
            "content_source": "content" if message.get("content") else "reasoning_content",
            "usage": response.get("usage", {})}


def backend_compat_smoke(image_path: Path) -> dict:
    """Drive the pinned OpenETA planner backend against DeepSeek."""
    from agent.backends.planner import (
        OpenAICompatiblePlannerBackend,
        PlannerBackendRequest,
    )

    backend = OpenAICompatiblePlannerBackend(_config(), transport=_audited_transport)
    start = len(AUDIT_BODIES)

    # Use the canonical pinned planner prompt + closed-loop XML contract so the
    # model emits the real wire format (nested <parameters>).
    from benchmark.openeta_bridge.freeze import planner_prompt

    system_prompt, prompt_hash = planner_prompt()

    tool_context = {
        "schema_version": "openeta.planner.tool_context.v1",
        "observation": {
            "task": "Find the mug on the table and pick it up.",
            "cameras": [{"name": "head_rgb", "role": "scene_primary"}],
            "robot": {},
        },
        "vision_image_paths": [str(image_path)],
        "available_tools": [
            {"name": "find", "description": "Locate an object.", "parameters": {"type": "object", "properties": {"object": {"type": "string"}}}},
            {"name": "pick_up", "description": "Pick up an object.", "parameters": {"type": "object", "properties": {"object": {"type": "string"}}}},
            {"name": "done", "description": "Finish the episode.", "parameters": {"type": "object", "properties": {}}},
        ],
    }
    request = PlannerBackendRequest(
        tool_context=tool_context,
        system_prompt=system_prompt,
        conversation_messages=[{"role": "user",
                                "content": "Observation ready. Choose the next tool call."}],
    )
    result = backend.decide(request)

    # Validate exactly the way ToolCallingPlanner does, with a minimal benchmark
    # tool registry holding the three dummy actions.
    from agent.runtime.planner import _decision_from_backend_result
    from agent.runtime.skills import build_default_skill_registry
    from agent.tools.registry import ToolRegistry, ToolSpec

    tools = ToolRegistry()
    for spec in tool_context["available_tools"]:
        tools.register(ToolSpec(name=spec["name"], description=spec["description"],
                                category="benchmark_action", parameters=spec["parameters"]),
                       handler=lambda context: None)
    decision, validation_errors = _decision_from_backend_result(
        result, tools=tools, skills=build_default_skill_registry(),
        tool_context=tool_context)
    audit = _last_audit(start) if len(AUDIT_BODIES) > start else None
    body = audit["body"] if audit else {}
    user_msgs = [m for m in body.get("messages", []) if m.get("role") == "user"]
    has_image = any(
        isinstance(m.get("content"), list)
        and any(part.get("type") == "image_url" for part in m["content"])
        for m in user_msgs
    )
    payload = result.payload
    return {
        "url": audit["url"] if audit else None,
        "request_model": body.get("model"),
        "no_chat_template_kwargs": "chat_template_kwargs" not in body,
        "image_in_user_message": bool(has_image),
        "status": str(result.status),
        "provider": result.provider,
        "model": result.model,
        "payload_kind": payload.get("kind") if isinstance(payload, dict) else type(payload).__name__,
        "payload": str(payload)[:1200],
        "planner_prompt_hash": prompt_hash,
        "parsed_action_type": getattr(decision, "action_type", None),
        "parsed_action": getattr(decision, "action", None),
        "validation_errors": list(validation_errors)[:3],
    }


def main() -> int:
    out_dir = PROJECT / "outputs" / "api_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(os.environ.get("API_SMOKE_IMAGE", out_dir / "sample_rgb.png"))

    checks: list[tuple[str, bool, str]] = []
    report: dict = {
        "base_url": BASE_URL,
        "requested_model": MODEL,
        "request_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": {},
    }

    if not API_KEY:
        print("[FAIL] DEEPSEEK_API_KEY is not set")
        return 1

    try:
        text = text_smoke()
        ok = bool(text["content"].strip())
        checks.append(("text smoke", ok, text["content"][:80]))
        report["text_smoke"] = text
        report["resolved_model"] = text.get("returned_model")
    except Exception as exc:  # noqa: BLE001
        checks.append(("text smoke", False, f"{type(exc).__name__}: {exc}"))
        report["text_smoke"] = {"error": str(exc)}

    try:
        if not image_path.exists():
            raise FileNotFoundError(f"sample RGB missing: {image_path}")
        vision = vision_smoke(image_path)
        ok = bool(vision["content"].strip())
        checks.append(("vision smoke", ok, vision["content"][:80]))
        report["vision_smoke"] = vision
    except Exception as exc:  # noqa: BLE001
        checks.append(("vision smoke", False, f"{type(exc).__name__}: {exc}"))
        report["vision_smoke"] = {"error": str(exc)}

    try:
        compat = backend_compat_smoke(image_path)
        ok = (
            compat["url"] == EXPECTED_URL
            and compat["request_model"] == MODEL
            and compat["no_chat_template_kwargs"]
            and compat["image_in_user_message"]
            and not compat["validation_errors"]
            and compat["parsed_action_type"] in ("tool_call", "response")
        )
        checks.append(("openeta backend compat smoke", ok, json.dumps(
            {k: compat[k] for k in ("url", "request_model", "no_chat_template_kwargs",
                                     "image_in_user_message", "status", "payload_kind")})))
        report["openeta_backend_compat"] = compat
    except Exception as exc:  # noqa: BLE001
        checks.append(("openeta backend compat smoke", False, f"{type(exc).__name__}: {exc}"))
        report["openeta_backend_compat"] = {"error": str(exc)}

    for name, ok, detail in checks:
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    report["checks"] = {name: bool(ok) for name, ok, _ in checks}
    report["status"] = "PASS" if all(ok for _, ok, _ in checks) else "FAIL"
    (out_dir / "deepseek_flash.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nG2 MODEL GATE: {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
