"""Action schema formatting, deterministic JSON parsing and validation.

JSON repair is deterministic only: markdown fences are stripped and the first
balanced JSON object is extracted. No regex guessing of action ids, no random
fallback actions.
"""
from __future__ import annotations

import json

import jsonschema

_PRIVATE_PARAM_KEYS = {
    "golden_actions", "goal_preds", "success_conditions", "target_pose",
    "physical_task_id", "source_task_id", "global_task_id", "transfer_pair_id",
    "retention_anchor_id", "private_eval_metadata",
}


def format_tool_specs_for_prompt(tool_specs: list[dict]) -> str:
    return json.dumps(tool_specs or [], ensure_ascii=False, indent=2)


def _strip_code_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def parse_agent_json(raw: str) -> tuple[dict | None, list[str]]:
    """Deterministic parse: fenced JSON / first balanced object."""
    text = _strip_code_fences(raw)
    if not text:
        return None, ["empty response"]
    candidate = _first_balanced_object(text)
    if candidate is None:
        return None, ["no JSON object found in response"]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON: {exc}"]
    if not isinstance(payload, dict):
        return None, ["response JSON is not an object"]
    return payload, []


def validate_agent_action(payload: dict, tool_specs: list[dict]) -> tuple[str | None, dict | None, list[str]]:
    """Validate exactly one legal action; returns (name, parameters, errors)."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return None, None, ["payload is not a JSON object"]
    if "actions" in payload:
        return None, None, ["multiple actions are not allowed; return exactly one action"]
    action = payload.get("action")
    if not isinstance(action, dict):
        return None, None, ["missing 'action' object"]
    name = action.get("name")
    parameters = action.get("parameters", {})
    if not isinstance(name, str) or not name:
        return None, None, ["action.name must be a non-empty string"]
    if not isinstance(parameters, dict):
        return None, None, ["action.parameters must be an object"]

    by_name = {spec.get("name"): spec for spec in (tool_specs or []) if isinstance(spec, dict)}
    spec = by_name.get(name)
    if spec is None:
        return None, None, [f"action {name!r} is not in the legal action set"]

    for key in parameters:
        if key in _PRIVATE_PARAM_KEYS:
            errors.append(f"parameter {key!r} is not allowed")

    schema = spec.get("parameters")
    if isinstance(schema, dict):
        try:
            jsonschema.validate(instance=parameters, schema=schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"parameters fail schema for {name!r}: {exc.message}")
        except jsonschema.SchemaError as exc:
            errors.append(f"tool spec schema invalid for {name!r}: {exc.message}")

    if errors:
        return None, None, errors
    return name, dict(parameters), []
