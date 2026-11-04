"""JSON Schema validation for tool arguments and model output.

Errors are turned into short, path-qualified strings because they are fed back to the model as
repair instructions; a model fixes ``$.severity: 'high' is not one of ['sev1', 'sev2', 'sev3']``
far more reliably than a stack trace.
"""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

MAX_REPORTED_ERRORS = 5


def make_validator(schema: dict[str, Any]) -> Draft202012Validator:
    """Check that ``schema`` is itself valid, then build a validator for it."""
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _path(error: ValidationError) -> str:
    parts = ["$"]
    for item in error.absolute_path:
        parts.append(f"[{item}]" if isinstance(item, int) else f".{item}")
    return "".join(parts)


def validation_errors(validator: Draft202012Validator, instance: Any) -> list[str]:
    """Return up to ``MAX_REPORTED_ERRORS`` readable errors; an empty list means valid."""
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(map(str, e.absolute_path)))
    messages = [f"{_path(e)}: {e.message}" for e in errors[:MAX_REPORTED_ERRORS]]
    if len(errors) > MAX_REPORTED_ERRORS:
        messages.append(f"... and {len(errors) - MAX_REPORTED_ERRORS} more")
    return messages


def parse_json_output(content: str | None) -> tuple[Any, str | None]:
    """Parse model content as JSON, tolerating one surrounding Markdown code fence.

    Returns ``(value, None)`` on success and ``(None, error)`` on failure.
    """
    if content is None or not content.strip():
        return None, "response was empty; expected a JSON object"
    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1 and text.endswith("```"):
            text = text[first_newline + 1 : -3].strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc.msg} at line {exc.lineno} col {exc.colno}"
