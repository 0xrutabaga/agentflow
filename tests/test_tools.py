from __future__ import annotations

import time
from typing import Any

import pytest
from conftest import make_tool
from jsonschema.exceptions import SchemaError

from agentflow.schema import make_validator, parse_json_output, validation_errors
from agentflow.tools import (
    RetryPolicy,
    Tool,
    ToolError,
    ToolRegistry,
    ToolTimeout,
    invoke_with_timeout,
)

PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {"service": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}},
    "required": ["service"],
    "additionalProperties": False,
}


def registry() -> ToolRegistry:
    return ToolRegistry([make_tool("search", lambda **_: [], parameters=PARAMS)])


def test_valid_call_has_no_errors() -> None:
    assert registry().check_call("search", {"service": "checkout", "limit": 5}) == []


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({}, "$: 'service' is a required property"),
        ({"service": "x", "limit": 0}, "$.limit: 0 is less than the minimum of 1"),
        ({"service": "x", "extra": 1}, "Additional properties are not allowed"),
        ("{not json", "$: '{not json' is not of type 'object'"),
    ],
)
def test_invalid_arguments_are_described(arguments: Any, expected: str) -> None:
    errors = registry().check_call("search", arguments)
    assert any(expected in e for e in errors), errors


def test_unknown_tool_lists_available_tools() -> None:
    assert registry().check_call("delete_everything", {}) == [
        "unknown tool 'delete_everything'; available tools: search"
    ]


def test_duplicate_registration_and_bad_schemas_are_refused() -> None:
    reg = registry()
    with pytest.raises(ValueError, match="already registered"):
        reg.register(make_tool("search", lambda: None))
    with pytest.raises(ValueError, match="type 'object'"):
        Tool("t", "d", {"type": "string"}, lambda: None, idempotent=True)
    with pytest.raises(SchemaError):
        Tool("t", "d", {"type": "object", "properties": {"a": {"type": 7}}}, lambda: None, True)


def test_retry_policy_backoff_is_capped() -> None:
    policy = RetryPolicy(max_attempts=6, backoff_s=1.0, multiplier=3.0, max_backoff_s=5.0)
    assert [policy.delay_before(n) for n in range(1, 6)] == [0.0, 1.0, 3.0, 5.0, 5.0]
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


def test_invoke_returns_value_and_propagates_errors() -> None:
    assert invoke_with_timeout(make_tool("t", lambda: {"a": 1}), {}, 1.0) == {"a": 1}

    def boom() -> None:
        raise KeyError("x")

    with pytest.raises(KeyError):
        invoke_with_timeout(make_tool("t", boom), {}, 1.0)


def test_invoke_times_out() -> None:
    started = time.monotonic()
    with pytest.raises(ToolTimeout):
        invoke_with_timeout(make_tool("t", lambda: time.sleep(2)), {}, 0.05)
    assert time.monotonic() - started < 1.0


def test_non_json_result_is_a_tool_error() -> None:
    with pytest.raises(ToolError, match="not JSON"):
        invoke_with_timeout(make_tool("t", lambda: {1, 2}), {}, 1.0)


def test_parse_json_output() -> None:
    assert parse_json_output('{"a": 1}') == ({"a": 1}, None)
    assert parse_json_output('```json\n{"a": 1}\n```') == ({"a": 1}, None)
    assert parse_json_output("")[1] == "response was empty; expected a JSON object"
    value, error = parse_json_output("{'a': 1}")
    assert value is None
    assert error is not None
    assert error.startswith("response is not valid JSON")


def test_validation_errors_are_capped() -> None:
    schema = {"type": "object", "properties": {k: {"type": "string"} for k in "abcdefg"}}
    errors = validation_errors(make_validator(schema), {k: 1 for k in "abcdefg"})
    assert len(errors) == 6
    assert errors[-1] == "... and 2 more"
