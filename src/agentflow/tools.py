"""Tool registry: JSON Schema arguments, per-tool timeout, retry policy and idempotency.

How a failed attempt is classified decides whether it may be retried:

========================================  ==============  ==================
attempt outcome                           idempotent      non-idempotent
========================================  ==============  ==================
returned a value                          recorded        recorded
raised ``ToolError``                      failed, no      failed, no retry
                                          retry
raised ``TransientToolError``             retried         retried
timed out, or raised any other exception  retried         run blocked for a
                                                          human decision
========================================  ==============  ==================

``TransientToolError`` is the tool author's statement that the attempt had no side effect (for
example the request was rejected before it was sent), which is the only thing that makes retrying
a non-idempotent tool safe. Any other exception and every timeout is *ambiguous*: the side effect
may or may not have happened.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from agentflow.schema import make_validator, validation_errors


class ToolError(Exception):
    """Definite, permanent failure. Not retried; the message is shown to the model."""


class TransientToolError(ToolError):
    """Definite failure with no side effect. Safe to retry even for non-idempotent tools."""


class ToolTimeout(Exception):
    """The attempt did not finish within its timeout. Its outcome is unknown."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_s: float = 0.2
    multiplier: float = 2.0
    max_backoff_s: float = 5.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def delay_before(self, attempt: int) -> float:
        """Delay before attempt number ``attempt`` (2 is the first retry)."""
        if attempt <= 1:
            return 0.0
        return float(min(self.max_backoff_s, self.backoff_s * self.multiplier ** (attempt - 2)))


ToolFn = Callable[..., Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn
    idempotent: bool
    timeout_s: float = 10.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    validator: Draft202012Validator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.parameters.get("type") != "object":
            raise ValueError(f"tool {self.name!r}: parameters schema must have type 'object'")
        if self.timeout_s <= 0:
            raise ValueError(f"tool {self.name!r}: timeout_s must be positive")
        object.__setattr__(self, "validator", make_validator(self.parameters))

    def spec(self) -> dict[str, Any]:
        """Description handed to the model provider."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def check_call(self, name: str, arguments: Any) -> list[str]:
        """Validate a requested call before anything runs. Empty list means it may run."""
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(sorted(self._tools)) or "none"
            return [f"unknown tool {name!r}; available tools: {known}"]
        return validation_errors(tool.validator, arguments)


def invoke_with_timeout(tool: Tool, arguments: dict[str, Any], timeout_s: float) -> Any:
    """Run one attempt in a daemon thread and wait at most ``timeout_s``.

    Python cannot kill a thread, so a timed-out attempt may still be running (and may still
    produce its side effect) after this returns. That is exactly why a timeout is classified as
    ambiguous. The result must be JSON-serialisable because it is checkpointed.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = tool.fn(**arguments)
        except BaseException as exc:  # handed back to the caller's thread
            box["error"] = exc

    worker = threading.Thread(target=target, name=f"agentflow-tool-{tool.name}", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise ToolTimeout(f"tool {tool.name!r} did not finish within {timeout_s:.2f}s")
    if "error" in box:
        raise box["error"]
    value = box.get("value")
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"tool {tool.name!r} returned a value that is not JSON: {exc}") from exc
    return value
