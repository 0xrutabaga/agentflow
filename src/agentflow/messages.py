"""Provider-neutral conversation types.

These are plain dataclasses that round-trip through JSON, because every message is persisted to
SQLite and replayed to the model after a resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCallRequest:
    """A tool call as the model asked for it.

    ``arguments`` is whatever the model produced. It is usually a dict, but a provider passes
    through anything it could not parse (for example a malformed JSON string) so that argument
    validation, not the provider, decides to reject it and ask the model to repair it.
    """

    id: str
    name: str
    arguments: Any

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ToolCallRequest:
        return cls(id=str(data["id"]), name=str(data["name"]), arguments=data["arguments"])


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCallRequest, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    """One model turn: either tool calls, or final content (expected to be JSON)."""

    content: str | None = None
    tool_calls: tuple[ToolCallRequest, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)

    def to_message(self) -> Message:
        return Message(role="assistant", content=self.content, tool_calls=self.tool_calls)
