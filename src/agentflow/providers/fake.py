"""Deterministic scripted model.

The response is chosen by counting the assistant messages already in the conversation, not by
an internal counter. That makes the fake correct across a crash and resume: a fresh process that
replays the persisted conversation gets the same next response the crashed process would have.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from agentflow.messages import Message, ModelResponse
from agentflow.providers.base import ProviderError

ScriptStep = ModelResponse | Callable[[Sequence[Message]], ModelResponse]


class ScriptedModel:
    def __init__(self, script: Sequence[ScriptStep]) -> None:
        self._script = list(script)
        self.calls = 0  # calls made in this process, for tests

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        output_schema: dict[str, Any],
    ) -> ModelResponse:
        self.calls += 1
        turn = sum(1 for m in messages if m.role == "assistant")
        if turn >= len(self._script):
            raise ProviderError(f"scripted model has no response for turn {turn}")
        step = self._script[turn]
        return step(messages) if callable(step) else step
