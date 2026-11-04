from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from agentflow.agent import Agent, Limits
from agentflow.messages import ModelResponse, ToolCallRequest
from agentflow.providers.fake import ScriptedModel, ScriptStep
from agentflow.runtime import Runtime
from agentflow.tools import RetryPolicy, Tool, ToolRegistry

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string", "minLength": 1}},
    "additionalProperties": False,
}
NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
FINAL = ModelResponse(content='{"answer": "done"}')


def call(name: str, turn: int = 0, **arguments: Any) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCallRequest(f"p{turn}", name, arguments),))


def make_tool(
    name: str,
    fn: Callable[..., Any],
    *,
    idempotent: bool = True,
    timeout_s: float = 5.0,
    max_attempts: int = 3,
    parameters: dict[str, Any] | None = None,
) -> Tool:
    return Tool(
        name=name,
        description=f"test tool {name}",
        parameters=parameters or NO_ARGS,
        fn=fn,
        idempotent=idempotent,
        timeout_s=timeout_s,
        retry=RetryPolicy(max_attempts=max_attempts, backoff_s=0.2),
    )


def make_agent(
    script: Sequence[ScriptStep],
    tools: Sequence[Tool] = (),
    *,
    limits: Limits | None = None,
    provider: Any = None,
) -> Agent:
    return Agent(
        name="test",
        system_prompt="test agent",
        tools=ToolRegistry(list(tools)),
        provider=provider or ScriptedModel(script),
        output_schema=ANSWER_SCHEMA,
        limits=limits or Limits(max_wall_s=None),
    )


class Harness:
    """A runtime whose sleeps are recorded instead of slept."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.sleeps: list[float] = []
        self.runtime = Runtime(data_dir, sleep=self.sleeps.append)

    def start(self, agent: Agent, goal: str = "do the thing") -> tuple[str, Any]:
        run_id = self.runtime.create("tests:inline", goal, agent=agent)
        return run_id, self.runtime.run(run_id, agent=agent)


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    h = Harness(tmp_path)
    yield h
    h.runtime.close()
