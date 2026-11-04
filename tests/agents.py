"""Agent used by the subprocess crash tests (imported by reference ``agents:build``).

Script: read_a -> slow_read -> send_email -> final answer. Every execution is logged with its
pid to ``executions.jsonl``. If ``AGENTS_FAIL_IN`` names a tool, it raises ``ConnectionError``
after logging (an ambiguous failure). If ``AGENTS_HANG_IN`` names a tool, that tool logs its
execution (the side effect has happened), writes ``hang.marker`` with its pid and then sleeps
until killed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from agentflow.agent import Agent, Limits
from agentflow.messages import ModelResponse, ToolCallRequest
from agentflow.providers.fake import ScriptedModel
from agentflow.tools import Tool, ToolRegistry

REF = "agents:build"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["done"],
    "properties": {"done": {"type": "boolean"}},
    "additionalProperties": False,
}
ARGS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


def executions(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "executions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _tool(data_dir: Path, name: str, idempotent: bool) -> Tool:
    def fn() -> dict[str, Any]:
        with (data_dir / "executions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "tool": name}) + "\n")
        if os.environ.get("AGENTS_FAIL_IN") == name:
            raise ConnectionError(f"{name}: connection reset after sending")
        if os.environ.get("AGENTS_HANG_IN") == name:
            marker = data_dir / "hang.marker"
            tmp = data_dir / "hang.marker.tmp"
            tmp.write_text(str(os.getpid()), encoding="utf-8")
            os.replace(tmp, marker)
            while True:
                time.sleep(3600)
        return {"tool": name, "ok": True}

    return Tool(
        name=name,
        description=name,
        parameters=ARGS,
        fn=fn,
        idempotent=idempotent,
        timeout_s=120.0,
    )


def build(data_dir: Path) -> Agent:
    def call(turn: int, name: str) -> ModelResponse:
        return ModelResponse(tool_calls=(ToolCallRequest(f"c{turn}", name, {}),))

    return Agent(
        name="crash-test",
        system_prompt="test agent",
        tools=ToolRegistry(
            [
                _tool(data_dir, "read_a", idempotent=True),
                _tool(data_dir, "slow_read", idempotent=True),
                _tool(data_dir, "send_email", idempotent=False),
            ]
        ),
        provider=ScriptedModel(
            [
                call(0, "read_a"),
                call(1, "slow_read"),
                call(2, "send_email"),
                ModelResponse(content='{"done": true}'),
            ]
        ),
        output_schema=SCHEMA,
        limits=Limits(max_wall_s=None),
    )
