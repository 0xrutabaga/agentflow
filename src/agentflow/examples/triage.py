"""Incident-triage agent over local fixtures.

Four tools, chosen to exercise each idempotency rule:

* ``get_alert``, ``get_recent_deploys``: idempotent reads.
* ``search_logs``: idempotent and flaky. In each process its first two attempts raise
  ``ConnectionError``, so the runtime's retry path runs every time.
* ``create_ticket``: NOT idempotent. Each execution writes a new ticket file with the next
  number, so running it twice really does create a duplicate.

Every execution is appended to ``executions.jsonl`` in the data directory with the process id,
which is how the demo and tests show what ran in which process.

The model is a ``ScriptedModel`` unless ``AGENTFLOW_PROVIDER=openai`` is set, in which case the
OpenAI-compatible HTTP provider is configured from the environment (see
``agentflow.providers.openai_compat``).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from agentflow.agent import Agent, Limits
from agentflow.messages import Message, ModelResponse, ToolCallRequest
from agentflow.providers.base import Provider
from agentflow.providers.fake import ScriptedModel
from agentflow.tools import RetryPolicy, Tool, ToolError, ToolRegistry
from agentflow.trace import Redactor, chain_redactors, redact_sensitive_keys

AGENT_REF = "agentflow.examples.triage:build_agent"
ALERT_ID = "ALERT-1042"
GOAL = (
    f"Triage alert {ALERT_ID}: find the most likely cause, open a ticket, and report the "
    "severity, evidence and next steps."
)

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["alert_id", "severity", "suspected_cause", "evidence", "ticket_id", "next_steps"],
    "properties": {
        "alert_id": {"type": "string"},
        "severity": {"enum": ["sev1", "sev2", "sev3"]},
        "suspected_cause": {"type": "string", "minLength": 10},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "ticket_id": {"type": "string", "pattern": "^TCK-[0-9]{4}$"},
        "next_steps": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
}

SYSTEM_PROMPT = (
    "You are an on-call triage assistant. Use the tools to read the alert, the service logs and "
    "recent deploys. Open exactly one ticket. Base every claim on tool output."
)

PHONE = re.compile(r"\+\d[\d ()-]{6,}\d")  # international format only


def _fixture(name: str) -> str:
    return resources.files("agentflow.examples").joinpath("fixtures", name).read_text("utf-8")


def _log_execution(data_dir: Path, tool: str, outcome: str) -> None:
    line = json.dumps({"pid": os.getpid(), "tool": tool, "outcome": outcome})
    with (data_dir / "executions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_executions(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "executions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def mask_phone_numbers(event: dict[str, Any]) -> dict[str, Any]:
    """Example custom redactor: mask anything that looks like a phone number."""

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return PHONE.sub("[PHONE]", value)
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    result: dict[str, Any] = walk(event)
    return result


def build_tools(data_dir: Path, flaky_failures: int = 2) -> ToolRegistry:
    state = {"search_attempts": 0}

    def get_alert(alert_id: str) -> dict[str, Any]:
        _log_execution(data_dir, "get_alert", "ok")
        alert: dict[str, Any] = json.loads(_fixture("alert.json"))
        if alert["alert_id"] != alert_id:
            raise ToolError(f"no alert with id {alert_id}")
        return alert

    def search_logs(service: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        state["search_attempts"] += 1
        if state["search_attempts"] <= flaky_failures:
            _log_execution(data_dir, "search_logs", "connection_error")
            raise ConnectionError("log index did not respond")
        _log_execution(data_dir, "search_logs", "ok")
        rows = [json.loads(line) for line in _fixture("logs.jsonl").splitlines() if line.strip()]
        hits = [r for r in rows if r["service"] == service and query.lower() in r["msg"].lower()]
        return hits[:limit]

    def get_recent_deploys(service: str) -> list[dict[str, Any]]:
        _log_execution(data_dir, "get_recent_deploys", "ok")
        deploys: list[dict[str, Any]] = json.loads(_fixture("deploys.json"))
        return [d for d in deploys if d["service"] == service]

    def create_ticket(title: str, severity: str, summary: str) -> dict[str, Any]:
        tickets = data_dir / "tickets"
        tickets.mkdir(parents=True, exist_ok=True)
        ticket_id = f"TCK-{len(list(tickets.glob('TCK-*.json'))) + 1:04d}"
        body = {"id": ticket_id, "title": title, "severity": severity, "summary": summary}
        (tickets / f"{ticket_id}.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
        _log_execution(data_dir, "create_ticket", "ok")
        return {"ticket_id": ticket_id}

    string = {"type": "string", "minLength": 1}
    return ToolRegistry(
        [
            Tool(
                name="get_alert",
                description="Fetch an alert by id.",
                parameters={
                    "type": "object",
                    "properties": {"alert_id": string},
                    "required": ["alert_id"],
                    "additionalProperties": False,
                },
                fn=get_alert,
                idempotent=True,
                timeout_s=5.0,
            ),
            Tool(
                name="search_logs",
                description="Search a service's logs for a case-insensitive substring.",
                parameters={
                    "type": "object",
                    "properties": {
                        "service": string,
                        "query": string,
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["service", "query"],
                    "additionalProperties": False,
                },
                fn=search_logs,
                idempotent=True,
                timeout_s=5.0,
                retry=RetryPolicy(max_attempts=4, backoff_s=0.1),
            ),
            Tool(
                name="get_recent_deploys",
                description="List recent deploys of a service.",
                parameters={
                    "type": "object",
                    "properties": {"service": string},
                    "required": ["service"],
                    "additionalProperties": False,
                },
                fn=get_recent_deploys,
                idempotent=True,
                timeout_s=5.0,
            ),
            Tool(
                name="create_ticket",
                description="Open an incident ticket. Each call creates a new ticket.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": string,
                        "severity": {"enum": ["sev1", "sev2", "sev3"]},
                        "summary": string,
                    },
                    "required": ["title", "severity", "summary"],
                    "additionalProperties": False,
                },
                fn=create_ticket,
                idempotent=False,
                timeout_s=5.0,
            ),
        ]
    )


def _call(turn: int, idx: int, name: str, **arguments: Any) -> ToolCallRequest:
    return ToolCallRequest(id=f"call_{turn}_{idx}", name=name, arguments=arguments)


def _final_answer(messages: Sequence[Message]) -> ModelResponse:
    ticket_id = "TCK-0000"
    for message in messages:
        if message.role == "tool" and message.name == "create_ticket" and message.content:
            payload = json.loads(message.content)
            if payload.get("ok"):
                ticket_id = payload["result"]["ticket_id"]
    answer = {
        "alert_id": ALERT_ID,
        "severity": "sev2",
        "suspected_cause": (
            "deploy 2026.10.28-2 reduced the checkout db pool from 50 to 10 connections; "
            "requests now time out waiting for a connection"
        ),
        "evidence": [
            "13:58 deploy 2026.10.28-2: db pool size 50 -> 10",
            "14:01 db pool exhausted (size=10, waiting=37)",
            "14:02 and 14:05 timeout acquiring db connection after 2000ms",
            "payments reports healthy (p95 120ms), so the upstream timeout is a symptom",
        ],
        "ticket_id": ticket_id,
        "next_steps": [
            "roll back checkout to 2026.10.27-1 or restore pool size 50",
            "add an alert on db pool wait count",
        ],
    }
    return ModelResponse(content=json.dumps(answer))


def scripted_triage_model() -> ScriptedModel:
    return ScriptedModel(
        [
            ModelResponse(tool_calls=(_call(0, 0, "get_alert", alert_id=ALERT_ID),)),
            ModelResponse(
                tool_calls=(
                    _call(1, 0, "search_logs", service="checkout", query="timeout", limit=5),
                    _call(1, 1, "get_recent_deploys", service="checkout"),
                )
            ),
            ModelResponse(
                tool_calls=(
                    _call(
                        2,
                        0,
                        "create_ticket",
                        title="checkout p95 latency above SLO after deploy 2026.10.28-2",
                        severity="sev2",
                        summary="db pool reduced to 10; connection acquisition timeouts",
                    ),
                )
            ),
            # Deliberately invalid first answer: wrong severity label, no evidence, missing
            # fields. The runtime feeds the schema errors back and the next turn repairs it.
            ModelResponse(
                content=json.dumps(
                    {
                        "alert_id": ALERT_ID,
                        "severity": "high",
                        "suspected_cause": "database",
                        "evidence": [],
                    }
                )
            ),
            _final_answer,
        ]
    )


def _provider() -> Provider:
    if os.environ.get("AGENTFLOW_PROVIDER") == "openai":
        from agentflow.providers.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider.from_env()
    return scripted_triage_model()


def build_agent(data_dir: Path) -> Agent:
    redactor: Redactor = chain_redactors(redact_sensitive_keys, mask_phone_numbers)
    return Agent(
        name="incident-triage",
        system_prompt=SYSTEM_PROMPT,
        tools=build_tools(data_dir),
        provider=_provider(),
        output_schema=OUTPUT_SCHEMA,
        limits=Limits(max_steps=10, max_tool_calls=10, max_repairs=2, max_wall_s=120.0),
        redactor=redactor,
    )
