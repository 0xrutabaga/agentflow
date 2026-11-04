from __future__ import annotations

import json
from pathlib import Path

from conftest import FINAL, Harness, call, make_agent, make_tool

from agentflow.examples.triage import mask_phone_numbers
from agentflow.trace import (
    REDACTED,
    TraceWriter,
    chain_redactors,
    format_trace,
    read_trace,
    redact_sensitive_keys,
)


def test_default_redactor_masks_credential_keys_at_any_depth() -> None:
    event = {
        "event": "tool_finished",
        "result": {"rows": [{"session_token": "abc", "msg": "ok"}], "API_KEY": "k"},
        "arguments": {"password": "p", "user": "u"},
    }
    redacted = redact_sensitive_keys(event)
    assert redacted["result"]["rows"][0] == {"session_token": REDACTED, "msg": "ok"}
    assert redacted["result"]["API_KEY"] == REDACTED
    assert redacted["arguments"] == {"password": REDACTED, "user": "u"}
    assert event["arguments"]["password"] == "p"  # noqa: S105 - input is not mutated


def test_redactors_chain() -> None:
    redact = chain_redactors(redact_sensitive_keys, mask_phone_numbers)
    out = redact({"reporter": {"phone": "+1 555 0100", "name": "n"}, "secret": "s"})
    assert out == {"reporter": {"phone": "[PHONE]", "name": "n"}, "secret": REDACTED}
    # Dates and version strings are not phone numbers.
    text = "2026.10.28-2 at 2026-10-28T14:07:00Z"
    assert mask_phone_numbers({"v": text})["v"] == text


def test_run_trace_is_redacted_on_disk(harness: Harness) -> None:
    tool = make_tool("login", lambda: {"token": "very-secret-value", "user": "u"})
    run_id, _ = harness.start(make_agent([call("login"), FINAL], [tool]))
    raw = (harness.runtime.trace_dir / f"{run_id}.jsonl").read_text(encoding="utf-8")
    assert "very-secret-value" not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    finished = next(e for e in events if e["event"] == "tool_finished")
    assert finished["result"] == {"token": REDACTED, "user": "u"}
    names = [e["event"] for e in events]
    assert names[0] == "run_created"
    assert names[-1] == "run_returned"


def test_read_trace_skips_a_torn_final_line(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path)
    writer.write("r1", "state", **{"from": "pending", "to": "running"})
    with writer.path_for("r1").open("a", encoding="utf-8") as handle:
        handle.write('{"event": "tool_sta')
    events = read_trace(writer.path_for("r1"))
    assert [e["event"] for e in events] == ["state"]
    assert "pending -> running" in format_trace(events)
