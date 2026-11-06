"""Regression tests for defects found in independent review."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import agents
import pytest
from conftest import FINAL, Harness, call, make_agent, make_tool

from agentflow.agent import Agent, Limits
from agentflow.cli import main
from agentflow.messages import ModelResponse, ToolCallRequest
from agentflow.runtime import Runtime
from agentflow.state import RunStatus, ToolCallStatus
from agentflow.trace import REDACTED, redact_sensitive_keys

AMOUNT = {
    "type": "object",
    "properties": {"amount": {"type": "integer"}},
    "required": ["amount"],
}


def _simulate_interrupted_call(
    runtime: Runtime, run_id: str, tool: str, *, idempotent: bool, max_attempts: int = 3
) -> None:
    """Leave the run as a process killed mid-attempt would: call `started`, no result."""
    request = call(tool).tool_calls[0]
    with runtime.store.tx(run_id, None) as tx:
        tx.set_status(RunStatus.RUNNING)
        tx.bump("model_calls")
        tx.add_message(call(tool).to_message())
        tx.add_tool_call(
            "0.0",
            0,
            0,
            request,
            ToolCallStatus.REQUESTED,
            idempotent=idempotent,
            max_attempts=max_attempts,
        )
        tx.update_tool_call("0.0", ToolCallStatus.STARTED, attempts=1, started=True)
        tx.set_status(RunStatus.WAITING_TOOL)


# -- 1. blocked run past its deadline --------------------------------------------------------


def _blocked_run(tmp_path: Path, clock: list[float]) -> tuple[Runtime, Agent, str, list[int]]:
    effects: list[int] = []

    def slow_pay(amount: int) -> dict[str, int]:
        time.sleep(0.3)
        effects.append(amount)
        return {"paid": amount}

    tool = make_tool("pay", slow_pay, idempotent=False, timeout_s=0.05, parameters=AMOUNT)
    agent = make_agent([call("pay", amount=5), FINAL], [tool], limits=Limits(max_wall_s=10.0))
    runtime = Runtime(tmp_path, sleep=lambda _: None, clock=lambda: clock[0])
    run_id = runtime.create("tests:inline", "goal", agent=agent)
    assert runtime.run(run_id, agent=agent).status is RunStatus.BLOCKED
    return runtime, agent, run_id, effects


def test_blocked_run_past_deadline_fails_cleanly_after_resolve(tmp_path: Path) -> None:
    clock = [1000.0]
    runtime, agent, run_id, _ = _blocked_run(tmp_path, clock)
    clock[0] += 60
    runtime.resolve(run_id, "0.0", "succeed", {"paid": 5})
    outcome = runtime.run(run_id, agent=agent)
    assert outcome.status is RunStatus.FAILED
    assert outcome.error == "budget_exhausted: wall-clock deadline passed"
    # The operator's decision is kept even though the run could not continue.
    assert runtime.store.get_tool_call(run_id, "0.0").status is ToolCallStatus.SUCCEEDED
    runtime.close()


def test_blocked_run_past_deadline_fails_cleanly_without_resolve(tmp_path: Path) -> None:
    clock = [1000.0]
    runtime, agent, run_id, _ = _blocked_run(tmp_path, clock)
    clock[0] += 60
    outcome = runtime.run(run_id, agent=agent)
    assert outcome.status is RunStatus.FAILED
    assert runtime.store.get_tool_call(run_id, "0.0").status is ToolCallStatus.AMBIGUOUS
    runtime.close()


def test_cli_reports_runtime_errors_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = Runtime(tmp_path)
    run_id = runtime.create("no_such_module_xyz:build", "g", agent=make_agent([FINAL]))
    runtime.close()
    code = main(["--data-dir", str(tmp_path), "resume", run_id])
    err = capsys.readouterr().err
    assert code == 2
    assert "Traceback" not in err
    assert "no_such_module_xyz" in err


# -- 2. non-idempotent call near the deadline -----------------------------------------------


def test_non_idempotent_call_is_not_started_without_time_for_its_timeout(tmp_path: Path) -> None:
    effects: list[int] = []

    def pay(amount: int) -> dict[str, int]:
        effects.append(amount)
        return {"paid": amount}

    clock = [1000.0]
    tool = make_tool("pay", pay, idempotent=False, timeout_s=5.0, parameters=AMOUNT)
    agent = make_agent([call("pay", amount=5), FINAL], [tool], limits=Limits(max_wall_s=3.0))
    runtime = Runtime(tmp_path, sleep=lambda _: None, clock=lambda: clock[0])
    run_id = runtime.create("tests:inline", "goal", agent=agent)
    outcome = runtime.run(run_id, agent=agent)
    assert outcome.status is RunStatus.FAILED
    assert "not started" in (outcome.error or "")
    assert effects == []
    record = runtime.store.get_tool_call(run_id, "0.0")
    assert (record.status, record.attempts) == (ToolCallStatus.REQUESTED, 0)
    runtime.close()


# -- 3. arguments the function cannot accept ------------------------------------------------


def test_arguments_the_function_cannot_bind_are_rejected_not_ambiguous(harness: Harness) -> None:
    executed: list[int] = []

    def pay(amount: int) -> dict[str, int]:
        executed.append(amount)
        return {"ok": 1}

    tool = make_tool("pay", pay, idempotent=False, parameters=AMOUNT)  # schema allows extras
    script = [call("pay", amount=5, memo="x"), call("pay", 1, amount=5), FINAL]
    run_id, outcome = harness.start(make_agent(script, [tool]))
    assert outcome.status is RunStatus.COMPLETED
    assert executed == [5]  # only the second, valid call ran
    first = harness.runtime.store.get_tool_call(run_id, "0.0")
    assert first.status is ToolCallStatus.REJECTED
    assert "memo" in (first.error or "")


# -- 4. redaction -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "redacted"),
    [
        ("session_token", True),
        ("API_KEY", True),
        ("apiKey", True),
        ("x-api-key", True),
        ("Authorization", True),
        ("client_secret", True),
        ("password", True),
        ("prompt_tokens", False),
        ("completion_tokens", False),
        ("max_tokens", False),
        ("tokenizer", False),
        ("secretary", False),
    ],
)
def test_key_redaction_matches_whole_name_segments(key: str, redacted: bool) -> None:
    out = redact_sensitive_keys({key: 12345})
    assert (out[key] == REDACTED) is redacted


def test_secret_shaped_values_are_redacted_anywhere(harness: Harness) -> None:
    tool = make_tool("pay", lambda amount: {"ok": 1}, parameters=AMOUNT)
    malformed = ToolCallRequest("c1", "pay", '{"api_key": "sk-LIVEabcdef123456", "amount": ')
    agent = make_agent([ModelResponse(tool_calls=(malformed,)), FINAL], [tool])
    run_id, _ = harness.start(agent, goal="use key sk-LIVE999abcdef9999 and Bearer abc.def.ghi")
    raw = (harness.runtime.trace_dir / f"{run_id}.jsonl").read_text(encoding="utf-8")
    assert "sk-LIVEabcdef123456" not in raw
    assert "sk-LIVE999abcdef9999" not in raw
    assert "abc.def.ghi" not in raw
    usage = redact_sensitive_keys({"usage": {"prompt_tokens": 10, "completion_tokens": 3}})
    assert usage == {"usage": {"prompt_tokens": 10, "completion_tokens": 3}}


def test_operator_events_use_the_agents_redactor(harness: Harness) -> None:
    def mark(event: dict[str, Any]) -> dict[str, Any]:
        return {**event, "redacted_by": "agent"}

    tool = make_tool("pay", lambda amount: {"ok": 1}, idempotent=False, parameters=AMOUNT)
    base = make_agent([call("pay", amount=5), FINAL], [tool])
    agent = Agent(
        name=base.name,
        system_prompt=base.system_prompt,
        tools=base.tools,
        provider=base.provider,
        output_schema=base.output_schema,
        limits=base.limits,
        redactor=mark,
    )
    run_id = harness.runtime.create("tests:inline", "goal", agent=agent)
    harness.runtime.cancel(run_id, agent=agent)
    events = [
        json.loads(line)
        for line in (harness.runtime.trace_dir / f"{run_id}.jsonl").read_text("utf-8").splitlines()
    ]
    cancel = next(e for e in events if e["event"] == "cancel_requested")
    assert cancel["redacted_by"] == "agent"


# -- 5. tool missing on resume; persisted policy ----------------------------------------------


def test_interrupted_call_of_a_removed_tool_blocks_instead_of_failing(harness: Harness) -> None:
    agent = make_agent([call("send"), FINAL], [make_tool("send", lambda: {}, idempotent=False)])
    run_id = harness.runtime.create("tests:inline", "goal", agent=agent)
    _simulate_interrupted_call(harness.runtime, run_id, "send", idempotent=False)
    without_tool = make_agent([call("send"), FINAL], [])
    outcome = harness.runtime.run(run_id, agent=without_tool)
    assert outcome.status is RunStatus.BLOCKED
    assert harness.runtime.store.get_tool_call(run_id, "0.0").status is ToolCallStatus.AMBIGUOUS


def test_idempotency_recorded_at_call_time_wins_over_current_code(harness: Harness) -> None:
    executed: list[int] = []
    tool = make_tool("send", lambda: executed.append(1) or {}, idempotent=True)
    agent = make_agent([call("send"), FINAL], [tool])
    run_id = harness.runtime.create("tests:inline", "goal", agent=agent)
    # The call was made while the tool was declared non-idempotent.
    _simulate_interrupted_call(harness.runtime, run_id, "send", idempotent=False)
    outcome = harness.runtime.run(run_id, agent=agent)
    assert outcome.status is RunStatus.BLOCKED
    assert executed == []


def test_policy_is_persisted_when_the_model_requests_the_call(harness: Harness) -> None:
    tool = make_tool("send", lambda: {}, idempotent=False, max_attempts=7)
    run_id, _ = harness.start(make_agent([call("send"), FINAL], [tool]))
    record = harness.runtime.store.get_tool_call(run_id, "0.0")
    assert (record.idempotent, record.max_attempts) == (False, 7)


# -- 6. cancel with an in-flight call -----------------------------------------------------------


def test_cancel_marks_an_interrupted_call_ambiguous(harness: Harness) -> None:
    agent = make_agent([call("send"), FINAL], [make_tool("send", lambda: {}, idempotent=False)])
    run_id = harness.runtime.create("tests:inline", "goal", agent=agent)
    _simulate_interrupted_call(harness.runtime, run_id, "send", idempotent=False)
    harness.runtime.cancel(run_id)
    outcome = harness.runtime.run(run_id, agent=agent)
    assert outcome.status is RunStatus.CANCELLED
    record = harness.runtime.store.get_tool_call(run_id, "0.0")
    assert record.status is ToolCallStatus.AMBIGUOUS
    assert "outcome unknown" in (record.error or "")


# -- follow-ups from re-reading the fixes ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "leak"),
    [
        ("session_token=abc123", "abc123"),
        ("client_secret: abc123", "abc123"),
        ('{"password": "two words here"}', "words"),
        ("password='two words'", "words"),
    ],
)
def test_assignments_inside_strings_are_fully_redacted(text: str, leak: str) -> None:
    assert leak not in redact_sensitive_keys({"error": text})["error"]


def test_token_counts_in_strings_are_left_alone() -> None:
    text = "used max_tokens: 100"
    assert redact_sensitive_keys({"note": text})["note"] == text


def test_cli_resume_of_expired_blocked_run_exits_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTS_FAIL_IN", "send_email")
    assert main(["--data-dir", str(tmp_path), "start", "--agent", agents.REF, "--goal", "g"]) == 3
    run_id = capsys.readouterr().out.split()[2]
    monkeypatch.delenv("AGENTS_FAIL_IN")
    runtime = Runtime(tmp_path)
    runtime.store.conn.execute("UPDATE runs SET deadline_at = 1.0 WHERE id = ?", (run_id,))
    runtime.close()

    assert main(["--data-dir", str(tmp_path), "resume", run_id]) == 1
    assert "budget_exhausted: wall-clock deadline passed" in capsys.readouterr().out
    trace = (tmp_path / "traces" / f"{run_id}.jsonl").read_text(encoding="utf-8")
    assert json.loads(trace.splitlines()[-1])["event"] == "run_returned"


def test_cli_reports_an_unreadable_database_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "agentflow.db").write_bytes(b"this is not a sqlite database" * 100)
    assert main(["--data-dir", str(tmp_path), "runs"]) == 2
    assert "error:" in capsys.readouterr().err


def test_requested_call_of_a_removed_tool_fails_definitely(harness: Harness) -> None:
    agent = make_agent([call("send"), FINAL], [make_tool("send", lambda: {}, idempotent=False)])
    run_id = harness.runtime.create("tests:inline", "goal", agent=agent)
    _simulate_interrupted_call(harness.runtime, run_id, "send", idempotent=False)
    with harness.runtime.store.tx(run_id, None) as tx:
        tx.update_tool_call("0.0", ToolCallStatus.REQUESTED, attempts=0)
    outcome = harness.runtime.run(run_id, agent=make_agent([call("send"), FINAL], []))
    assert outcome.status is RunStatus.COMPLETED  # the model saw the failure and finished
    record = harness.runtime.store.get_tool_call(run_id, "0.0")
    assert record.status is ToolCallStatus.FAILED
    assert "no longer registered" in (record.error or "")


def test_schema_v1_database_is_migrated_and_old_calls_use_current_policy(tmp_path: Path) -> None:
    executed: list[int] = []
    tool = make_tool("send", lambda: executed.append(1) or {}, idempotent=False)
    agent = make_agent([call("send"), FINAL], [tool])
    runtime = Runtime(tmp_path)
    run_id = runtime.create("tests:inline", "goal", agent=agent)
    _simulate_interrupted_call(runtime, run_id, "send", idempotent=False)
    conn = runtime.store.conn  # rewind to the v1 layout, with data in it
    conn.execute("ALTER TABLE tool_calls DROP COLUMN idempotent")
    conn.execute("ALTER TABLE tool_calls DROP COLUMN max_attempts")
    conn.execute("PRAGMA user_version = 1")
    runtime.close()

    migrated = Runtime(tmp_path)
    record = migrated.store.get_tool_call(run_id, "0.0")
    assert (record.idempotent, record.max_attempts) == (None, None)
    assert migrated.run(run_id, agent=agent).status is RunStatus.BLOCKED
    assert executed == []
    migrated.close()
    reopened = Runtime(tmp_path)  # a second open must not try to migrate again
    assert reopened.store.get_run(run_id).status is RunStatus.BLOCKED
    reopened.close()


def test_short_sk_keys_are_redacted_but_words_containing_sk_are_not() -> None:
    out = redact_sensitive_keys({"goal": "use key sk-LIVE999 for this task-manager"})
    assert out["goal"] == f"use key {REDACTED} for this task-manager"
