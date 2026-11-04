"""In-process behaviour of the run loop: retries, idempotency, repair, budgets, cancellation."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from conftest import FINAL, Harness, call, make_agent, make_tool

from agentflow.agent import Limits
from agentflow.messages import Message, ModelResponse
from agentflow.providers.base import ProviderError, TransientProviderError
from agentflow.runtime import Runtime
from agentflow.state import RunStatus, ToolCallStatus
from agentflow.store import LostOwnership
from agentflow.tools import ToolError, TransientToolError


class Counter:
    def __init__(self, fail_first: int = 0, exc: BaseException | None = None) -> None:
        self.calls = 0
        self.fail_first = fail_first
        self.exc = exc or ConnectionError("backend unavailable")

    def __call__(self) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise self.exc
        return {"n": self.calls}


def tool_payloads(h: Harness, run_id: str) -> list[dict[str, Any]]:
    return [
        json.loads(m.content or "{}") for m in h.runtime.store.messages(run_id) if m.role == "tool"
    ]


# -- happy path and recorded results ------------------------------------------------------


def test_tool_then_final_answer_completes(harness: Harness) -> None:
    fn = Counter()
    agent = make_agent([call("t"), FINAL], [make_tool("t", fn)])
    run_id, outcome = harness.start(agent)
    assert outcome.status is RunStatus.COMPLETED
    assert outcome.result == {"answer": "done"}
    assert fn.calls == 1
    assert tool_payloads(harness, run_id) == [{"ok": True, "result": {"n": 1}}]
    statuses = [t[1] for t in harness.runtime.store.transitions(run_id)]
    assert statuses == ["pending", "running", "waiting_tool", "running", "completed"]


def test_running_a_finished_run_again_is_a_no_op(harness: Harness) -> None:
    fn = Counter()
    agent = make_agent([call("t"), FINAL], [make_tool("t", fn)])
    run_id, _ = harness.start(agent)
    again = harness.runtime.run(run_id, agent=agent)
    assert again.status is RunStatus.COMPLETED
    assert fn.calls == 1


# -- retries and idempotency --------------------------------------------------------------


def test_flaky_idempotent_tool_is_retried_with_backoff(harness: Harness) -> None:
    fn = Counter(fail_first=2)
    run_id, outcome = harness.start(make_agent([call("t"), FINAL], [make_tool("t", fn)]))
    assert outcome.status is RunStatus.COMPLETED
    assert fn.calls == 3
    record = harness.runtime.store.get_tool_call(run_id, "0.0")
    assert (record.status, record.attempts) == (ToolCallStatus.SUCCEEDED, 3)
    assert harness.sleeps == [0.2, 0.4]


def test_idempotent_tool_failing_every_attempt_is_reported_to_the_model(harness: Harness) -> None:
    fn = Counter(fail_first=99)
    run_id, outcome = harness.start(make_agent([call("t"), FINAL], [make_tool("t", fn)]))
    assert outcome.status is RunStatus.COMPLETED  # the model decided to finish anyway
    assert fn.calls == 3
    [payload] = tool_payloads(harness, run_id)
    assert payload["ok"] is False
    assert "attempts exhausted (3)" in payload["error"]


def test_non_idempotent_tool_is_not_retried_after_ambiguous_failure(harness: Harness) -> None:
    fn = Counter(fail_first=1)
    tool = make_tool("charge", fn, idempotent=False)
    run_id, outcome = harness.start(make_agent([call("charge"), FINAL], [tool]))
    assert outcome.status is RunStatus.BLOCKED
    assert fn.calls == 1
    record = harness.runtime.store.get_tool_call(run_id, "0.0")
    assert record.status is ToolCallStatus.AMBIGUOUS
    assert "ConnectionError" in (record.error or "")


def test_non_idempotent_transient_error_is_retried(harness: Harness) -> None:
    fn = Counter(fail_first=1, exc=TransientToolError("rejected before sending"))
    tool = make_tool("charge", fn, idempotent=False)
    _, outcome = harness.start(make_agent([call("charge"), FINAL], [tool]))
    assert outcome.status is RunStatus.COMPLETED
    assert fn.calls == 2


def test_permanent_tool_error_is_not_retried(harness: Harness) -> None:
    fn = Counter(fail_first=1, exc=ToolError("no such alert"))
    run_id, outcome = harness.start(make_agent([call("t"), FINAL], [make_tool("t", fn)]))
    assert outcome.status is RunStatus.COMPLETED
    assert fn.calls == 1
    assert tool_payloads(harness, run_id)[0] == {"ok": False, "error": "ToolError: no such alert"}


def test_resolve_fail_feeds_failure_to_model_and_completes(harness: Harness) -> None:
    fn = Counter(fail_first=1)
    agent = make_agent([call("charge"), FINAL], [make_tool("charge", fn, idempotent=False)])
    run_id, _ = harness.start(agent)
    harness.runtime.resolve(run_id, "0.0", "fail", "card was not charged (checked by hand)")
    outcome = harness.runtime.run(run_id, agent=agent)
    assert outcome.status is RunStatus.COMPLETED
    assert fn.calls == 1
    assert tool_payloads(harness, run_id)[-1]["error"] == "card was not charged (checked by hand)"


def test_resolve_rejects_calls_that_are_not_ambiguous(harness: Harness) -> None:
    run_id, _ = harness.start(make_agent([call("t"), FINAL], [make_tool("t", Counter())]))
    with pytest.raises(ValueError, match="only ambiguous calls"):
        harness.runtime.resolve(run_id, "0.0", "retry")


def test_crash_during_retry_backoff_resumes_as_clean_retry(tmp_path: Path) -> None:
    """A failed attempt is committed back to `requested` before the backoff sleep."""

    class Crash(BaseException):
        pass

    def crash_on_retry(event: dict[str, Any]) -> None:
        if event["event"] == "tool_retry":
            raise Crash

    fn = Counter(fail_first=1, exc=TransientToolError("rate limited"))
    agent = make_agent([call("charge"), FINAL], [make_tool("charge", fn, idempotent=False)])
    first = Runtime(tmp_path, sleep=lambda _: None, listeners=[crash_on_retry])
    run_id = first.create("tests:inline", "goal", agent=agent)
    with pytest.raises(Crash):
        first.run(run_id, agent=agent)
    record = first.store.get_tool_call(run_id, "0.0")
    assert (record.status, record.attempts) == (ToolCallStatus.REQUESTED, 1)
    first.close()

    second = Runtime(tmp_path, sleep=lambda _: None)
    assert second.run(run_id, agent=agent).status is RunStatus.COMPLETED
    assert second.store.get_tool_call(run_id, "0.0").attempts == 2
    second.close()


def test_interrupted_idempotent_call_with_no_attempts_left_fails(harness: Harness) -> None:
    fn = Counter()
    agent = make_agent([call("t"), FINAL], [make_tool("t", fn, max_attempts=2)])
    run_id = harness.runtime.create("tests:inline", "goal", agent=agent)
    store = harness.runtime.store
    # Simulate a process that died during its last allowed attempt.
    with store.tx(run_id, None) as tx:
        tx.set_status(RunStatus.RUNNING)
        tx.bump("model_calls")
        tx.add_message(call("t").to_message())
        tx.add_tool_call("0.0", 0, 0, call("t").tool_calls[0], ToolCallStatus.STARTED)
        tx.update_tool_call("0.0", ToolCallStatus.STARTED, attempts=2)
        tx.set_status(RunStatus.WAITING_TOOL)
    outcome = harness.runtime.run(run_id, agent=agent)
    assert outcome.status is RunStatus.COMPLETED
    assert fn.calls == 0
    assert "the last attempt was interrupted" in (store.get_tool_call(run_id, "0.0").error or "")


# -- timeouts -----------------------------------------------------------------------------


def test_idempotent_timeout_is_retried_then_reported(harness: Harness) -> None:
    def slow() -> None:
        time.sleep(1.0)

    tool = make_tool("slow", slow, timeout_s=0.05, max_attempts=2)
    run_id, outcome = harness.start(make_agent([call("slow"), FINAL], [tool]))
    assert outcome.status is RunStatus.COMPLETED
    record = harness.runtime.store.get_tool_call(run_id, "0.0")
    assert record.status is ToolCallStatus.FAILED
    assert record.attempts == 2
    assert "did not finish within 0.05s" in (record.error or "")


def test_non_idempotent_timeout_blocks(harness: Harness) -> None:
    def slow() -> None:
        time.sleep(1.0)

    tool = make_tool("send", slow, idempotent=False, timeout_s=0.05)
    run_id, outcome = harness.start(make_agent([call("send"), FINAL], [tool]))
    assert outcome.status is RunStatus.BLOCKED
    record = harness.runtime.store.get_tool_call(run_id, "0.0")
    assert (record.status, record.attempts) == (ToolCallStatus.AMBIGUOUS, 1)


# -- structured output and argument repair ------------------------------------------------


def test_invalid_output_is_repaired_with_the_validation_error(harness: Harness) -> None:
    bad = ModelResponse(content='{"answer": 42}')
    run_id, outcome = harness.start(make_agent([bad, FINAL]))
    assert outcome.status is RunStatus.COMPLETED
    run = harness.runtime.store.get_run(run_id)
    assert run.repairs == 1
    repair = [m for m in harness.runtime.store.messages(run_id) if m.role == "user"][-1]
    assert "$.answer: 42 is not of type 'string'" in (repair.content or "")


def test_output_that_never_validates_fails_after_the_repair_limit(harness: Harness) -> None:
    bad = ModelResponse(content="the answer is done")
    agent = make_agent([bad, bad, bad, FINAL], limits=Limits(max_repairs=2, max_wall_s=None))
    run_id, outcome = harness.start(agent)
    assert outcome.status is RunStatus.FAILED
    assert outcome.error is not None
    assert outcome.error.startswith("output_invalid: repair limit 2 exceeded")
    assert "not valid JSON" in outcome.error
    assert harness.runtime.store.get_run(run_id).model_calls == 3


def test_code_fenced_json_output_is_accepted(harness: Harness) -> None:
    fenced = ModelResponse(content='```json\n{"answer": "done"}\n```')
    _, outcome = harness.start(make_agent([fenced]))
    assert outcome.result == {"answer": "done"}


def test_invalid_tool_arguments_are_rejected_without_executing(harness: Harness) -> None:
    fn = Counter()
    params = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }
    tool = make_tool("t", lambda n: fn(), parameters=params)
    run_id, outcome = harness.start(
        make_agent([call("t", n="seven"), call("t", 1, n=7), FINAL], [tool])
    )
    assert outcome.status is RunStatus.COMPLETED
    assert fn.calls == 1
    first = harness.runtime.store.get_tool_call(run_id, "0.0")
    assert first.status is ToolCallStatus.REJECTED
    assert "$.n: 'seven' is not of type 'integer'" in (first.error or "")
    assert harness.runtime.store.get_run(run_id).repairs == 1


def test_unknown_tool_is_rejected_and_repeated_rejections_fail(harness: Harness) -> None:
    agent = make_agent(
        [call("nope"), call("nope", 1), call("nope", 2)],
        limits=Limits(max_repairs=1, max_wall_s=None),
    )
    _, outcome = harness.start(agent)
    assert outcome.status is RunStatus.FAILED
    assert outcome.error is not None
    assert "invalid_tool_call" in outcome.error
    assert "unknown tool 'nope'" in outcome.error


# -- budgets ------------------------------------------------------------------------------


def test_max_steps_budget(harness: Harness) -> None:
    loop = [call("t", i) for i in range(10)]
    agent = make_agent(
        loop, [make_tool("t", Counter())], limits=Limits(max_steps=3, max_wall_s=None)
    )
    run_id, outcome = harness.start(agent)
    assert outcome.status is RunStatus.FAILED
    assert outcome.error == "budget_exhausted: max_steps=3 model calls"
    assert harness.runtime.store.get_run(run_id).model_calls == 3


def test_max_tool_calls_budget(harness: Harness) -> None:
    fn = Counter()
    loop = [call("t", i) for i in range(10)]
    agent = make_agent(loop, [make_tool("t", fn)], limits=Limits(max_tool_calls=2, max_wall_s=None))
    _, outcome = harness.start(agent)
    assert outcome.status is RunStatus.FAILED
    assert outcome.error == "budget_exhausted: max_tool_calls=2"
    assert fn.calls == 2


def test_wall_clock_deadline_stops_the_run(tmp_path: Path) -> None:
    now = [1000.0]

    def advance() -> dict[str, int]:
        now[0] += 30.0
        return {"ok": 1}

    loop = [call("t", i) for i in range(10)]
    agent = make_agent(loop, [make_tool("t", advance)], limits=Limits(max_wall_s=60.0))
    runtime = Runtime(tmp_path, sleep=lambda _: None, clock=lambda: now[0])
    run_id = runtime.create("tests:inline", "goal", agent=agent)
    outcome = runtime.run(run_id, agent=agent)
    assert outcome.status is RunStatus.FAILED
    assert outcome.error == "budget_exhausted: wall-clock deadline passed"
    assert runtime.store.get_run(run_id).model_calls == 2
    runtime.close()


def test_deadline_caps_the_tool_timeout(tmp_path: Path) -> None:
    def slow() -> None:
        time.sleep(2.0)

    agent = make_agent(
        [call("slow"), FINAL],
        [make_tool("slow", slow, timeout_s=30.0, max_attempts=1)],
        limits=Limits(max_wall_s=0.3),
    )
    runtime = Runtime(tmp_path, sleep=lambda _: None)
    run_id = runtime.create("tests:inline", "goal", agent=agent)
    started = time.monotonic()
    outcome = runtime.run(run_id, agent=agent)
    assert time.monotonic() - started < 1.5
    assert outcome.status is RunStatus.FAILED
    assert outcome.error == "budget_exhausted: wall-clock deadline passed"
    runtime.close()


# -- provider failures --------------------------------------------------------------------


class FlakyProvider:
    def __init__(self, failures: Sequence[Exception]) -> None:
        self.failures = list(failures)
        self.calls = 0

    def complete(self, messages: Sequence[Message], tools: Any, schema: Any) -> ModelResponse:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return FINAL


def test_transient_provider_errors_are_retried(harness: Harness) -> None:
    provider = FlakyProvider([TransientProviderError("429"), TransientProviderError("503")])
    _, outcome = harness.start(make_agent([], provider=provider))
    assert outcome.status is RunStatus.COMPLETED
    assert provider.calls == 3
    assert harness.sleeps == [0.5, 1.0]


def test_provider_unavailable_after_retries_fails(harness: Harness) -> None:
    provider = FlakyProvider([TransientProviderError("503")] * 5)
    _, outcome = harness.start(make_agent([], provider=provider))
    assert outcome.status is RunStatus.FAILED
    assert (outcome.error or "").startswith("provider_unavailable after 3 attempts")


def test_permanent_provider_error_fails_immediately(harness: Harness) -> None:
    provider = FlakyProvider([ProviderError("HTTP 401")])
    _, outcome = harness.start(make_agent([], provider=provider))
    assert outcome.status is RunStatus.FAILED
    assert outcome.error == "provider_error: HTTP 401"
    assert provider.calls == 1


# -- cancellation and fencing -------------------------------------------------------------


def test_cancel_pending_run_is_immediate(harness: Harness) -> None:
    agent = make_agent([FINAL])
    run_id = harness.runtime.create("tests:inline", "goal", agent=agent)
    assert harness.runtime.cancel(run_id) is RunStatus.CANCELLED
    assert harness.runtime.run(run_id, agent=agent).status is RunStatus.CANCELLED


def test_cancel_requested_during_a_tool_stops_before_the_next_model_call(harness: Harness) -> None:
    model_calls = []
    holder: dict[str, str] = {}

    def tool() -> dict[str, bool]:
        harness.runtime.cancel(holder["run_id"])  # an operator cancels while the tool runs
        return {"ok": True}

    def record(messages: Sequence[Message]) -> ModelResponse:
        model_calls.append(len(messages))
        return FINAL

    agent = make_agent([call("t"), record], [make_tool("t", tool)])
    run_id = harness.runtime.create("tests:inline", "goal", agent=agent)
    holder["run_id"] = run_id
    outcome = harness.runtime.run(run_id, agent=agent)
    assert outcome.status is RunStatus.CANCELLED
    assert model_calls == []
    # The tool result that finished before the cancel was still recorded.
    assert harness.runtime.store.get_tool_call(run_id, "0.0").status is ToolCallStatus.SUCCEEDED


def test_cancel_from_another_thread_interrupts_retry_loop(harness: Harness) -> None:
    started = threading.Event()

    def failing() -> None:
        started.set()
        raise ConnectionError("down")

    def sleep(seconds: float) -> None:
        time.sleep(0.05)

    runtime = Runtime(harness.data_dir / "other", sleep=sleep)
    tool = make_tool("t", failing, max_attempts=1000)
    agent = make_agent([call("t"), FINAL], [tool])
    run_id = runtime.create("tests:inline", "goal", agent=agent)
    result: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: result.update(o=runtime.run(run_id, agent=agent)))
    worker.start()
    assert started.wait(5)
    Runtime(harness.data_dir / "other").cancel(run_id)
    worker.join(10)
    assert result["o"].status is RunStatus.CANCELLED
    runtime.close()


def test_cancel_blocked_run(harness: Harness) -> None:
    tool = make_tool("send", Counter(fail_first=1), idempotent=False)
    run_id, outcome = harness.start(make_agent([call("send"), FINAL], [tool]))
    assert outcome.status is RunStatus.BLOCKED
    assert harness.runtime.cancel(run_id) is RunStatus.CANCELLED


def test_stale_runner_is_fenced_out(harness: Harness) -> None:
    holder: dict[str, str] = {}

    def tool() -> dict[str, bool]:
        harness.runtime.store.claim(holder["run_id"])  # another runner takes over
        return {"ok": True}

    agent = make_agent([call("t"), FINAL], [make_tool("t", tool)])
    run_id = harness.runtime.create("tests:inline", "goal", agent=agent)
    holder["run_id"] = run_id
    with pytest.raises(LostOwnership):
        harness.runtime.run(run_id, agent=agent)
    # The stale runner's result was not written.
    assert harness.runtime.store.get_tool_call(run_id, "0.0").status is ToolCallStatus.STARTED


def test_tool_arguments_reach_the_tool(harness: Harness) -> None:
    seen: list[Any] = []
    params = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    tool = make_tool("t", lambda q: seen.append(q) or {"q": q}, parameters=params)
    harness.start(make_agent([call("t", q="abc"), FINAL], [tool]))
    assert seen == ["abc"]


def test_run_cancelled_between_read_and_claim_does_not_start(harness: Harness) -> None:
    agent = make_agent([FINAL])
    run_id = harness.runtime.create("tests:inline", "goal", agent=agent)
    store = harness.runtime.store
    real_claim = store.claim

    def claim_after_cancel(rid: str) -> str:
        harness.runtime.cancel(rid)  # operator acts after run() read `pending`
        return real_claim(rid)

    store.claim = claim_after_cancel  # type: ignore[method-assign]
    outcome = harness.runtime.run(run_id, agent=agent)
    assert outcome.status is RunStatus.CANCELLED
