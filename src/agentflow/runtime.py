"""The run loop.

``Runtime.run`` claims a run and drives it until it is completed, failed, cancelled or blocked.
It is the same function for a fresh run and for a resumed one: all progress is read back from the
store at every step, so there is no separate "recovery mode".

Guarantees, stated exactly:

* A model turn and the tool calls it requested are committed in one transaction. A crash before
  that commit loses the turn and the model is asked again on resume (a model call has no side
  effect besides cost).
* A tool call whose result was committed is never executed again; on resume the recorded result
  is what the model sees.
* Before each tool attempt the call is committed as ``started``. If the process dies before the
  result is committed, the attempt is *ambiguous*: on resume an idempotent tool is executed again
  (bounded by its ``max_attempts``); a non-idempotent tool is marked ``ambiguous`` and the run moves
  to ``blocked`` until an operator resolves it with ``Runtime.resolve``.
* A failed attempt that will be retried is committed back to ``requested`` with its attempt count,
  so a crash during the backoff sleep resumes as a clean retry, not as an ambiguous call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from agentflow.agent import Agent, load_agent
from agentflow.messages import Message, ModelResponse
from agentflow.providers.base import ProviderError, TransientProviderError
from agentflow.schema import parse_json_output, validation_errors
from agentflow.state import TERMINAL_CALL_STATUSES, TERMINAL_RUN_STATUSES, RunStatus, ToolCallStatus
from agentflow.store import Store, ToolCallRecord, Tx
from agentflow.tools import (
    Tool,
    ToolError,
    ToolTimeout,
    TransientToolError,
    invoke_with_timeout,
)
from agentflow.trace import TraceWriter

Listener = Callable[[dict[str, Any]], None]

OUTPUT_INSTRUCTION = (
    "When you have finished, reply with only a JSON object (no prose) that matches this JSON "
    "Schema:\n{schema}"
)
REPAIR_INSTRUCTION = (
    "Your final answer was rejected by schema validation:\n{errors}\n"
    "Reply again with only a corrected JSON object. Repair attempt {n} of {limit}."
)


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: RunStatus
    result: Any
    error: str | None


class _Step(Enum):
    DONE = "done"  # the call reached a terminal status
    STOP = "stop"  # the run left waiting_tool (cancelled, failed or blocked)


class Runtime:
    def __init__(
        self,
        data_dir: Path,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        listeners: Sequence[Listener] = (),
    ) -> None:
        self.data_dir = data_dir
        self.store = Store(data_dir / "agentflow.db")
        self.trace_dir = data_dir / "traces"
        self.sleep = sleep
        self.clock = clock
        self.listeners = list(listeners)

    def close(self) -> None:
        self.store.close()

    # -- public API ---------------------------------------------------------------------------

    def create(self, agent_ref: str, goal: str, agent: Agent | None = None) -> str:
        """Persist a new ``pending`` run. Nothing executes until ``run``."""
        agent = agent or load_agent(agent_ref, self.data_dir)
        system = (
            agent.system_prompt
            + "\n\n"
            + OUTPUT_INSTRUCTION.format(schema=json.dumps(agent.output_schema, sort_keys=True))
        )
        limits = agent.limits
        run_id = self.store.create_run(
            agent_ref=agent_ref,
            goal=goal,
            messages=[Message(role="system", content=system), Message(role="user", content=goal)],
            max_steps=limits.max_steps,
            max_tool_calls=limits.max_tool_calls,
            max_repairs=limits.max_repairs,
            deadline_at=None if limits.max_wall_s is None else self.clock() + limits.max_wall_s,
        )
        TraceWriter(self.trace_dir, agent.redactor).write(
            run_id, "run_created", agent=agent.name, agent_ref=agent_ref, goal=goal
        )
        return run_id

    def run(self, run_id: str, agent: Agent | None = None) -> RunOutcome:
        """Claim the run and drive it. Used both to start and to resume."""
        record = self.store.get_run(run_id)
        if record.status in TERMINAL_RUN_STATUSES:
            return self.outcome(run_id)
        agent = agent or load_agent(record.agent_ref, self.data_dir)
        token = self.store.claim(run_id)
        return _Driver(self, agent, run_id, token).drive()

    def cancel(self, run_id: str) -> RunStatus:
        """Cancel now if no runner can be mid-step, otherwise flag it for the runner.

        A ``pending`` or ``blocked`` run is cancelled immediately. For ``running`` and
        ``waiting_tool`` the flag is checked by the runner before every model call and tool
        attempt; if the runner has died, the next ``run`` applies it before doing anything.
        """
        record = self.store.get_run(run_id)
        if record.status in TERMINAL_RUN_STATUSES:
            return record.status
        with self.store.tx(run_id, None) as tx:
            tx.conn.execute("UPDATE runs SET cancel_requested = 1 WHERE id = ?", (run_id,))
            current = tx.status()
            if current in (RunStatus.PENDING, RunStatus.BLOCKED):
                tx.set_outcome(error="cancelled by operator")
                tx.set_status(RunStatus.CANCELLED, "cancelled by operator")
                status = RunStatus.CANCELLED
            else:
                status = current
        TraceWriter(self.trace_dir).write(run_id, "cancel_requested", status=status.value)
        return status

    def resolve(
        self,
        run_id: str,
        call_id: str,
        action: Literal["retry", "fail", "succeed"],
        value: Any = None,
    ) -> None:
        """Operator decision for an ``ambiguous`` call on a ``blocked`` run.

        ``retry``: run the call again (the operator accepts the risk of a duplicate effect).
        ``fail``: record it as failed with message ``value``; the model sees the failure.
        ``succeed``: record ``value`` as its result (for example after checking the external
        system by hand); the model sees that result.
        Call ``run`` afterwards to continue.
        """
        record = self.store.get_run(run_id)
        call = self.store.get_tool_call(run_id, call_id)
        if record.status is not RunStatus.BLOCKED or call.status is not ToolCallStatus.AMBIGUOUS:
            raise ValueError(
                f"call {call_id} is {call.status} on a {record.status} run; only ambiguous calls"
                " on blocked runs can be resolved"
            )
        with self.store.tx(run_id, None) as tx:
            if action == "retry":
                tx.update_tool_call(
                    call_id, ToolCallStatus.REQUESTED, attempts=0, error="operator chose retry"
                )
            elif action == "fail":
                message = str(value or "operator marked the call as failed")
                tx.update_tool_call(call_id, ToolCallStatus.FAILED, error=message, finished=True)
                tx.add_message(_tool_message(call, {"ok": False, "error": message}))
            elif action == "succeed":
                tx.update_tool_call(call_id, ToolCallStatus.SUCCEEDED, result=value, finished=True)
                tx.add_message(_tool_message(call, {"ok": True, "result": value}))
            else:
                raise ValueError(f"unknown action {action!r}")
        TraceWriter(self.trace_dir).write(
            run_id, "operator_resolved", call_id=call_id, tool=call.tool, action=action
        )

    def outcome(self, run_id: str) -> RunOutcome:
        record = self.store.get_run(run_id)
        return RunOutcome(run_id, record.status, record.result, record.error)


def _tool_message(call: ToolCallRecord, payload: dict[str, Any]) -> Message:
    return Message(
        role="tool",
        content=json.dumps(payload, sort_keys=True),
        tool_call_id=call.provider_id,
        name=call.tool,
    )


class _Driver:
    """State for one claim of one run. Discarded when ``drive`` returns."""

    def __init__(self, runtime: Runtime, agent: Agent, run_id: str, token: str) -> None:
        self.rt = runtime
        self.store = runtime.store
        self.agent = agent
        self.run_id = run_id
        self.token = token
        self.trace = TraceWriter(runtime.trace_dir, agent.redactor)
        # Events describing a commit are buffered and written only after the commit succeeds.
        self._pending_events: list[tuple[str, dict[str, Any]]] = []

    # -- plumbing -----------------------------------------------------------------------------

    def emit(self, event: str, **fields: Any) -> None:
        record = self.trace.write(self.run_id, event, **fields)
        for listener in self.rt.listeners:
            listener(record)

    def transition(
        self, target: RunStatus, reason: str, *, result: Any = None, error: str | None = None
    ) -> None:
        with self.store.tx(self.run_id, self.token) as tx:
            previous = self._apply_transition(tx, target, reason, result, error)
        self.emit("state", **{"from": previous.value, "to": target.value, "reason": reason})

    @staticmethod
    def _apply_transition(
        tx: Tx, target: RunStatus, reason: str, result: Any, error: str | None
    ) -> RunStatus:
        if result is not None or error is not None:
            tx.set_outcome(result=result, error=error)
        return tx.set_status(target, reason)

    def _stop_reason(self) -> tuple[RunStatus, str] | None:
        record = self.store.get_run(self.run_id)
        if record.cancel_requested:
            return RunStatus.CANCELLED, "cancelled by operator"
        if record.deadline_at is not None and self.rt.clock() >= record.deadline_at:
            return RunStatus.FAILED, "budget_exhausted: wall-clock deadline passed"
        return None

    def _stop_if_needed(self) -> bool:
        stop = self._stop_reason()
        if stop is None:
            return False
        status, reason = stop
        self.transition(status, reason, error=reason)
        return True

    def _remaining_s(self) -> float | None:
        deadline = self.store.get_run(self.run_id).deadline_at
        return None if deadline is None else deadline - self.rt.clock()

    # -- main loop ----------------------------------------------------------------------------

    def drive(self) -> RunOutcome:
        record = self.store.get_run(self.run_id)
        self.emit(
            "run_claimed",
            status=record.status.value,
            model_calls=record.model_calls,
            tool_calls=record.tool_call_count,
        )
        if record.status is RunStatus.PENDING:
            self.transition(RunStatus.RUNNING, "started")
        elif record.status is RunStatus.BLOCKED:
            if self._stop_if_needed():
                return self.rt.outcome(self.run_id)
            unresolved = [
                c.call_id
                for c in self.store.tool_calls(self.run_id)
                if c.status is ToolCallStatus.AMBIGUOUS
            ]
            if unresolved:
                self.emit("still_blocked", ambiguous_calls=unresolved)
                return self.rt.outcome(self.run_id)
            self.transition(RunStatus.WAITING_TOOL, "ambiguous calls resolved by operator")

        while True:
            status = self.store.get_run(self.run_id).status
            if status in TERMINAL_RUN_STATUSES or status is RunStatus.BLOCKED:
                outcome = self.rt.outcome(self.run_id)
                self.emit("run_returned", status=outcome.status.value, error=outcome.error)
                return outcome
            if self._stop_if_needed():
                continue
            if status is RunStatus.WAITING_TOOL:
                self._run_pending_tools()
            elif status is RunStatus.RUNNING:
                self._model_turn()

    # -- model turns --------------------------------------------------------------------------

    def _call_model(self, messages: list[Message]) -> ModelResponse | None:
        attempts = 1 + self.agent.limits.model_retries
        for attempt in range(1, attempts + 1):
            try:
                return self.agent.provider.complete(
                    messages, self.agent.tools.specs(), self.agent.output_schema
                )
            except TransientProviderError as exc:
                if attempt == attempts:
                    reason = f"provider_unavailable after {attempts} attempts: {exc}"
                    self.transition(RunStatus.FAILED, reason, error=reason)
                    return None
                self.emit("model_retry", attempt=attempt, error=str(exc))
                self.rt.sleep(min(5.0, 0.5 * 2 ** (attempt - 1)))
                if self._stop_if_needed():
                    return None
            except ProviderError as exc:
                reason = f"provider_error: {exc}"
                self.transition(RunStatus.FAILED, reason, error=reason)
                return None
        raise AssertionError("unreachable")  # pragma: no cover

    def _model_turn(self) -> None:
        record = self.store.get_run(self.run_id)
        if record.model_calls >= record.max_steps:
            reason = f"budget_exhausted: max_steps={record.max_steps} model calls"
            self.transition(RunStatus.FAILED, reason, error=reason)
            return
        turn = record.model_calls
        started = time.perf_counter()
        response = self._call_model(self.store.messages(self.run_id))
        if response is None:
            return
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        summary: Any = (
            [c.to_json() for c in response.tool_calls] if response.tool_calls else response.content
        )
        if response.tool_calls:
            self._record_tool_turn(turn, response)
        else:
            self._record_final_turn(response)
        # Emitted after the commit so the trace never shows a turn the store does not have.
        self.emit(
            "model_call", turn=turn, duration_ms=duration_ms, response=summary, usage=response.usage
        )
        self._emit_after_turn()

    def _emit_after_turn(self) -> None:
        for event, fields in self._pending_events:
            self.emit(event, **fields)
        self._pending_events = []

    def _record_tool_turn(self, turn: int, response: ModelResponse) -> None:
        record = self.store.get_run(self.run_id)
        events: list[tuple[str, dict[str, Any]]] = []
        with self.store.tx(self.run_id, self.token) as tx:
            tx.bump("model_calls")
            tx.add_message(response.to_message())
            total = tx.bump("tool_call_count", len(response.tool_calls))
            if total > record.max_tool_calls:
                reason = f"budget_exhausted: max_tool_calls={record.max_tool_calls}"
                previous = self._apply_transition(tx, RunStatus.FAILED, reason, None, reason)
                events.append(("state", {"from": previous.value, "to": "failed", "reason": reason}))
                self._pending_events = events
                return
            rejected: list[str] = []
            for idx, request in enumerate(response.tool_calls):
                call_id = f"{turn}.{idx}"
                errors = self.agent.tools.check_call(request.name, request.arguments)
                if errors:
                    error = "; ".join(errors)
                    rejected.append(f"{call_id} {request.name}: {error}")
                    tx.add_tool_call(call_id, turn, idx, request, ToolCallStatus.REJECTED, error)
                    tx.add_message(
                        Message(
                            role="tool",
                            content=json.dumps(
                                {"ok": False, "error": f"invalid tool call, not executed: {error}"}
                            ),
                            tool_call_id=request.id,
                            name=request.name,
                        )
                    )
                    events.append(
                        (
                            "tool_rejected",
                            {
                                "call_id": call_id,
                                "tool": request.name,
                                "arguments": request.arguments,
                                "error": error,
                            },
                        )
                    )
                else:
                    tx.add_tool_call(call_id, turn, idx, request, ToolCallStatus.REQUESTED)
            if rejected:
                repairs = tx.bump("repairs")
                events.append(("repair", {"kind": "tool_arguments", "repairs": repairs}))
                if repairs > record.max_repairs:
                    reason = (
                        f"invalid_tool_call: repair limit {record.max_repairs} exceeded"
                        f" ({'; '.join(rejected)})"
                    )
                    previous = self._apply_transition(tx, RunStatus.FAILED, reason, None, reason)
                    events.append(
                        ("state", {"from": previous.value, "to": "failed", "reason": reason})
                    )
                    self._pending_events = events
                    return
            previous = self._apply_transition(
                tx, RunStatus.WAITING_TOOL, "model requested tools", None, None
            )
            events.append(
                (
                    "state",
                    {
                        "from": previous.value,
                        "to": "waiting_tool",
                        "reason": "model requested tools",
                    },
                )
            )
        self._pending_events = events

    def _record_final_turn(self, response: ModelResponse) -> None:
        record = self.store.get_run(self.run_id)
        value, parse_error = parse_json_output(response.content)
        errors = (
            [parse_error] if parse_error else validation_errors(self.agent.output_validator, value)
        )
        events: list[tuple[str, dict[str, Any]]] = []
        with self.store.tx(self.run_id, self.token) as tx:
            tx.bump("model_calls")
            tx.add_message(response.to_message())
            if not errors:
                previous = self._apply_transition(
                    tx, RunStatus.COMPLETED, "output valid", value, None
                )
                events.append(
                    ("state", {"from": previous.value, "to": "completed", "reason": "output valid"})
                )
            else:
                repairs = tx.bump("repairs")
                events.append(("repair", {"kind": "output", "repairs": repairs, "errors": errors}))
                if repairs > record.max_repairs:
                    reason = (
                        f"output_invalid: repair limit {record.max_repairs} exceeded; last errors: "
                        + "; ".join(errors)
                    )
                    previous = self._apply_transition(tx, RunStatus.FAILED, reason, None, reason)
                    events.append(
                        ("state", {"from": previous.value, "to": "failed", "reason": reason})
                    )
                else:
                    tx.add_message(
                        Message(
                            role="user",
                            content=REPAIR_INSTRUCTION.format(
                                errors="\n".join(f"- {e}" for e in errors),
                                n=repairs,
                                limit=record.max_repairs,
                            ),
                        )
                    )
        self._pending_events = events

    # -- tool calls ---------------------------------------------------------------------------

    def _run_pending_tools(self) -> None:
        for call in self.store.tool_calls(self.run_id):
            if call.status in TERMINAL_CALL_STATUSES:
                continue
            if self._handle_call(call) is _Step.STOP:
                return
        self.transition(RunStatus.RUNNING, "tool results recorded")

    def _handle_call(self, call: ToolCallRecord) -> _Step:
        tool = self.agent.tools.get(call.tool)
        if tool is None:
            return self._finish_call(call, None, f"tool {call.tool!r} is no longer registered")
        if call.status is ToolCallStatus.REQUESTED:
            return self._execute(call, tool)
        if call.status is ToolCallStatus.STARTED:
            # The previous process started this attempt and died before recording a result.
            self.emit(
                "tool_interrupted", call_id=call.call_id, tool=call.tool, attempt=call.attempts
            )
            if tool.idempotent:
                if call.attempts < tool.retry.max_attempts:
                    return self._execute(call, tool)
                return self._finish_call(
                    call,
                    None,
                    f"attempts exhausted ({call.attempts}); the last attempt was interrupted",
                )
            return self._block(call, "attempt was interrupted before its result was recorded")
        if call.status is ToolCallStatus.AMBIGUOUS:
            return self._block(call, call.error or "unresolved ambiguous call")
        raise AssertionError(f"unexpected call status {call.status}")  # pragma: no cover

    def _block(self, call: ToolCallRecord, why: str) -> _Step:
        reason = (
            f"tool call {call.call_id} ({call.tool}) is not idempotent and its outcome is unknown:"
            f" {why}"
        )
        with self.store.tx(self.run_id, self.token) as tx:
            tx.update_tool_call(
                call.call_id, ToolCallStatus.AMBIGUOUS, attempts=call.attempts, error=why
            )
            previous = self._apply_transition(tx, RunStatus.BLOCKED, reason, None, None)
        self.emit("tool_ambiguous", call_id=call.call_id, tool=call.tool, reason=why)
        self.emit("state", **{"from": previous.value, "to": "blocked", "reason": reason})
        return _Step.STOP

    def _finish_call(self, call: ToolCallRecord, attempt: int | None, error: str) -> _Step:
        with self.store.tx(self.run_id, self.token) as tx:
            tx.update_tool_call(
                call.call_id, ToolCallStatus.FAILED, attempts=attempt, error=error, finished=True
            )
            tx.add_message(_tool_message(call, {"ok": False, "error": error}))
        self.emit("tool_failed", call_id=call.call_id, tool=call.tool, error=error)
        return _Step.DONE

    def _execute(self, call: ToolCallRecord, tool: Tool) -> _Step:
        attempt = call.attempts
        while True:
            attempt += 1
            delay = tool.retry.delay_before(attempt)
            if delay:
                self.rt.sleep(delay)
            if self._stop_if_needed():
                return _Step.STOP
            remaining = self._remaining_s()
            timeout = (
                tool.timeout_s if remaining is None else max(0.0, min(tool.timeout_s, remaining))
            )
            with self.store.tx(self.run_id, self.token) as tx:
                tx.update_tool_call(
                    call.call_id, ToolCallStatus.STARTED, attempts=attempt, started=True
                )
            self.emit(
                "tool_started",
                call_id=call.call_id,
                tool=call.tool,
                attempt=attempt,
                arguments=call.arguments,
            )
            started = time.perf_counter()
            try:
                value = invoke_with_timeout(tool, call.arguments, timeout)
            except TransientToolError as exc:
                kind, error = "transient", f"{type(exc).__name__}: {exc}"
            except ToolError as exc:
                kind, error = "permanent", f"{type(exc).__name__}: {exc}"
            except ToolTimeout as exc:
                kind, error = "ambiguous", str(exc)
            except Exception as exc:
                kind, error = "ambiguous", f"{type(exc).__name__}: {exc}"
            else:
                with self.store.tx(self.run_id, self.token) as tx:
                    tx.update_tool_call(
                        call.call_id, ToolCallStatus.SUCCEEDED, result=value, finished=True
                    )
                    tx.add_message(_tool_message(call, {"ok": True, "result": value}))
                self.emit(
                    "tool_finished",
                    call_id=call.call_id,
                    tool=call.tool,
                    attempt=attempt,
                    result=value,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
                return _Step.DONE

            if kind == "ambiguous" and not tool.idempotent:
                return self._block(replace(call, attempts=attempt), error)
            retryable = kind == "transient" or kind == "ambiguous"
            if retryable and attempt < tool.retry.max_attempts:
                with self.store.tx(self.run_id, self.token) as tx:
                    tx.update_tool_call(
                        call.call_id, ToolCallStatus.REQUESTED, attempts=attempt, error=error
                    )
                self.emit(
                    "tool_retry",
                    call_id=call.call_id,
                    tool=call.tool,
                    attempt=attempt,
                    error=error,
                    next_delay_s=tool.retry.delay_before(attempt + 1),
                )
                continue
            if retryable:
                error = f"attempts exhausted ({attempt}): {error}"
            return self._finish_call(call, attempt, error)
