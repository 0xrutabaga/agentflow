"""Run state machine.

The allowed transitions live in one table, ``TRANSITIONS``. The store refuses any status change
that is not in it, and the README's Mermaid diagram is generated from it (a test checks that the
two stay identical).
"""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolCallStatus(StrEnum):
    REQUESTED = "requested"  # recorded from the model, never started
    STARTED = "started"  # an attempt began; no result recorded yet
    SUCCEEDED = "succeeded"
    FAILED = "failed"  # definite failure, recorded and shown to the model
    REJECTED = "rejected"  # arguments failed validation; never executed
    AMBIGUOUS = "ambiguous"  # non-idempotent call that may or may not have taken effect


TERMINAL_RUN_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})

TERMINAL_CALL_STATUSES = frozenset(
    {ToolCallStatus.SUCCEEDED, ToolCallStatus.FAILED, ToolCallStatus.REJECTED}
)

TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset(
        {RunStatus.WAITING_TOOL, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.WAITING_TOOL: frozenset(
        {RunStatus.RUNNING, RunStatus.BLOCKED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    # blocked -> failed: the wall-clock deadline keeps running while a run waits for an operator.
    RunStatus.BLOCKED: frozenset({RunStatus.WAITING_TOOL, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}


class InvalidTransition(RuntimeError):
    """Raised when code asks for a status change that ``TRANSITIONS`` does not allow."""


def check_transition(current: RunStatus, target: RunStatus) -> None:
    if target not in TRANSITIONS[current]:
        raise InvalidTransition(f"run cannot move from {current} to {target}")


def mermaid_diagram() -> str:
    """Render ``TRANSITIONS`` as a Mermaid state diagram (used verbatim in the README)."""
    lines = ["stateDiagram-v2", "    [*] --> pending"]
    for source in RunStatus:
        for target in sorted(TRANSITIONS[source], key=list(RunStatus).index):
            lines.append(f"    {source.value} --> {target.value}")
    for status in RunStatus:
        if status in TERMINAL_RUN_STATUSES:
            lines.append(f"    {status.value} --> [*]")
    return "\n".join(lines)
