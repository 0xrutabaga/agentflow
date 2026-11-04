from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import FINAL, make_agent

from agentflow.runtime import Runtime
from agentflow.state import (
    TERMINAL_RUN_STATUSES,
    TRANSITIONS,
    InvalidTransition,
    RunStatus,
    check_transition,
    mermaid_diagram,
)

README = Path(__file__).resolve().parent.parent / "README.md"


def test_terminal_states_have_no_exits() -> None:
    for status in TERMINAL_RUN_STATUSES:
        assert TRANSITIONS[status] == frozenset()


def test_every_status_is_reachable_from_pending() -> None:
    seen = {RunStatus.PENDING}
    frontier = [RunStatus.PENDING]
    while frontier:
        for target in TRANSITIONS[frontier.pop()]:
            if target not in seen:
                seen.add(target)
                frontier.append(target)
    assert seen == set(RunStatus)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunStatus.COMPLETED, RunStatus.RUNNING),
        (RunStatus.PENDING, RunStatus.COMPLETED),
        (RunStatus.BLOCKED, RunStatus.RUNNING),
        (RunStatus.RUNNING, RunStatus.BLOCKED),
    ],
)
def test_illegal_transitions_are_refused(current: RunStatus, target: RunStatus) -> None:
    with pytest.raises(InvalidTransition):
        check_transition(current, target)


def test_store_enforces_the_transition_table(tmp_path: Path) -> None:
    runtime = Runtime(tmp_path)
    agent = make_agent([FINAL])
    run_id = runtime.create("tests:inline", "goal", agent=agent)
    with pytest.raises(InvalidTransition), runtime.store.tx(run_id, None) as tx:
        tx.set_status(RunStatus.COMPLETED)
    assert runtime.store.get_run(run_id).status is RunStatus.PENDING  # rolled back
    runtime.close()


def test_readme_state_diagram_matches_the_code() -> None:
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n(stateDiagram-v2\n.*?)\n```", text, flags=re.S)
    assert blocks, "README has no Mermaid state diagram"
    assert blocks[0].strip() == mermaid_diagram()
