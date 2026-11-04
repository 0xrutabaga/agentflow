"""Crash-and-resume with a real process kill (Popen.kill: SIGKILL / TerminateProcess)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import agents
import pytest

from agentflow.faults import kill_when_marked
from agentflow.runtime import Runtime
from agentflow.state import RunStatus, ToolCallStatus

TESTS_DIR = Path(__file__).parent


def _spawn(
    data_dir: Path, run_id: str, *extra: str, hang_in: str | None = None
) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(TESTS_DIR), env.get("PYTHONPATH", "")])
    env.pop("AGENTS_HANG_IN", None)
    if hang_in:
        env["AGENTS_HANG_IN"] = hang_in
    return subprocess.Popen(
        [sys.executable, "-m", "agentflow", "--data-dir", str(data_dir), "resume", run_id, *extra],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture
def created(tmp_path: Path) -> tuple[Path, str]:
    runtime = Runtime(tmp_path)
    run_id = runtime.create(agents.REF, "crash test", agent=agents.build(tmp_path))
    runtime.close()
    return tmp_path, run_id


def _counts(data_dir: Path) -> Counter[str]:
    return Counter(e["tool"] for e in agents.executions(data_dir))


def _resume(data_dir: Path, run_id: str) -> tuple[Runtime, RunStatus]:
    runtime = Runtime(data_dir)
    outcome = runtime.run(run_id, agent=agents.build(data_dir))
    return runtime, outcome.status


def test_kill_between_steps_does_not_reexecute_recorded_calls(created: tuple[Path, str]) -> None:
    data_dir, run_id = created
    marker = data_dir / "stall.marker"
    proc = _spawn(
        data_dir, run_id, "--stall-at", "tool_finished:read_a", "--stall-marker", str(marker)
    )
    killed_pid = kill_when_marked(proc, marker)

    runtime = Runtime(data_dir)
    assert runtime.store.get_run(run_id).status is RunStatus.WAITING_TOOL
    assert runtime.store.get_tool_call(run_id, "0.0").status is ToolCallStatus.SUCCEEDED
    runtime.close()

    runtime, status = _resume(data_dir, run_id)
    assert status is RunStatus.COMPLETED
    assert _counts(data_dir) == {"read_a": 1, "slow_read": 1, "send_email": 1}
    by_pid = {(e["pid"], e["tool"]) for e in agents.executions(data_dir)}
    assert (killed_pid, "read_a") in by_pid
    assert (os.getpid(), "read_a") not in by_pid
    runtime.close()


def test_kill_inside_idempotent_tool_retries_it_on_resume(created: tuple[Path, str]) -> None:
    data_dir, run_id = created
    proc = _spawn(data_dir, run_id, hang_in="slow_read")
    kill_when_marked(proc, data_dir / "hang.marker")

    runtime = Runtime(data_dir)
    call = runtime.store.get_tool_call(run_id, "1.0")
    assert (call.status, call.attempts) == (ToolCallStatus.STARTED, 1)
    runtime.close()

    runtime, status = _resume(data_dir, run_id)
    assert status is RunStatus.COMPLETED
    # The interrupted attempt ran once in the killed process and once more on resume.
    assert _counts(data_dir) == {"read_a": 1, "slow_read": 2, "send_email": 1}
    assert runtime.store.get_tool_call(run_id, "1.0").attempts == 2
    runtime.close()


def test_kill_inside_non_idempotent_tool_blocks_for_a_human(created: tuple[Path, str]) -> None:
    data_dir, run_id = created
    proc = _spawn(data_dir, run_id, hang_in="send_email")
    kill_when_marked(proc, data_dir / "hang.marker")

    runtime, status = _resume(data_dir, run_id)
    assert status is RunStatus.BLOCKED
    call = runtime.store.get_tool_call(run_id, "2.0")
    assert call.status is ToolCallStatus.AMBIGUOUS
    assert _counts(data_dir)["send_email"] == 1  # not executed again

    # Resuming without a decision keeps it blocked and still does not execute it.
    assert runtime.run(run_id, agent=agents.build(data_dir)).status is RunStatus.BLOCKED
    assert _counts(data_dir)["send_email"] == 1

    # The operator checked the mail system: the email did go out.
    runtime.resolve(run_id, "2.0", "succeed", {"tool": "send_email", "ok": True})
    assert runtime.run(run_id, agent=agents.build(data_dir)).status is RunStatus.COMPLETED
    assert _counts(data_dir) == {"read_a": 1, "slow_read": 1, "send_email": 1}
    runtime.close()


def test_operator_retry_reexecutes_ambiguous_call(created: tuple[Path, str]) -> None:
    data_dir, run_id = created
    proc = _spawn(data_dir, run_id, hang_in="send_email")
    kill_when_marked(proc, data_dir / "hang.marker")

    runtime, status = _resume(data_dir, run_id)
    assert status is RunStatus.BLOCKED
    runtime.resolve(run_id, "2.0", "retry")
    assert runtime.run(run_id, agent=agents.build(data_dir)).status is RunStatus.COMPLETED
    assert _counts(data_dir)["send_email"] == 2  # the operator accepted a possible duplicate
    runtime.close()


def test_resume_via_cli_subprocess_completes(created: tuple[Path, str]) -> None:
    data_dir, run_id = created
    marker = data_dir / "stall.marker"
    proc = _spawn(
        data_dir, run_id, "--stall-at", "tool_finished:slow_read", "--stall-marker", str(marker)
    )
    kill_when_marked(proc, marker)
    second = _spawn(data_dir, run_id)
    assert second.wait(timeout=60) == 0
    assert _counts(data_dir) == {"read_a": 1, "slow_read": 1, "send_email": 1}
