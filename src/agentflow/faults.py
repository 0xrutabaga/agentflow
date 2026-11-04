"""Fault injection used by the crash tests and the demo.

``stall_listener`` freezes the process right after a chosen trace event (which is always emitted
after its database commit) and writes a marker file. The parent then kills the process with
``Popen.kill`` (SIGKILL on Linux, TerminateProcess on Windows), so "crash between steps" means a
real, abrupt process death at a known point, on both platforms.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from agentflow.runtime import Listener


def parse_stall_spec(spec: str) -> tuple[str, str | None]:
    """``"tool_finished:search_logs"`` -> ``("tool_finished", "search_logs")``."""
    event, _, tool = spec.partition(":")
    return event, tool or None


def stall_listener(spec: str, marker: Path) -> Listener:
    event_name, tool = parse_stall_spec(spec)

    def listener(record: dict[str, Any]) -> None:
        if record.get("event") != event_name:
            return
        if tool is not None and record.get("tool") != tool:
            return
        tmp = marker.with_name(marker.name + ".tmp")
        tmp.write_text(str(os.getpid()), encoding="utf-8")
        os.replace(tmp, marker)  # atomic: the parent never reads a half-written pid
        while True:  # wait to be killed
            time.sleep(3600)

    return listener


def _hard_kill(pid: int) -> None:
    # SIGKILL does not exist on Windows; there os.kill with any other signal is TerminateProcess.
    sig = getattr(signal, "SIGKILL", signal.SIGTERM)
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(pid, sig)


def _wait_gone(pid: int, timeout_s: float) -> None:
    """Wait until ``pid`` has fully exited and released its file handles."""
    if sys.platform == "win32":
        import ctypes

        synchronize = 0x00100000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if handle:  # 0 means the process is already gone
            try:
                kernel32.WaitForSingleObject(handle, int(timeout_s * 1000))
            finally:
                kernel32.CloseHandle(handle)
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)


def kill_when_marked(proc: subprocess.Popen[Any], marker: Path, timeout_s: float = 60.0) -> int:
    """Wait for ``marker`` to appear, then hard-kill the stalled interpreter.

    The marker holds the pid of the interpreter that stalled. It can differ from ``proc.pid``:
    on Windows a virtualenv's ``python.exe`` is a launcher that starts the real interpreter as a
    child. Both are killed. Returns the stalled interpreter's pid.
    """
    deadline = time.monotonic() + timeout_s
    while not marker.exists():
        if proc.poll() is not None:
            raise RuntimeError(f"process exited with {proc.returncode} before reaching the stall")
        if time.monotonic() > deadline:
            proc.kill()
            raise TimeoutError("process did not reach the stall point in time")
        time.sleep(0.05)
    stalled_pid = int(marker.read_text(encoding="utf-8"))
    _hard_kill(stalled_pid)
    proc.kill()
    proc.wait(timeout=30)
    if stalled_pid != proc.pid:
        _wait_gone(stalled_pid, 30.0)
    return stalled_pid
