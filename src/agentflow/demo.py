"""``agentflow demo``: crash an incident-triage run mid-flight, resume it, print the trace.

1. Create a run of ``agentflow.examples.triage``.
2. Start it in a child process that stalls right after the ``search_logs`` result is committed
   (after two failed, retried attempts), and kill that process.
3. Show what the store recorded at the moment of the kill.
4. Resume the run in this process. Completed calls are not executed again; the remaining calls,
   one invalid final answer and its repair run to completion.
5. Print per-process tool executions, the final output and the trace.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from agentflow.examples import triage
from agentflow.faults import kill_when_marked
from agentflow.runtime import Runtime
from agentflow.state import RunStatus
from agentflow.trace import format_trace, read_trace

STALL_AT = "tool_finished:search_logs"


def run_demo(workdir: Path | None = None, out: Callable[[str], None] = print) -> int:
    if workdir is None:
        data_dir = Path(tempfile.mkdtemp(prefix="agentflow-demo-"))
    else:
        data_dir = workdir
        if data_dir.exists() and any(data_dir.iterdir()):
            out(f"error: {data_dir} is not empty; pass a new or empty directory")
            return 2
    data_dir.mkdir(parents=True, exist_ok=True)
    out(f"data dir: {data_dir}")

    runtime = Runtime(data_dir)
    try:
        run_id = runtime.create(triage.AGENT_REF, triage.GOAL)
        out(f"created run {run_id} ({triage.AGENT_REF})")

        marker = data_dir / "stalled.marker"
        out(f"\n[1] starting the run in a child process; it will be killed after {STALL_AT}")
        child = subprocess.Popen(  # noqa: S603 - fixed argv: this interpreter running this package
            [
                sys.executable,
                "-m",
                "agentflow",
                "--data-dir",
                str(data_dir),
                "resume",
                run_id,
                "--stall-at",
                STALL_AT,
                "--stall-marker",
                str(marker),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pid = kill_when_marked(child, marker)
        out(f"    killed interpreter pid {pid} with a hard kill; no cleanup code ran")

        record = runtime.store.get_run(run_id)
        out(f"\n[2] state recorded at the moment of the kill: status={record.status}")
        for call in runtime.store.tool_calls(run_id):
            out(f"    {call.call_id} {call.tool:<20} {call.status:<10} attempts={call.attempts}")

        out("\n[3] resuming in a new process (this one)")
        outcome = runtime.run(run_id)
        out(f"    final status: {outcome.status}")
        if outcome.error:
            out(f"    error: {outcome.error}")

        out("\n[4] tool executions by process (from executions.jsonl)")
        executions = triage.read_executions(data_dir)
        labels = {pid: "killed child"}
        for row in executions:
            labels.setdefault(row["pid"], "resumed parent")
        counts = Counter((labels[r["pid"]], r["tool"], r["outcome"]) for r in executions)
        for (label, tool, result), n in sorted(counts.items()):
            out(f"    {label:<15} {tool:<20} {result:<17} x{n}")
        tickets = sorted(p.name for p in (data_dir / "tickets").glob("TCK-*.json"))
        out(f"    tickets on disk: {', '.join(tickets) or 'none'}")

        out("\n[5] validated output")
        out(json.dumps(outcome.result, indent=2))

        out(f"\n[6] trace ({runtime.trace_dir.name}/{run_id}.jsonl)")
        out(format_trace(read_trace(runtime.trace_dir / f"{run_id}.jsonl")))
        out(f"\nre-print with: agentflow --data-dir {data_dir} trace {run_id}")
        return 0 if outcome.status is RunStatus.COMPLETED else 1
    finally:
        runtime.close()
