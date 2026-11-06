"""Command line interface.

Exit codes for ``start`` and ``resume``: 0 completed, 1 failed, 3 blocked (needs ``resolve``),
4 cancelled, 5 still in progress (cancel requested on a live run). 2 is a usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from agentflow.agent import Agent, load_agent
from agentflow.faults import stall_listener
from agentflow.runtime import Listener, RunOutcome, Runtime
from agentflow.state import RunStatus
from agentflow.store import RunNotFound
from agentflow.trace import format_trace, read_trace

EXIT_CODES = {
    RunStatus.COMPLETED: 0,
    RunStatus.FAILED: 1,
    RunStatus.BLOCKED: 3,
    RunStatus.CANCELLED: 4,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentflow", description="Checkpointed runtime for tool-using LLM agents."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("AGENTFLOW_DATA_DIR", ".agentflow")),
        help="directory holding agentflow.db and traces/ (default: .agentflow or "
        "$AGENTFLOW_DATA_DIR)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="crash-and-resume incident-triage demo")
    demo.add_argument("--workdir", type=Path, help="new or empty directory (default: a temp dir)")

    def add_stall(p: argparse.ArgumentParser) -> None:
        p.add_argument("--stall-at", help=argparse.SUPPRESS)  # fault injection for tests/demo
        p.add_argument("--stall-marker", type=Path, help=argparse.SUPPRESS)

    start = sub.add_parser("start", help="create a run and drive it")
    start.add_argument("--agent", required=True, help="agent factory, e.g. pkg.module:build")
    start.add_argument("--goal", required=True, help="task given to the agent")
    add_stall(start)

    resume = sub.add_parser("resume", help="drive a pending, interrupted or resolved run")
    resume.add_argument("run_id")
    add_stall(resume)

    sub.add_parser("runs", help="list runs")
    show = sub.add_parser("show", help="run status, counters and tool calls")
    show.add_argument("run_id")
    trace = sub.add_parser("trace", help="print a run's trace")
    trace.add_argument("run_id")
    trace.add_argument("--json", action="store_true", help="raw JSONL events")
    cancel = sub.add_parser("cancel", help="cancel a run")
    cancel.add_argument("run_id")

    resolve = sub.add_parser("resolve", help="decide an ambiguous tool call on a blocked run")
    resolve.add_argument("run_id")
    resolve.add_argument("call_id")
    choice = resolve.add_mutually_exclusive_group(required=True)
    choice.add_argument("--retry", action="store_true", help="execute the call again")
    choice.add_argument("--fail", metavar="MESSAGE", help="record it as failed")
    choice.add_argument("--result", metavar="JSON", help="record this JSON value as its result")
    return parser


def _listeners(args: argparse.Namespace) -> list[Listener]:
    if args.stall_at:
        marker = args.stall_marker or args.data_dir / "stalled.marker"
        return [stall_listener(args.stall_at, marker)]
    return []


def _agent_for(runtime: Runtime, run_id: str) -> Agent | None:
    """The run's agent, so operator events use its redactor; None if it cannot be built here."""
    try:
        return load_agent(runtime.store.get_run(run_id).agent_ref, runtime.data_dir)
    except RunNotFound:
        raise
    except Exception:
        return None  # operator events carry no payloads; the default redactor is enough


def _report(outcome: RunOutcome) -> int:
    print(f"run {outcome.run_id}: {outcome.status}")
    if outcome.error:
        print(f"error: {outcome.error}")
    if outcome.result is not None:
        print(json.dumps(outcome.result, indent=2))
    return EXIT_CODES.get(outcome.status, 5)


def _show(runtime: Runtime, run_id: str) -> int:
    run = runtime.store.get_run(run_id)
    print(f"run       {run.id}")
    print(f"agent     {run.agent_ref}")
    print(f"status    {run.status}")
    print(
        f"counters  model_calls={run.model_calls}/{run.max_steps}"
        f" tool_calls={run.tool_call_count}/{run.max_tool_calls}"
        f" repairs={run.repairs}/{run.max_repairs}"
    )
    if run.cancel_requested:
        print("cancel    requested")
    if run.error:
        print(f"error     {run.error}")
    calls = runtime.store.tool_calls(run_id)
    if calls:
        print("tool calls:")
        for call in calls:
            line = f"  {call.call_id:<6} {call.tool:<20} {call.status:<10} attempts={call.attempts}"
            if call.error:
                line += f"  {call.error}"
            print(line)
    if run.result is not None:
        print("result:")
        print(json.dumps(run.result, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "demo":
        from agentflow.demo import run_demo

        return run_demo(args.workdir)

    listeners = _listeners(args) if args.command in {"start", "resume"} else []
    try:
        runtime = Runtime(args.data_dir, listeners=listeners)
    except Exception as exc:
        if os.environ.get("AGENTFLOW_DEBUG"):
            raise
        print(f"error: cannot open {args.data_dir}: {exc}", file=sys.stderr)
        return 2
    try:
        if args.command == "start":
            run_id = runtime.create(args.agent, args.goal)
            print(f"created run {run_id}", flush=True)
            return _report(runtime.run(run_id))
        if args.command == "resume":
            return _report(runtime.run(args.run_id))
        if args.command == "runs":
            for run in runtime.store.list_runs():
                print(f"{run.id}  {run.status:<12} {run.agent_ref}  {run.goal[:60]}")
            return 0
        if args.command == "show":
            return _show(runtime, args.run_id)
        if args.command == "trace":
            path = runtime.trace_dir / f"{args.run_id}.jsonl"
            if not path.exists():
                print(f"no trace for run {args.run_id} in {runtime.trace_dir}", file=sys.stderr)
                return 2
            if args.json:
                print(path.read_text(encoding="utf-8"), end="")
            else:
                print(format_trace(read_trace(path)))
            return 0
        if args.command == "cancel":
            status = runtime.cancel(args.run_id, agent=_agent_for(runtime, args.run_id))
            if status is RunStatus.CANCELLED:
                print(f"run {args.run_id}: cancelled")
            elif status in (RunStatus.COMPLETED, RunStatus.FAILED):
                print(f"run {args.run_id} already {status}")
            else:
                print(
                    f"run {args.run_id}: cancel requested; a live runner stops at its next step."
                    f" If its process is gone, `agentflow resume {args.run_id}` applies it."
                )
            return 0
        if args.command == "resolve":
            agent = _agent_for(runtime, args.run_id)
            if args.retry:
                runtime.resolve(args.run_id, args.call_id, "retry", agent=agent)
            elif args.fail is not None:
                runtime.resolve(args.run_id, args.call_id, "fail", args.fail, agent=agent)
            else:
                runtime.resolve(
                    args.run_id, args.call_id, "succeed", json.loads(args.result), agent=agent
                )
            print(f"resolved {args.call_id}; continue with: agentflow resume {args.run_id}")
            return 0
    except RunNotFound as exc:
        print(f"no such run: {exc.args[0]}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Report, do not dump a traceback; AGENTFLOW_DEBUG=1 re-raises for debugging.
        if os.environ.get("AGENTFLOW_DEBUG"):
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        runtime.close()
    raise AssertionError(f"unhandled command {args.command}")  # pragma: no cover
