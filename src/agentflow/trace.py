"""JSONL trace per run, with a redaction hook applied before anything reaches disk.

The trace is an observability log, not the source of truth: the SQLite store is. A trace event is
written after the matching database commit, so a crash can lose at most the trace line for the
last committed step, never a step itself.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

Redactor = Callable[[dict[str, Any]], dict[str, Any]]

SENSITIVE_KEY = re.compile(r"(api[_-]?key|token|secret|password|passwd|authorization|cookie)", re.I)
REDACTED = "[REDACTED]"


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: REDACTED if SENSITIVE_KEY.search(str(k)) else _redact_value(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    return value


def redact_sensitive_keys(event: dict[str, Any]) -> dict[str, Any]:
    """Default redactor: replace the value of any key that looks like a credential."""
    return cast(dict[str, Any], _redact_value(event))


def chain_redactors(*redactors: Redactor) -> Redactor:
    def redact(event: dict[str, Any]) -> dict[str, Any]:
        for redactor in redactors:
            event = redactor(event)
        return event

    return redact


class TraceWriter:
    def __init__(self, trace_dir: Path, redactor: Redactor = redact_sensitive_keys) -> None:
        self.trace_dir = trace_dir
        self.redactor = redactor
        trace_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        return self.trace_dir / f"{run_id}.jsonl"

    def write(self, run_id: str, event: str, **fields: Any) -> dict[str, Any]:
        record = self.redactor(
            {"ts": round(time.time(), 6), "run_id": run_id, "event": event, **fields}
        )
        with self.path_for(run_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record


def read_trace(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A process killed mid-write can leave a torn final line; skip it.
            continue
    return events


def _short(value: Any, limit: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_trace(events: Iterable[dict[str, Any]]) -> str:
    """Human-readable timeline used by ``agentflow trace <run_id>``."""
    rows = []
    start: float | None = None
    for event in events:
        ts = float(event.get("ts", 0.0))
        start = ts if start is None else start
        name = str(event.get("event"))
        detail = ""
        if name == "state":
            detail = f"{event.get('from')} -> {event.get('to')}"
            if event.get("reason"):
                detail += f"  ({event['reason']})"
        elif name == "model_call":
            detail = f"turn {event.get('turn')} {event.get('duration_ms')}ms -> " + _short(
                event.get("response")
            )
        elif name == "repair":
            detail = f"{event.get('kind')} repair #{event.get('repairs')}"
            if event.get("errors"):
                detail += ": " + _short("; ".join(map(str, event["errors"])), 110)
        elif name.startswith("tool_"):
            detail = f"{event.get('call_id')} {event.get('tool')}"
            if "attempt" in event:
                detail += f" attempt {event['attempt']}"
            for key in ("arguments", "result", "error", "reason"):
                if key in event:
                    detail += f" {key}=" + _short(event[key], 70)
            if "duration_ms" in event:
                detail += f" {event['duration_ms']}ms"
        else:
            extras = {k: v for k, v in event.items() if k not in {"ts", "run_id", "event"}}
            detail = _short(extras) if extras else ""
        rows.append(f"+{ts - start:8.3f}s  {name:<18} {detail}".rstrip())
    return "\n".join(rows)
