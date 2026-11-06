"""SQLite checkpoint store.

Every step of a run is a transaction: a model turn is committed together with the tool calls it
requested, and a tool result is committed together with the tool message the model will see.
Resume therefore never has to guess which half of a step happened.

Writes on behalf of a runner are fenced by ``owner_token``. Claiming a run (``claim``) writes a
fresh token; every later write checks it inside the same transaction. A process that was presumed
dead and replaced by a resumed runner cannot commit afterwards: its writes raise ``LostOwnership``.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentflow.messages import Message, ToolCallRequest
from agentflow.state import RunStatus, ToolCallStatus, check_transition

SCHEMA_VERSION = 2

_RUN_STATUSES = ", ".join(f"'{s.value}'" for s in RunStatus)
_CALL_STATUSES = ", ".join(f"'{s.value}'" for s in ToolCallStatus)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    agent_ref        TEXT NOT NULL,
    goal             TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ({_RUN_STATUSES})),
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    deadline_at      REAL,
    max_steps        INTEGER NOT NULL,
    max_tool_calls   INTEGER NOT NULL,
    max_repairs      INTEGER NOT NULL,
    model_calls      INTEGER NOT NULL DEFAULT 0,
    tool_call_count  INTEGER NOT NULL DEFAULT 0,
    repairs          INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    owner_token      TEXT,
    result           TEXT,
    error            TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    run_id       TEXT NOT NULL REFERENCES runs(id),
    seq          INTEGER NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT,
    tool_calls   TEXT,
    tool_call_id TEXT,
    name         TEXT,
    created_at   REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS tool_calls (
    run_id      TEXT NOT NULL REFERENCES runs(id),
    call_id     TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    idx         INTEGER NOT NULL,
    provider_id TEXT NOT NULL,
    tool        TEXT NOT NULL,
    arguments   TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ({_CALL_STATUSES})),
    attempts    INTEGER NOT NULL DEFAULT 0,
    result      TEXT,
    error       TEXT,
    started_at  REAL,
    finished_at REAL,
    idempotent   INTEGER,  -- tool policy recorded when the model requested the call
    max_attempts INTEGER,
    PRIMARY KEY (run_id, call_id)
);
CREATE TABLE IF NOT EXISTS transitions (
    run_id      TEXT NOT NULL REFERENCES runs(id),
    seq         INTEGER NOT NULL,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    reason      TEXT,
    at          REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


class LostOwnership(RuntimeError):
    """Another runner claimed this run; this process must stop writing to it."""


class RunNotFound(KeyError):
    pass


@dataclass(frozen=True)
class RunRecord:
    id: str
    agent_ref: str
    goal: str
    status: RunStatus
    created_at: float
    updated_at: float
    deadline_at: float | None
    max_steps: int
    max_tool_calls: int
    max_repairs: int
    model_calls: int
    tool_call_count: int
    repairs: int
    cancel_requested: bool
    owner_token: str | None
    result: Any
    error: str | None


@dataclass(frozen=True)
class ToolCallRecord:
    run_id: str
    call_id: str
    turn: int
    idx: int
    provider_id: str
    tool: str
    arguments: Any
    status: ToolCallStatus
    attempts: int
    result: Any
    error: str | None
    started_at: float | None
    finished_at: float | None
    idempotent: bool | None  # None: rejected call, never had a policy
    max_attempts: int | None


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


class Tx:
    """Write operations available inside one fenced transaction."""

    def __init__(self, conn: sqlite3.Connection, run_id: str, now: float) -> None:
        self.conn = conn
        self.run_id = run_id
        self.now = now

    def status(self) -> RunStatus:
        row = self.conn.execute("SELECT status FROM runs WHERE id = ?", (self.run_id,)).fetchone()
        return RunStatus(row[0])

    def set_status(self, target: RunStatus, reason: str | None = None) -> RunStatus:
        current = self.status()
        check_transition(current, target)
        self.conn.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
            (target.value, self.now, self.run_id),
        )
        self._log_transition(current, target, reason)
        return current

    def _log_transition(
        self, current: RunStatus | None, target: RunStatus, reason: str | None
    ) -> None:
        (seq,) = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM transitions WHERE run_id = ?", (self.run_id,)
        ).fetchone()
        self.conn.execute(
            "INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?)",
            (self.run_id, seq, current.value if current else None, target.value, reason, self.now),
        )

    def add_message(self, message: Message) -> int:
        (seq,) = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE run_id = ?", (self.run_id,)
        ).fetchone()
        calls = (
            json.dumps([c.to_json() for c in message.tool_calls]) if message.tool_calls else None
        )
        self.conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.run_id,
                seq,
                message.role,
                message.content,
                calls,
                message.tool_call_id,
                message.name,
                self.now,
            ),
        )
        return int(seq)

    def bump(self, column: str, amount: int = 1) -> int:
        if column not in {"model_calls", "tool_call_count", "repairs"}:
            raise ValueError(column)
        self.conn.execute(
            f"UPDATE runs SET {column} = {column} + ? WHERE id = ?",  # noqa: S608 - allow-listed
            (amount, self.run_id),
        )
        (value,) = self.conn.execute(
            f"SELECT {column} FROM runs WHERE id = ?",  # noqa: S608 - allow-listed column
            (self.run_id,),
        ).fetchone()
        return int(value)

    def set_outcome(self, result: Any = None, error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE runs SET result = ?, error = ? WHERE id = ?",
            (None if result is None else json.dumps(result), error, self.run_id),
        )

    def add_tool_call(
        self,
        call_id: str,
        turn: int,
        idx: int,
        request: ToolCallRequest,
        status: ToolCallStatus,
        error: str | None = None,
        *,
        idempotent: bool | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO tool_calls (run_id, call_id, turn, idx, provider_id, tool, arguments,"
            " status, error, finished_at, idempotent, max_attempts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.run_id,
                call_id,
                turn,
                idx,
                request.id,
                request.name,
                json.dumps(request.arguments),
                status.value,
                error,
                self.now if status is ToolCallStatus.REJECTED else None,
                None if idempotent is None else int(idempotent),
                max_attempts,
            ),
        )

    def update_tool_call(
        self,
        call_id: str,
        status: ToolCallStatus,
        *,
        attempts: int | None = None,
        result: Any = None,
        error: str | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        sets = ["status = ?", "result = ?", "error = ?"]
        params: list[Any] = [status.value, None if result is None else json.dumps(result), error]
        if attempts is not None:
            sets.append("attempts = ?")
            params.append(attempts)
        if started:
            sets.append("started_at = ?")
            params.append(self.now)
        if finished:
            sets.append("finished_at = ?")
            params.append(self.now)
        params += [self.run_id, call_id]
        self.conn.execute(
            f"UPDATE tool_calls SET {', '.join(sets)} WHERE run_id = ? AND call_id = ?",  # noqa: S608 - fixed fragments
            params,
        )


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: no implicit transactions; every write uses BEGIN IMMEDIATE.
        self.conn = sqlite3.connect(
            path, isolation_level=None, timeout=30.0, check_same_thread=False
        )
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")
        self.conn.execute("PRAGMA foreign_keys = ON")
        (version,) = self.conn.execute("PRAGMA user_version").fetchone()
        if version not in (0, 1, SCHEMA_VERSION):
            raise RuntimeError(f"unsupported store schema version {version}")
        self.conn.executescript(SCHEMA)
        if version == 1:  # v2 records each call's idempotency and attempt policy
            self.conn.execute("ALTER TABLE tool_calls ADD COLUMN idempotent INTEGER")
            self.conn.execute("ALTER TABLE tool_calls ADD COLUMN max_attempts INTEGER")
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()

    # -- transactions -------------------------------------------------------------------------

    @contextmanager
    def _begin(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    @contextmanager
    def tx(self, run_id: str, token: str | None) -> Iterator[Tx]:
        """Fenced transaction: aborts with ``LostOwnership`` if ``token`` no longer owns the run.

        ``token=None`` is for operator actions (cancel, resolve) that are allowed regardless of
        which runner owns the run.
        """
        with self._begin() as conn:
            row = conn.execute("SELECT owner_token FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise RunNotFound(run_id)
            if token is not None and row[0] != token:
                raise LostOwnership(f"run {run_id} was claimed by another runner")
            now = time.time()
            conn.execute("UPDATE runs SET updated_at = ? WHERE id = ?", (now, run_id))
            yield Tx(conn, run_id, now)

    # -- run lifecycle ------------------------------------------------------------------------

    def create_run(
        self,
        *,
        agent_ref: str,
        goal: str,
        messages: list[Message],
        max_steps: int,
        max_tool_calls: int,
        max_repairs: int,
        deadline_at: float | None,
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or uuid.uuid4().hex[:12]
        now = time.time()
        with self._begin() as conn:
            conn.execute(
                "INSERT INTO runs (id, agent_ref, goal, status, created_at, updated_at,"
                " deadline_at, max_steps, max_tool_calls, max_repairs)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    agent_ref,
                    goal,
                    RunStatus.PENDING.value,
                    now,
                    now,
                    deadline_at,
                    max_steps,
                    max_tool_calls,
                    max_repairs,
                ),
            )
            tx = Tx(conn, run_id, now)
            tx._log_transition(None, RunStatus.PENDING, "created")
            for message in messages:
                tx.add_message(message)
        return run_id

    def claim(self, run_id: str) -> str:
        """Become the run's only writer. Any previous owner is fenced out."""
        token = uuid.uuid4().hex
        with self._begin() as conn:
            cursor = conn.execute(
                "UPDATE runs SET owner_token = ?, updated_at = ? WHERE id = ?",
                (token, time.time(), run_id),
            )
            if cursor.rowcount != 1:
                raise RunNotFound(run_id)
        return token

    def request_cancel(self, run_id: str) -> None:
        with self.tx(run_id, None) as tx:
            tx.conn.execute("UPDATE runs SET cancel_requested = 1 WHERE id = ?", (run_id,))

    # -- reads --------------------------------------------------------------------------------

    def get_run(self, run_id: str) -> RunRecord:
        cursor = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        if row is None:
            raise RunNotFound(run_id)
        data = dict(zip([d[0] for d in cursor.description], row, strict=True))
        data["status"] = RunStatus(data["status"])
        data["cancel_requested"] = bool(data["cancel_requested"])
        data["result"] = _loads(data["result"])
        return RunRecord(**data)

    def list_runs(self) -> list[RunRecord]:
        ids = [r[0] for r in self.conn.execute("SELECT id FROM runs ORDER BY created_at")]
        return [self.get_run(i) for i in ids]

    def messages(self, run_id: str) -> list[Message]:
        rows = self.conn.execute(
            "SELECT role, content, tool_calls, tool_call_id, name FROM messages"
            " WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
        return [
            Message(
                role=role,
                content=content,
                tool_calls=tuple(ToolCallRequest.from_json(c) for c in _loads(calls) or []),
                tool_call_id=tool_call_id,
                name=name,
            )
            for role, content, calls, tool_call_id, name in rows
        ]

    def tool_calls(self, run_id: str) -> list[ToolCallRecord]:
        cursor = self.conn.execute(
            "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY turn, idx", (run_id,)
        )
        columns = [d[0] for d in cursor.description]
        records = []
        for row in cursor.fetchall():
            data = dict(zip(columns, row, strict=True))
            data["arguments"] = _loads(data["arguments"])
            data["status"] = ToolCallStatus(data["status"])
            data["result"] = _loads(data["result"])
            if data["idempotent"] is not None:
                data["idempotent"] = bool(data["idempotent"])
            records.append(ToolCallRecord(**data))
        return records

    def get_tool_call(self, run_id: str, call_id: str) -> ToolCallRecord:
        for call in self.tool_calls(run_id):
            if call.call_id == call_id:
                return call
        raise KeyError(f"run {run_id} has no tool call {call_id}")

    def transitions(self, run_id: str) -> list[tuple[str | None, str, str | None, float]]:
        return [
            (r[0], r[1], r[2], r[3])
            for r in self.conn.execute(
                "SELECT from_status, to_status, reason, at FROM transitions"
                " WHERE run_id = ? ORDER BY seq",
                (run_id,),
            )
        ]
