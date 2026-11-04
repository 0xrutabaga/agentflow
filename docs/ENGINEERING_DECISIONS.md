# Engineering decisions

Each entry: the decision, why, what was rejected, and what it costs.

## 1. The store is the source of truth; the loop has no recovery mode

`_Driver.drive` (`src/agentflow/runtime.py`) reads the run's status from SQLite at the top of
every iteration and dispatches on it: `running` means "call the model", `waiting_tool` means
"execute the non-terminal tool calls". A fresh run and a resumed run take the same path.

Rejected: keeping progress in memory and writing a snapshot periodically, then writing a separate
recovery routine. Two code paths for the same state drift apart, and the recovery path is the one
that is rarely exercised.

Cost: one extra `SELECT` per step. Irrelevant next to a model call.

## 2. Step boundaries are transactions

- `_Driver._record_tool_turn`: the assistant message, the model-call counter, the tool-call
  counter, every requested call (`requested` or `rejected`), the rejection messages and the
  `running -> waiting_tool` transition commit together.
- `_Driver._execute`: an attempt is committed as `started` (with its attempt number) *before* the
  tool runs; the result and the tool message the model will see commit together afterwards.
- A failed attempt that will be retried is committed back to `requested` before the backoff
  sleep, so a crash during the sleep resumes as a clean retry rather than an ambiguous call
  (`test_crash_during_retry_backoff_resumes_as_clean_retry`).

All writes use `BEGIN IMMEDIATE` (`Store._begin`) so the writer lock is taken up front, WAL
journal mode, and `synchronous=FULL`, so a committed step survives power loss, not only a process
kill.

## 3. Started-but-unrecorded is ambiguous, and idempotency decides what happens

The runtime cannot know whether a tool that was started had its effect. `_Driver._handle_call`
treats a call found in `started` on resume like a timeout: idempotent tools are executed again
(bounded by `RetryPolicy.max_attempts`, which counts the interrupted attempt), non-idempotent
tools are marked `ambiguous` and the run moves to `blocked` (`_Driver._block`).

`TransientToolError` exists so that a tool author can say "this attempt definitely had no effect"
(for example a 429 returned before the request was processed). That is the only case in which a
non-idempotent tool is retried automatically. `ToolError` is a definite, permanent failure and is
never retried. Any other exception is ambiguous.

Rejected: retry everything (duplicates emails and tickets); retry nothing (a flaky read fails the
whole run); fail the run on ambiguity (loses all progress for a question a human can answer in a
minute). `Runtime.resolve` gives the operator three answers: `retry`, `fail`, `succeed` with a
result.

## 4. Fencing tokens instead of locks or leases

`Store.claim` writes a fresh `owner_token`; `Store.tx(run_id, token)` checks it inside the same
transaction as the write and raises `LostOwnership` if another runner has claimed the run since.
This guarantees a stale process cannot commit results after a newer runner took over
(`test_stale_runner_is_fenced_out`).

Rejected: a lease with heartbeat. It would let a second runner detect a live first one, but it is
the core of a job queue, and the portfolio's queue project (queueforge) already covers leases and
reapers. Here durability is per-run step checkpoints. Cost: two runners started at once on the
same run can both begin an in-flight tool before one is fenced out; documented as a limitation.

## 5. The wall-clock budget is an absolute deadline stored with the run

`Runtime.create` stores `deadline_at = now + max_wall_s`. `_Driver._stop_reason` checks it before
each model call and each tool attempt, and `_Driver._execute` caps each tool timeout at the
remaining time (`test_deadline_caps_the_tool_timeout`). A run resumed after its deadline fails
immediately.

Rejected: a per-process timer. It would reset on every resume, so a run that crashes repeatedly
would never run out of time.

## 6. Tool timeouts use a daemon thread per attempt

`invoke_with_timeout` (`src/agentflow/tools.py`) runs the tool in a daemon thread and joins with a
timeout. Python cannot kill a thread, so the attempt may keep running and may still produce its
side effect. This is exactly why a timeout is classified as ambiguous.

Rejected: `signal.alarm` (POSIX only, main thread only); a process pool (tools would need to be
picklable and process start-up dominates short reads); `asyncio` (would force every tool to be
async). Future work lists process isolation for tools that must be killable.

## 7. Repair is bounded and generic

`_Driver._record_final_turn` parses the answer with `parse_json_output` (one Markdown code fence
is tolerated) and validates it with the agent's schema via `validation_errors`, which produces up
to five path-qualified messages such as `$.severity: 'high' is not one of [...]`. Those messages go
back to the model as a user message. Invalid tool calls (unknown tool, arguments failing schema,
unparseable JSON passed through by the HTTP provider) are rejected without execution and the error
is returned as the tool result. Both kinds share one counter, `max_repairs`, per run; exceeding it
fails the run with the last errors in `error`.

The validation is deliberately generic JSON Schema. Business rules for a specific document type
belong to the agent, not to the runtime.

## 8. The fake model chooses its response from the conversation

`ScriptedModel.complete` counts assistant messages in the history it is given. A resumed process
replays the persisted conversation and therefore gets the same next response the crashed process
would have. An internal counter would desynchronise after a crash. Steps can also be callables of
the history, which is how the triage script's final answer quotes the ticket id that
`create_ticket` actually returned.

## 9. Agents are rebuilt from code on resume

A run stores `agent_ref` (`module:factory`); `load_agent` imports it and calls it with the data
directory. Tools are functions and the provider holds a client, so neither can be stored in the
database. The cost is that the database names code to import; the data directory must be trusted
(listed in the README limitations).

## 10. The trace is written after the commit and is not authoritative

`TraceWriter.write` appends one JSON line per event, after the database commit it describes, with
the agent's redactor applied first (`redact_sensitive_keys` masks values of credential-looking
keys at any depth; the triage example chains `mask_phone_numbers`). A crash can lose the trace
line for the last step but never the step. `read_trace` skips a torn final line.

## 11. Crash tests kill a real process

`faults.stall_listener` blocks the child process right after a chosen trace event and writes a
marker with its pid; `faults.kill_when_marked` waits for the marker and kills that pid
(`os.kill` with SIGKILL on Linux; on Windows `os.kill` is `TerminateProcess`). On Windows a
virtualenv's `python.exe` is a launcher whose child is the real interpreter, so the pid in the
marker differs from `Popen.pid`; `_wait_gone` then waits on a process handle until the
interpreter has exited, because opening the SQLite file while the dying process still holds it
produced `disk I/O error` during development.

Rejected: `os._exit` inside the child. It is a crash the process chooses, at a point in its own
code; a kill from outside is closer to an OOM kill or a node going away.

## 12. State added beyond the original plan: `blocked`

The plan listed `pending -> running -> waiting_tool -> completed | failed | cancelled`. A
non-idempotent ambiguous call needs a state that is neither terminal nor runnable, so `blocked`
was added, reachable only from `waiting_tool` and left only by an operator decision
(`blocked -> waiting_tool`) or a cancel.
