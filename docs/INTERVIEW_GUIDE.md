# Interview guide

How to present agentflow, the questions a senior reviewer is likely to ask, and answers that
point at the code.

## Two-minute summary

agentflow is a run loop for tool-using LLM agents in which every step is a SQLite transaction.
After a crash, `agentflow resume <run_id>` continues from the last committed step. Recorded tool
results are reused, never re-executed. An attempt that started but has no recorded result is
ambiguous: idempotent tools are retried, non-idempotent tools put the run in `blocked` for an
operator. Tool arguments and final answers are validated against JSON Schema with bounded repair,
and every run has step, tool-call and wall-clock budgets. The demo kills a real process mid-run
and resumes it.

## Live demo (about five minutes)

```sh
uv sync --locked
uv run agentflow demo --workdir .agentflow-demo
```

Points to show in the output:

1. Section [2]: at the kill, `search_logs` is `succeeded` with `attempts=3` (two
   `ConnectionError`s, retried) and `get_recent_deploys` is still `requested`.
2. Section [4]: `get_alert` and `search_logs` ran only in the killed child; the resumed parent ran
   only `get_recent_deploys` and `create_ticket`. One ticket file exists.
3. Section [6]: the second `run_claimed` line shows the resume starting from `waiting_tool` with
   counters `model_calls=2, tool_calls=3`; the `repair` line shows the schema errors that were
   sent back; the final state change is `running -> completed`.

Then show the blocked path, which the demo does not cover:

```sh
uv run pytest -q tests/test_crash_resume.py -k non_idempotent
```

and open `tests/test_crash_resume.py::test_kill_inside_non_idempotent_tool_blocks_for_a_human`:
the email tool logs its execution and is then killed; on resume the run is `blocked`, the tool is
not re-run, and `Runtime.resolve(..., "succeed", ...)` lets it complete with one execution.

Finally `uv run agentflow --data-dir .agentflow-demo show <run_id>` to show the stored counters
and tool-call table.

## Five hard questions

### 1. "You say completed tool calls are not re-executed. What exactly is the guarantee, and where can it break?"

The guarantee is about *recorded results*. In `_Driver._execute` (`src/agentflow/runtime.py`) the
result and the tool message are committed in one transaction; `_Driver._run_pending_tools` skips
every call whose status is terminal (`TERMINAL_CALL_STATUSES` in `state.py`). So once the result
commit returns, no future runner executes that call.

It breaks, by design, in the window between the tool's side effect and that commit. The attempt
was committed as `started` beforehand, so on resume `_Driver._handle_call` sees `started` and
knows it is ambiguous. It does not claim exactly-once; it makes the window explicit and applies
the idempotency flag. For non-idempotent tools the honest answer is a human decision
(`Runtime.resolve`). The proper fix for a remote API is an idempotency key derived from
`run_id` and `call_id`, which is in Future work.

Tradeoff: the `started` commit costs one extra fsync per attempt. Without it, a crash during a
tool would look the same as "never started", and a non-idempotent tool would be run twice
silently.

### 2. "Why not just retry failed tools a few times?"

Because a failure that is not definite may already have had its effect. `tools.py` classifies
outcomes: `ToolError` is definite and permanent (no retry), `TransientToolError` is definite with
no side effect (retry even for non-idempotent tools), and every timeout or other exception is
ambiguous (retry only if `Tool.idempotent`). Timeouts are ambiguous for a concrete reason:
`invoke_with_timeout` cannot kill the worker thread, so a timed-out attempt can still finish.

Two cases that look like tool failures are kept out of the ambiguous bucket because the tool
provably never ran: arguments the function cannot bind (`Tool.binding_errors`, checked before the
attempt is marked `started`) and a non-idempotent call with less time left than its timeout (not
started; the run fails with `budget_exhausted`). Both were found in review and have tests in
`tests/test_regressions.py`.

Evidence: `test_non_idempotent_tool_is_not_retried_after_ambiguous_failure` (one execution, run
`blocked`), `test_non_idempotent_transient_error_is_retried` (two executions, completed),
`test_flaky_idempotent_tool_is_retried_with_backoff` (backoff delays `[0.2, 0.4]` recorded by an
injected `sleep`).

### 3. "What stops two processes from driving the same run?"

Fencing, not mutual exclusion. `Store.claim` writes a new `owner_token`; every write by a runner
goes through `Store.tx(run_id, token)`, which re-reads the token inside the `BEGIN IMMEDIATE`
transaction and raises `LostOwnership` on mismatch. `test_stale_runner_is_fenced_out` has a tool
that claims the run mid-call; the original runner's result commit is refused and the call stays
`started`.

What it does not stop: two runners started together can both begin the same in-flight tool before
one is fenced. Preventing that needs a lease with heartbeat (a job-queue concern, deliberately
left out; see `docs/ENGINEERING_DECISIONS.md` section 4). Operationally, `resume` assumes the
previous process is gone.

### 4. "How do you handle a model that returns garbage?"

Three layers, all bounded by one `max_repairs` counter per run:

- Tool calls are checked by `ToolRegistry.check_call` before anything runs: unknown tool names
  and arguments failing the tool's JSON Schema are stored as `rejected` and returned to the model
  as the tool result (`_Driver._record_tool_turn`). The HTTP provider passes unparseable argument
  strings through unchanged (`_parse_arguments` in `providers/openai_compat.py`) so the same
  validation rejects them.
- Final answers go through `parse_json_output` and `validation_errors` (`schema.py`); errors are
  path-qualified and capped at five, then sent back as a user message
  (`_Driver._record_final_turn`).
- Past the limit the run fails with `output_invalid: repair limit N exceeded; last errors: ...`.

Evidence: `test_invalid_output_is_repaired_with_the_validation_error`,
`test_output_that_never_validates_fails_after_the_repair_limit`,
`test_invalid_tool_arguments_are_rejected_without_executing`,
`test_http_provider_drives_a_full_run`.

Tradeoff: one shared counter is simpler to reason about than separate budgets per error kind, but
a model that keeps getting arguments wrong uses up repairs it might have needed for the answer.

### 5. "How do you know the crash tests test a crash?"

They kill a separate OS process. `tests/test_crash_resume.py` starts `python -m agentflow resume`
in a subprocess. The child stops at a known point, either right after a commit
(`faults.stall_listener` on a trace event, which `_Driver.emit` writes after the commit) or inside
a tool after its side effect (`AGENTS_HANG_IN` in `tests/agents.py`). The parent then kills it
(`faults.kill_when_marked`), so no `finally` blocks, atexit handlers or SQLite close run in the
child. The test then opens the database in the parent, asserts the recorded state, resumes, and
counts executions per pid from `executions.jsonl`.

A portability detail found while building it: on Windows the venv `python.exe` is a launcher, so
the pid to kill is the one written in the marker, and `_wait_gone` waits for that interpreter to
exit before the database is reopened; without that wait the parent got `disk I/O error`.

## Tradeoffs worth raising unprompted

- SQLite over PostgreSQL: one file per data directory and no server, at the cost of a single
  writer. A run has one writer anyway.
- Sequential tools within a turn: simpler checkpoints and ordering, slower for independent reads.
- Absolute deadline: survives crashes, but a run resumed late fails immediately, including a
  `blocked` run whose operator answered after the deadline (`blocked -> failed`).
- Trace after commit: the trace may miss the last event; the store never does.
- No agent framework: the loop is small enough to read in one sitting, at the cost of features
  such as streaming and parallel tool calls.

## What not to claim

- Not exactly-once execution.
- Not tested against a real model API; the HTTP provider is tested with `httpx.MockTransport`.
- Not complete secret redaction; it is key-segment and value-shape matching, documented in the
  README.
- No numbers on model quality; the demo uses a scripted model.
