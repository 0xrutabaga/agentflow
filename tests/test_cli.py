from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import agents
import pytest

from agentflow.cli import main


def run_cli(capsys: pytest.CaptureFixture[str], data_dir: Path, *argv: str) -> tuple[int, str]:
    code = main(["--data-dir", str(data_dir), *argv])
    return code, capsys.readouterr().out


def test_start_show_trace_and_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run_cli(capsys, tmp_path, "start", "--agent", agents.REF, "--goal", "check")
    assert code == 0
    run_id = out.split()[2]
    assert '"done": true' in out

    code, out = run_cli(capsys, tmp_path, "show", run_id)
    assert code == 0
    assert "status    completed" in out
    assert "model_calls=4/20" in out
    assert "2.0    send_email           succeeded  attempts=1" in out

    code, out = run_cli(capsys, tmp_path, "trace", run_id)
    assert code == 0
    assert "pending -> running  (started)" in out
    assert "tool_finished      0.0 read_a attempt 1" in out
    assert "running -> completed  (output valid)" in out

    code, out = run_cli(capsys, tmp_path, "trace", run_id, "--json")
    assert all(json.loads(line)["run_id"] == run_id for line in out.splitlines())

    code, out = run_cli(capsys, tmp_path, "runs")
    assert run_id in out
    assert "completed" in out


def test_blocked_run_resolved_from_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTS_FAIL_IN", "send_email")
    code, out = run_cli(capsys, tmp_path, "start", "--agent", agents.REF, "--goal", "g")
    assert code == 3
    run_id = out.split()[2]
    assert "blocked" in out
    monkeypatch.delenv("AGENTS_FAIL_IN")

    code, out = run_cli(capsys, tmp_path, "show", run_id)
    assert "send_email           ambiguous" in out

    code, out = run_cli(capsys, tmp_path, "resolve", run_id, "2.0", "--result", '{"sent": true}')
    assert code == 0
    code, out = run_cli(capsys, tmp_path, "resume", run_id)
    assert code == 0
    assert [e["tool"] for e in agents.executions(tmp_path)].count("send_email") == 1


def test_cancel_from_the_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from agentflow.runtime import Runtime

    runtime = Runtime(tmp_path)
    run_id = runtime.create(agents.REF, "g", agent=agents.build(tmp_path))
    runtime.close()
    code, out = run_cli(capsys, tmp_path, "cancel", run_id)
    assert (code, out.strip()) == (0, f"run {run_id}: cancelled")
    code, _ = run_cli(capsys, tmp_path, "resume", run_id)
    assert code == 4


def test_unknown_run_and_bad_resolve(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--data-dir", str(tmp_path), "show", "nope"]) == 2
    _, out = run_cli(capsys, tmp_path, "start", "--agent", agents.REF, "--goal", "g")
    run_id = out.split()[2]
    assert main(["--data-dir", str(tmp_path), "resolve", run_id, "0.0", "--retry"]) == 2
    assert main(["--data-dir", str(tmp_path), "trace", "missing"]) == 2


def test_demo_end_to_end(tmp_path: Path) -> None:
    workdir = tmp_path / "demo"
    proc = subprocess.run(
        [sys.executable, "-m", "agentflow", "demo", "--workdir", str(workdir)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "status=waiting_tool" in out
    assert "1.1 get_recent_deploys   requested" in out
    assert "final status: completed" in out
    assert "killed child    search_logs          connection_error  x2" in out
    assert "killed child    search_logs          ok                x1" in out
    assert "resumed parent  create_ticket        ok                x1" in out
    assert "resumed parent  get_alert" not in out
    assert "resumed parent  search_logs" not in out
    assert "tickets on disk: TCK-0001.json" in out
    assert '"ticket_id": "TCK-0001"' in out
    assert "output repair #1" in out
    # The custom redactor masked the reporter's phone number and the default one the token.
    trace = next((workdir / "traces").glob("*.jsonl")).read_text(encoding="utf-8")
    assert "+1 555 0100" not in trace
    assert "demo-session-value" not in trace


def test_demo_refuses_a_non_empty_workdir(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    assert main(["demo", "--workdir", str(tmp_path)]) == 2
