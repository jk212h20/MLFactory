"""Tests for the runner's on-disk layout and event log."""

from __future__ import annotations

import json
from pathlib import Path

from mlfactory.runner.events import read_events, write_event
from mlfactory.runner.layout import RunLayout, list_runs, new_run_id


def test_new_run_id_format() -> None:
    rid = new_run_id("my-test")
    # YYYY-MM-DD-HHMMSS-my-test
    parts = rid.split("-")
    assert len(parts) >= 5
    assert rid.endswith("-my-test")


def test_new_run_id_sanitises_name() -> None:
    rid = new_run_id("weird slash/name!")
    assert "/" not in rid
    assert "!" not in rid
    assert " " not in rid


def test_layout_roundtrip(tmp_path: Path) -> None:
    layout = RunLayout(root=tmp_path, game="boop", run_id="2026-01-01-000000-test")
    layout.ensure()
    assert layout.dir.exists()
    assert layout.samples_dir.exists()
    assert layout.checkpoints_dir.exists()

    layout.write_status("running")
    assert layout.read_status() == "running"
    layout.write_status("finished")
    assert layout.read_status() == "finished"

    layout.write_pid(12345)
    assert layout.read_pid() == 12345

    meta = {"git_sha": "abc123", "python": "3.13.0"}
    layout.write_meta(meta)
    assert layout.read_meta() == meta


def test_list_runs(tmp_path: Path) -> None:
    a = RunLayout(root=tmp_path, game="boop", run_id="2026-01-01-a")
    b = RunLayout(root=tmp_path, game="boop", run_id="2026-01-02-b")
    c = RunLayout(root=tmp_path, game="connect4", run_id="2026-01-03-c")
    for r in (a, b, c):
        r.ensure()

    all_runs = list_runs(tmp_path)
    assert {r.run_id for r in all_runs} == {a.run_id, b.run_id, c.run_id}

    boop_only = list_runs(tmp_path, game="boop")
    assert {r.run_id for r in boop_only} == {a.run_id, b.run_id}

    assert list_runs(tmp_path / "nowhere") == []


def test_write_and_read_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    write_event(path, "run_start", trainer="dummy", iters=5)
    write_event(path, "iter_start", iter=1)
    write_event(path, "iter_end", iter=1, duration_s=2.5)
    write_event(path, "run_end", status="finished", duration_s=10.0)

    events = list(read_events(path))
    assert len(events) == 4
    assert events[0]["type"] == "run_start"
    assert events[0]["iters"] == 5
    assert events[-1]["status"] == "finished"
    # Every event has a timestamp.
    for e in events:
        assert isinstance(e["t"], float)


def test_read_events_tolerates_missing_file(tmp_path: Path) -> None:
    assert list(read_events(tmp_path / "missing.jsonl")) == []


def test_read_events_tolerates_partial_tail(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    write_event(path, "run_start", trainer="dummy")
    # Append a partial line (as if writer was mid-write).
    with path.open("a") as f:
        f.write('{"t":1.0,"type":"iter_start","ite')  # no trailing newline
    events = list(read_events(path))
    assert len(events) == 1
    assert events[0]["type"] == "run_start"


def test_events_roundtrip_path_serialisation(tmp_path: Path) -> None:
    """Paths must serialise so we can write `path=...` directly."""
    path = tmp_path / "events.jsonl"
    write_event(path, "checkpoint", iter=1, path=tmp_path / "ckpt.pt", is_champion=True)
    events = list(read_events(path))
    assert len(events) == 1
    # Path got serialised as a string.
    assert isinstance(events[0]["path"], str)
    assert events[0]["is_champion"] is True


def test_event_line_is_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    write_event(path, "train", iter=1, policy_loss=0.1234, value_loss=0.05)
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["type"] == "train"
    assert parsed["policy_loss"] == 0.1234
