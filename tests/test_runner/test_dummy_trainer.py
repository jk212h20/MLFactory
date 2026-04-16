"""End-to-end: spawn the dummy trainer, wait for it to finish, check outputs."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from mlfactory.runner.events import read_events
from mlfactory.runner.launcher import launch_run, run_is_alive, stop_run
from mlfactory.runner.layout import RunLayout, new_run_id


def test_dummy_trainer_runs_to_completion(tmp_path: Path) -> None:
    layout = RunLayout(root=tmp_path, game="testgame", run_id=new_run_id("unittest"))
    pid = launch_run(
        layout,
        trainer_module="mlfactory.runner.dummy_trainer",
        trainer_args=["--iters", "3", "--iter-seconds", "0.1", "--seed", "7"],
        config_summary={"trainer": "dummy", "iters": 3},
    )
    assert pid > 0

    # Wait up to 10s for the run to finish.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if layout.read_status() == "finished":
            break
        time.sleep(0.1)
    else:
        # Debug output if the trainer never finished.
        if layout.log_path.exists():
            print("--- run.log ---", file=sys.stderr)
            print(layout.log_path.read_text(), file=sys.stderr)
        raise AssertionError("dummy trainer did not finish within 10s")

    # Meta was written before the child started.
    meta = layout.read_meta()
    assert "python" in meta
    assert meta.get("config_summary", {}).get("iters") == 3

    # Events cover the full lifecycle.
    events = list(read_events(layout.events_path))
    types = [e["type"] for e in events]
    assert types[0] == "run_start"
    assert types[-1] == "run_end"
    # Expect at least 3 iter_start and 3 iter_end.
    assert types.count("iter_start") == 3
    assert types.count("iter_end") == 3
    assert types.count("selfplay") == 3
    assert types.count("train") == 3
    assert types.count("eval") == 3
    assert types.count("checkpoint") == 3
    assert types.count("sample_game") == 3

    # Checkpoints and samples actually exist.
    ckpts = list(layout.checkpoints_dir.glob("*.pt"))
    assert len(ckpts) == 3
    samples = list(layout.samples_dir.rglob("*.json"))
    assert len(samples) == 3


def test_stop_run_gracefully_terminates(tmp_path: Path) -> None:
    layout = RunLayout(root=tmp_path, game="testgame", run_id=new_run_id("stoptest"))
    launch_run(
        layout,
        trainer_module="mlfactory.runner.dummy_trainer",
        trainer_args=["--iters", "100", "--iter-seconds", "0.5", "--seed", "0"],
    )

    # Wait for the run to be alive and have emitted at least one event.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if run_is_alive(layout) and layout.events_path.exists():
            events = list(read_events(layout.events_path))
            if any(e["type"] == "run_start" for e in events):
                break
        time.sleep(0.1)
    else:
        raise AssertionError("trainer didn't start")

    result = stop_run(layout, timeout=5.0)
    assert result in ("stopped", "killed")

    # Give the OS a moment to fully reap the process after SIGKILL.
    for _ in range(20):
        if not run_is_alive(layout):
            break
        time.sleep(0.1)
    assert not run_is_alive(layout), f"process still alive after stop_run returned {result}"

    # Status file should reflect the stop.
    status = layout.read_status()
    assert status in ("stopped", "killed", "crashed"), status


def test_stop_run_on_nothing_returns_not_running(tmp_path: Path) -> None:
    layout = RunLayout(root=tmp_path, game="testgame", run_id="never-started")
    layout.ensure()
    assert stop_run(layout, timeout=1.0) == "not_running"
