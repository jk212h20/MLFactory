"""Launch a training run as a detached subprocess.

The parent process (the CLI) returns immediately after the child is started.
The child continues running even if the parent exits, writes everything to
disk in the run directory, and can be monitored via `mlfactory watch <run-id>`.
"""

from __future__ import annotations

import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from mlfactory.runner.layout import RunLayout


def launch_run(
    layout: RunLayout,
    trainer_module: str,
    trainer_args: list[str],
    *,
    config_summary: dict | None = None,
) -> int:
    """Spawn the trainer subprocess; return its pid.

    Parameters
    ----------
    layout
        The run's directory layout.
    trainer_module
        Python module path, e.g., "mlfactory.runner.dummy_trainer".
    trainer_args
        Arguments passed after the module; --run-dir is added automatically.
    config_summary
        Small dict saved to meta.json for display in list/watch.
    """
    layout.ensure()

    # Write meta (git sha, python, hardware) before the child starts so the
    # watcher can display it even before the first event arrives.
    meta = _collect_meta()
    if config_summary:
        meta["config_summary"] = config_summary
    layout.write_meta(meta)
    layout.write_status("starting")

    # Redirect child stdout+stderr to run.log so we can tail it.
    log = layout.log_path.open("ab", buffering=0)
    cmd = [
        sys.executable,
        "-u",  # unbuffered stdout
        "-m",
        trainer_module,
        "--run-dir",
        str(layout.dir),
        *trainer_args,
    ]
    # Detach: new session so parent exit doesn't kill child, and signals
    # (like ^C in the parent) don't propagate to the detached trainer.
    kwargs: dict = dict(
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        cwd=str(layout.root),
    )
    if hasattr(os, "setsid"):
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)
    layout.write_pid(proc.pid)
    # Don't wait. Let the child run.
    return proc.pid


def stop_run(layout: RunLayout, timeout: float = 30.0) -> str:
    """Request the run to stop gracefully; fall back to SIGKILL after timeout.

    Returns a status string: 'not_running', 'stopped', or 'killed'.
    """
    pid = layout.read_pid()
    if pid is None or not _pid_alive(pid):
        return "not_running"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "not_running"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return "stopped"
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "stopped"
    return "killed"


def run_is_alive(layout: RunLayout) -> bool:
    """True iff the run's pid exists and is a running process."""
    pid = layout.read_pid()
    if pid is None:
        return False
    return _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    """True iff `pid` refers to a running (non-zombie) process.

    On macOS (and BSD), `os.kill(pid, 0)` returns successfully even for
    zombies and for already-reaped-but-not-reallocated pids in some edge
    cases; the authoritative check is `ps -p`.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Confirm via ps to filter zombies (status Z).
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "stat="],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return False
    if not out:
        return False
    # Status starts with Z for zombies; anything else = running/sleeping/stopped.
    return not out.startswith("Z")


def _collect_meta() -> dict:
    meta: dict = {
        "start_time": time.time(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
    }
    # Git SHA (best-effort, may not be a git checkout)
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        meta["git_sha"] = sha
    except Exception:  # noqa: BLE001
        meta["git_sha"] = None
    # Torch / MPS (best-effort; don't fail if torch missing)
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        meta["torch"] = None
        meta["mps_available"] = False
    return meta
