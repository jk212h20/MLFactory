"""Runner: subprocess-based training orchestration with live event logs.

Philosophy: one run = one OS process = one directory on disk. No daemon.
The parent process spawns training as a detached subprocess that writes

    experiments/<game>/<run-id>/
        config.yaml        # run config (user-provided, frozen)
        meta.json          # git sha, hardware, python versions, start time
        pid                # process id of the training process
        run.log            # human-readable text log (tail -f friendly)
        events.jsonl       # one JSON event per line (machine-readable)
        status             # one of: running, finished, stopped, crashed
        samples/iter-NNN/  # sample games saved each iteration
        checkpoints/       # saved net weights

The watch TUI tails events.jsonl and renders a live dashboard. The run
process never depends on the viewer being attached.
"""

from mlfactory.runner.events import Event, EventType, read_events, write_event
from mlfactory.runner.layout import RunLayout, new_run_id

__all__ = ["Event", "EventType", "RunLayout", "new_run_id", "read_events", "write_event"]
