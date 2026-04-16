"""Event log: append-only JSONL of training events.

Schema (documented in wiki/techniques/events-log-format.md):

    {"t": <unix_ts>, "type": <EventType>, ...payload}

Types:
    "run_start"     — one per run, payload: config summary, meta
    "run_end"       — one per run, payload: status (finished|stopped|crashed), duration_s
    "iter_start"    — once per training iteration, payload: iter
    "iter_end"      — once per training iteration, payload: iter, duration_s
    "selfplay"      — per self-play batch, payload: iter, games, avg_moves, orange_win_rate
    "train"         — per training pass, payload: iter, batches, policy_loss, value_loss, total_loss
    "eval"          — per evaluation, payload: iter, opponent, wins, losses, draws, score, elo_delta
    "checkpoint"    — on checkpoint save, payload: iter, path, is_champion
    "sample_game"   — on sample save, payload: iter, path, kind ("selfplay"|"eval"|"champion")
    "log"           — free-form human message, payload: level ("info"|"warn"|"error"), msg

Readers must tolerate unknown event types (forward compat).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, Literal, TypedDict

EventType = Literal[
    "run_start",
    "run_end",
    "iter_start",
    "iter_end",
    "selfplay",
    "train",
    "eval",
    "checkpoint",
    "sample_game",
    "log",
]


class Event(TypedDict, total=False):
    t: float
    type: EventType
    # --- payloads are loosely typed; readers use .get() ---


def write_event(events_path: Path, event_type: EventType, /, **payload: Any) -> None:
    """Append a single JSONL event to `events_path`. Flushes immediately.

    The first two args are positional-only so event payloads can use the
    key `path` freely (e.g. for checkpoint paths).
    """
    evt: dict[str, Any] = {"t": time.time(), "type": event_type, **payload}
    line = json.dumps(evt, sort_keys=True, default=_json_default)
    with events_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def read_events(events_path: Path) -> Iterator[Event]:
    """Yield all events from `events_path`, skipping malformed lines."""
    if not events_path.exists():
        return
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip partially-written tail line (writer may not have flushed).
                continue


def _json_default(o: Any) -> Any:
    """Fallback for unserialisable types (numpy scalars, paths, etc.)."""
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "__float__"):
        return float(o)
    return str(o)
