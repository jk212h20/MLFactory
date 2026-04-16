---
title: Training run event log (events.jsonl) format
status: stable
tags: [infrastructure, runner, observability]
provenance: internal
last_updated: 2026-04-16
---

# events.jsonl — training run event log

Every training run writes a machine-readable, append-only JSONL event log
to `experiments/<game>/<run-id>/events.jsonl`. One line = one event. The
file is the single source of truth for everything the run did, consumed by
the `mlfactory watch` TUI and (eventually) `mlfactory report`.

## Why a separate log

`run.log` captures stdout/stderr for humans (tail -f friendly). `events.jsonl`
captures structured data for tools. Separating them lets us:
- Render dashboards without regex-parsing text.
- Post-process runs into reports after they finish.
- Resume a watcher that missed a portion of the run.

## Line shape

Every event is a single JSON object with at least:

```json
{"t": <unix_timestamp_float>, "type": "<event_type>", ...payload}
```

The writer flushes after every event (unbuffered). Readers tolerate
malformed tail lines (writer mid-flush) and skip unknown event types.

## Event types (current)

| type | when | payload keys |
|---|---|---|
| `run_start` | once, first event | `trainer`, `iters`, any config summary |
| `run_end` | once, last event | `status` ∈ {finished, stopped, crashed}, `duration_s` |
| `iter_start` | each iteration start | `iter` |
| `iter_end` | each iteration end | `iter`, `duration_s` |
| `selfplay` | after self-play batch | `iter`, `games`, `avg_moves`, `orange_win_rate` |
| `train` | after training pass | `iter`, `batches`, `policy_loss`, `value_loss`, `total_loss` |
| `eval` | after evaluation match | `iter`, `opponent`, `wins`, `losses`, `draws`, `score`, `elo`, `elo_delta` |
| `checkpoint` | on save | `iter`, `path` (relative to run dir), `is_champion` (bool) |
| `sample_game` | on sample save | `iter`, `path`, `kind` ∈ {selfplay, eval, champion} |
| `log` | free-form | `level` ∈ {info, warn, error}, `msg` |

The payload keys are guaranteed; additional keys may be added over time.
**Readers must use `.get()` and tolerate missing/extra fields** for forward compat.

## Example (excerpt from a dummy-trainer run)

```jsonl
{"t": 1762400700.1, "type": "run_start", "trainer": "dummy", "iters": 5}
{"t": 1762400700.1, "type": "iter_start", "iter": 1}
{"t": 1762400700.3, "type": "selfplay", "iter": 1, "games": 50, "avg_moves": 41.7, "orange_win_rate": 0.52}
{"t": 1762400700.5, "type": "train", "iter": 1, "batches": 400, "policy_loss": 2.83, "value_loss": 0.72, "total_loss": 3.55}
{"t": 1762400700.7, "type": "eval", "iter": 1, "opponent": "prev", "wins": 11, "losses": 9, "draws": 0, "score": 0.55, "elo": 1502.1, "elo_delta": 2.1}
{"t": 1762400700.7, "type": "checkpoint", "iter": 1, "path": "experiments/boop/run/checkpoints/iter-0001.pt", "is_champion": false}
{"t": 1762400700.7, "type": "sample_game", "iter": 1, "path": "experiments/boop/run/samples/iter-0001/selfplay-game-01.json", "kind": "selfplay"}
{"t": 1762400700.8, "type": "iter_end", "iter": 1, "duration_s": 0.7}
...
{"t": 1762400704.0, "type": "run_end", "status": "finished", "duration_s": 3.9}
```

## Invariants callers can rely on

1. First event is `run_start`; last is `run_end` (unless the process was
   `SIGKILL`ed, in which case there may be no `run_end`).
2. `iter_start` always comes before the corresponding `iter_end`.
3. Event order within a single iteration is: `iter_start`, then any mix of
   `selfplay`/`train`/`eval`/`checkpoint`/`sample_game`/`log`, then `iter_end`.
4. Every path is relative to the workspace root (the directory containing
   `experiments/`), never absolute — so runs are relocatable.

## How to add a new event type

1. Extend `EventType` literal in `src/mlfactory/runner/events.py`.
2. Update this file's table.
3. Handle it in `_DashState.ingest` in `runner/watch.py` if it should
   show in the TUI.
4. No migration needed for historical logs — unknown types are skipped.
