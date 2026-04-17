"""Live TUI for watching a run. Tails events.jsonl and renders a dashboard.

Exits cleanly on Ctrl-C. Does NOT affect the run itself — the viewer is
purely a reader of the run's event log on disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mlfactory.runner.events import Event, read_events
from mlfactory.runner.launcher import run_is_alive
from mlfactory.runner.layout import RunLayout


@dataclass
class _DashState:
    """Rolling aggregates derived from events for the dashboard."""

    run_start_t: float | None = None
    run_end_t: float | None = None
    run_status: str = "?"
    current_iter: int | None = None
    iters_total: int | None = None
    last_iter_end_t: float | None = None
    iter_durations: list[float] = field(default_factory=list)
    # latest-per-type
    last_selfplay: dict | None = None
    last_train: dict | None = None
    last_eval: dict | None = None
    # history (small rolling window)
    eval_history: list[dict] = field(default_factory=list)
    loss_history: list[dict] = field(default_factory=list)
    # champion
    champion_iter: int | None = None
    champion_path: str | None = None
    sample_count: int = 0
    log_messages: list[tuple[str, str]] = field(default_factory=list)  # (level, msg)

    def ingest(self, evt: Event) -> None:
        t = evt.get("t")
        typ = evt.get("type")
        if typ == "run_start":
            self.run_start_t = t
            self.iters_total = evt.get("iters")
            self.run_status = "running"
        elif typ == "run_end":
            self.run_end_t = t
            self.run_status = evt.get("status", "finished")
        elif typ == "iter_start":
            self.current_iter = evt.get("iter")
        elif typ == "iter_end":
            self.last_iter_end_t = t
            dur = evt.get("duration_s")
            if dur is not None:
                self.iter_durations.append(float(dur))
                # keep last 30
                if len(self.iter_durations) > 30:
                    self.iter_durations = self.iter_durations[-30:]
        elif typ == "selfplay":
            self.last_selfplay = dict(evt)
        elif typ == "train":
            self.last_train = dict(evt)
            self.loss_history.append(
                {
                    "iter": evt.get("iter"),
                    "total": evt.get("total_loss"),
                    "policy": evt.get("policy_loss"),
                    "value": evt.get("value_loss"),
                }
            )
            if len(self.loss_history) > 20:
                self.loss_history = self.loss_history[-20:]
        elif typ == "eval":
            self.last_eval = dict(evt)
            self.eval_history.append(dict(evt))
            if len(self.eval_history) > 20:
                self.eval_history = self.eval_history[-20:]
        elif typ == "checkpoint":
            if evt.get("is_champion"):
                self.champion_iter = evt.get("iter")
                self.champion_path = evt.get("path")
        elif typ == "sample_game":
            self.sample_count += 1
        elif typ == "log":
            self.log_messages.append((evt.get("level", "info"), evt.get("msg", "")))
            if len(self.log_messages) > 5:
                self.log_messages = self.log_messages[-5:]


def _render(layout: RunLayout, state: _DashState, *, alive: bool) -> Group:
    """Build the full dashboard renderable from the current state."""
    # Header
    meta = layout.read_meta()
    header_tbl = Table.grid(padding=(0, 1), expand=True)
    header_tbl.add_column(justify="left")
    header_tbl.add_column(justify="right")
    alive_tag = "[green]ALIVE[/green]" if alive else "[red]DEAD[/red]"
    header_tbl.add_row(
        f"[bold]{layout.game} / {layout.run_id}[/bold]",
        f"status: [bold]{state.run_status}[/bold]  proc: {alive_tag}",
    )
    config = meta.get("config_summary") or {}
    cfg_line = "  ".join(f"{k}={v}" for k, v in list(config.items())[:6]) or "(no config)"
    header_tbl.add_row(Text(cfg_line, style="dim"), "")

    # Progress
    cur = state.current_iter or 0
    tot = state.iters_total or 0
    eta_s: float | None = None
    if state.iter_durations and tot and cur < tot:
        avg = sum(state.iter_durations) / len(state.iter_durations)
        eta_s = avg * (tot - cur)
    bar = _progress_bar(cur, tot)
    eta_str = _format_duration(eta_s) if eta_s else "—"
    avg_iter = (
        f"{sum(state.iter_durations) / len(state.iter_durations):.1f}s"
        if state.iter_durations
        else "—"
    )
    header_tbl.add_row(
        f"iter {cur}/{tot or '?'}  {bar}",
        f"avg/iter {avg_iter}  ETA {eta_str}",
    )

    # Latest numbers
    nums = Table.grid(padding=(0, 2))
    nums.add_column()
    nums.add_column()
    nums.add_column()
    nums.add_column()
    if state.last_selfplay:
        sp = state.last_selfplay
        nums.add_row(
            "[bold]selfplay[/bold]",
            f"games={sp.get('games', '—')}",
            f"avg_moves={sp.get('avg_moves', '—')}",
            f"orange_wr={sp.get('orange_win_rate', '—')}",
        )
    if state.last_train:
        tr = state.last_train
        nums.add_row(
            "[bold]train[/bold]",
            f"batches={tr.get('batches', '—')}",
            f"policy_loss={tr.get('policy_loss', '—')}",
            f"value_loss={tr.get('value_loss', '—')}",
        )
        # Optional diagnostics: only show when present (backward compat with
        # older runs whose train events didn't carry them).
        if any(k in tr for k in ("policy_entropy", "value_abs_mean", "value_std")):
            nums.add_row(
                "[bold]diag[/bold]",
                f"pol_entropy={tr.get('policy_entropy', '—')}",
                f"|v|_mean={tr.get('value_abs_mean', '—')}",
                f"v_std={tr.get('value_std', '—')}",
            )
    if state.last_eval:
        ev = state.last_eval
        nums.add_row(
            "[bold]eval[/bold]",
            f"vs={ev.get('opponent', '—')}",
            f"{ev.get('wins', '?')}-{ev.get('losses', '?')}-{ev.get('draws', '?')}",
            f"elo={ev.get('elo', '—')}  Δ={ev.get('elo_delta', '—')}",
        )
    nums.add_row(
        "[bold]meta[/bold]",
        f"champion@iter={state.champion_iter or '—'}",
        f"samples={state.sample_count}",
        "",
    )

    # Eval history sparkline
    hist_lines: list[Text] = []
    if state.eval_history:
        spark = _sparkline([float(e.get("score", 0.0)) for e in state.eval_history])
        hist_lines.append(Text(f"eval score (recent): {spark}"))
    if state.loss_history:
        losses = [
            float(h.get("total", 0.0)) for h in state.loss_history if h.get("total") is not None
        ]
        if losses:
            spark = _sparkline(losses, reverse_color=True)
            hist_lines.append(Text(f"train loss (recent): {spark}"))

    # Recent log messages
    logs = Table.grid(padding=(0, 1))
    logs.add_column()
    if state.log_messages:
        for level, msg in state.log_messages[-3:]:
            color = {"error": "red", "warn": "yellow"}.get(level, "white")
            logs.add_row(Text(f"[{level}] {msg}", style=color))
    else:
        logs.add_row(Text("(no log messages)", style="dim"))

    return Group(
        Panel(header_tbl, title="run"),
        Panel(nums, title="latest"),
        Panel(
            Group(*hist_lines) if hist_lines else Text("(no history yet)", style="dim"),
            title="history",
        ),
        Panel(logs, title="log"),
    )


def _progress_bar(cur: int, tot: int, width: int = 30) -> str:
    if not tot:
        return "[" + " " * width + "]"
    filled = int(width * cur / max(tot, 1))
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def _format_duration(s: float | None) -> str:
    if s is None:
        return "—"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _sparkline(values: list[float], reverse_color: bool = False) -> str:
    if not values:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-9:
        return bars[3] * len(values)
    out = []
    for v in values:
        idx = int((v - vmin) / (vmax - vmin) * (len(bars) - 1))
        out.append(bars[idx])
    return "".join(out)


def watch(layout: RunLayout, refresh_hz: float = 4.0) -> None:
    """Run the live dashboard until the user Ctrl-Cs or the run ends + 3s."""
    state = _DashState()
    # Track file position between reads for efficient tail-follow.
    offset = 0
    console = Console()

    def _read_new() -> list[Event]:
        nonlocal offset
        evs: list[Event] = []
        p = layout.events_path
        if not p.exists():
            return evs
        size = p.stat().st_size
        if size < offset:
            # file rotated/truncated (shouldn't happen, but guard)
            offset = 0
        if size == offset:
            return evs
        import json

        with p.open("r", encoding="utf-8") as f:
            f.seek(offset)
            for line in f:
                if line.endswith("\n"):
                    try:
                        evs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            offset = f.tell()
        return evs

    # Warm-start: ingest any events already written before we attached.
    for e in read_events(layout.events_path):
        state.ingest(e)
    offset = layout.events_path.stat().st_size if layout.events_path.exists() else 0

    end_linger_until: float | None = None
    try:
        with Live(
            _render(layout, state, alive=run_is_alive(layout)),
            refresh_per_second=refresh_hz,
            console=console,
            transient=False,
        ) as live:
            while True:
                for e in _read_new():
                    state.ingest(e)
                alive = run_is_alive(layout)
                live.update(_render(layout, state, alive=alive))
                if state.run_end_t is not None and end_linger_until is None:
                    end_linger_until = time.monotonic() + 3.0
                if end_linger_until is not None and time.monotonic() >= end_linger_until:
                    break
                time.sleep(1.0 / refresh_hz)
    except KeyboardInterrupt:
        console.print("\n[dim](watcher detached; run continues)[/dim]")
