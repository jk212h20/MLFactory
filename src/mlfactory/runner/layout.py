"""On-disk layout for a run directory.

A `RunLayout` is a thin wrapper around the directory `experiments/<game>/<run-id>/`.
It knows the canonical names for all files and ensures parent directories exist.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path


def new_run_id(name: str | None = None) -> str:
    """Build a run id: YYYY-MM-DD-HHMMSS-<name>. Name is slugified lightly."""
    stamp = _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if name:
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.strip())
        slug = slug.strip("-") or "run"
        return f"{stamp}-{slug}"
    return f"{stamp}-run"


@dataclass(frozen=True)
class RunLayout:
    """Path helpers for one run directory."""

    root: Path
    game: str
    run_id: str

    @property
    def dir(self) -> Path:
        return self.root / "experiments" / self.game / self.run_id

    @property
    def config_path(self) -> Path:
        return self.dir / "config.yaml"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def pid_path(self) -> Path:
        return self.dir / "pid"

    @property
    def log_path(self) -> Path:
        return self.dir / "run.log"

    @property
    def events_path(self) -> Path:
        return self.dir / "events.jsonl"

    @property
    def status_path(self) -> Path:
        return self.dir / "status"

    @property
    def samples_dir(self) -> Path:
        return self.dir / "samples"

    @property
    def checkpoints_dir(self) -> Path:
        return self.dir / "checkpoints"

    def ensure(self) -> None:
        """Create the run directory and its children. Safe to call twice."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

    def write_status(self, status: str) -> None:
        self.status_path.write_text(status.strip() + "\n")

    def read_status(self) -> str:
        if not self.status_path.exists():
            return "unknown"
        return self.status_path.read_text().strip()

    def write_pid(self, pid: int) -> None:
        self.pid_path.write_text(f"{pid}\n")

    def read_pid(self) -> int | None:
        if not self.pid_path.exists():
            return None
        try:
            return int(self.pid_path.read_text().strip())
        except ValueError:
            return None

    def write_meta(self, meta: dict) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    def read_meta(self) -> dict:
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text())


def list_runs(root: Path, game: str | None = None) -> list[RunLayout]:
    """Enumerate all runs under `root`, optionally filtered by game."""
    out: list[RunLayout] = []
    exp_root = root / "experiments"
    if not exp_root.exists():
        return out
    games = [game] if game else [p.name for p in exp_root.iterdir() if p.is_dir()]
    for g in games:
        game_dir = exp_root / g
        if not game_dir.exists():
            continue
        for rd in sorted(game_dir.iterdir()):
            if rd.is_dir():
                out.append(RunLayout(root=root, game=g, run_id=rd.name))
    return out
