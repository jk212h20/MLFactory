"""MLFactory CLI."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mlfactory import __version__
from mlfactory.agents.base import Agent
from mlfactory.agents.mcts import MCTSAgent
from mlfactory.agents.random_agent import RandomAgent
from mlfactory.core.env import Env
from mlfactory.games.boop import Boop
from mlfactory.games.connect4 import Connect4
from mlfactory.runner.launcher import launch_run, run_is_alive, stop_run
from mlfactory.runner.layout import RunLayout, list_runs, new_run_id
from mlfactory.runner.replay import replay_file
from mlfactory.runner.watch import watch as watch_run
from mlfactory.tools.arena import round_robin

app = typer.Typer(
    name="mlfactory",
    help="MLFactory: a self-improving automated games strategy research factory.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed MLFactory version."""
    console.print(f"mlfactory {__version__}")


@app.command()
def doctor() -> None:
    """Check the environment: python, torch, MPS."""
    console.print(f"[bold]Python[/bold]: {sys.version.split()[0]}")
    try:
        import torch

        console.print(f"[bold]Torch[/bold]: {torch.__version__}")
        console.print(f"[bold]MPS available[/bold]: {torch.backends.mps.is_available()}")
        console.print(f"[bold]MPS built[/bold]: {torch.backends.mps.is_built()}")
    except ImportError:
        console.print("[red]Torch not installed[/red]")


# --- Game + agent registries --------------------------------------------

GAMES: dict[str, type] = {
    "connect4": Connect4,
    "boop": Boop,
}


def _make_env(game: str) -> Env:
    if game not in GAMES:
        raise typer.BadParameter(f"unknown game '{game}'. Options: {list(GAMES)}")
    return GAMES[game]()


def _make_agent(spec: str, seed: int) -> Agent:
    """Create an agent from a short spec string.

    Examples: 'random', 'mcts100', 'mcts:sims=800,c=1.5'.
    """
    if spec == "random":
        return RandomAgent(name="random", seed=seed)
    if spec.startswith("mcts"):
        rest = spec[len("mcts") :]
        # 'mcts200' → sims=200, or 'mcts:sims=800,c=1.4'
        if rest.startswith(":"):
            kwargs: dict[str, str] = {}
            for kv in rest[1:].split(","):
                k, v = kv.split("=")
                kwargs[k.strip()] = v.strip()
            sims = int(kwargs.get("sims", "200"))
            c = float(kwargs.get("c", "1.4142"))
        else:
            sims = int(rest) if rest else 200
            c = 1.4142
        return MCTSAgent(n_simulations=sims, c=c, seed=seed, name=spec)
    raise typer.BadParameter(f"unknown agent spec '{spec}'")


@app.command()
def tournament(
    game: str = typer.Option("connect4", help="Game name."),
    agents: str = typer.Option(
        "random,mcts50,mcts200,mcts800",
        help="Comma-separated agent specs.",
    ),
    games_per_match: int = typer.Option(40, help="Games per pairwise match."),
    seed: int = typer.Option(0, help="Base RNG seed."),
    progress: bool = typer.Option(False, help="Print progress during long matches."),
) -> None:
    """Run a full round-robin tournament and print the results."""
    env = _make_env(game)
    specs = [s.strip() for s in agents.split(",") if s.strip()]
    agent_list: list[Agent] = [_make_agent(spec, seed + i) for i, spec in enumerate(specs)]

    console.print(
        f"\n[bold]Tournament[/bold]: {game}, "
        f"{len(agent_list)} agents, {games_per_match} games/match"
    )
    console.print(f"Agents: {[a.name for a in agent_list]}\n")

    start = time.monotonic()
    result = round_robin(env, agent_list, games_per_match=games_per_match, progress=progress)
    elapsed = time.monotonic() - start

    # Win-rate matrix
    matrix = result.matrix()
    matrix_tbl = Table(title="Win-rate matrix (row vs column)", show_lines=True)
    matrix_tbl.add_column("agent")
    for name in result.agents:
        matrix_tbl.add_column(name)
    for row_name, row in zip(result.agents, matrix):
        matrix_tbl.add_row(row_name, *row)
    console.print(matrix_tbl)

    # Pairwise details
    detail_tbl = Table(title="Pairwise details (95% Wilson CI for A's score)", show_lines=False)
    for col in ("A", "B", "A_wins", "B_wins", "draws", "A_score", "95% CI"):
        detail_tbl.add_column(col)
    for p in result.pairwise:
        lo, hi = p.wilson_ci()
        detail_tbl.add_row(
            p.agent_a_name,
            p.agent_b_name,
            str(p.a_wins),
            str(p.b_wins),
            str(p.draws),
            f"{p.a_win_rate:.3f}",
            f"[{lo:.3f}, {hi:.3f}]",
        )
    console.print(detail_tbl)

    # ELO leaderboard
    elo_tbl = Table(title="ELO leaderboard (anchor = first agent at 1500)")
    elo_tbl.add_column("rank")
    elo_tbl.add_column("agent")
    elo_tbl.add_column("ELO", justify="right")
    sorted_elo = sorted(result.elo.items(), key=lambda kv: -kv[1])
    for i, (name, rating) in enumerate(sorted_elo, 1):
        elo_tbl.add_row(str(i), name, f"{rating:.0f}")
    console.print(elo_tbl)

    console.print(
        f"\n[dim]Wall time: {elapsed:.1f}s "
        f"({sum(p.total for p in result.pairwise)} games total)[/dim]"
    )


@app.command()
def match(
    game: str = typer.Option("connect4", help="Game name."),
    agent_a: str = typer.Option(..., help="First agent spec, e.g. 'mcts200'."),
    agent_b: str = typer.Option(..., help="Second agent spec."),
    games: int = typer.Option(100, help="Total games (colour-balanced)."),
    seed: int = typer.Option(0, help="Base RNG seed."),
) -> None:
    """Play two agents against each other and print the result."""
    env = _make_env(game)
    a = _make_agent(agent_a, seed)
    b = _make_agent(agent_b, seed + 1)
    from mlfactory.tools.arena import play_match

    console.print(f"\n[bold]{a.name} vs {b.name}[/bold] on {game}, {games} games\n")
    result = play_match(env, a, b, n_games=games, progress=True)
    lo, hi = result.wilson_ci()
    console.print(
        f"\n{a.name}: {result.a_wins}W  {b.name}: {result.b_wins}W  draws: {result.draws}\n"
        f"{a.name} score: {result.a_win_rate:.3f}  95% CI: [{lo:.3f}, {hi:.3f}]"
    )


# --- Runner commands ----------------------------------------------------

# Trainer registry: maps --trainer values to Python module paths.
TRAINERS: dict[str, str] = {
    "dummy": "mlfactory.runner.dummy_trainer",
    "alphazero": "mlfactory.training.trainer",
    "alphazero_mandala": "mlfactory.training.trainer_mandala",
}


def _workspace_root() -> Path:
    """Repo root — the directory containing `experiments/`."""
    # The CLI is installed as a package; use CWD so `mlfactory` picks up
    # the user's workspace, not the installed wheel location.
    return Path.cwd()


def _resolve_run(run_id: str, game: str | None = None) -> RunLayout:
    """Find a run by exact id or by trailing-slug match."""
    root = _workspace_root()
    runs = list_runs(root, game=game)
    # Exact match first
    for r in runs:
        if r.run_id == run_id:
            return r
    # Endswith / contains match (useful for tab-completion-lite)
    matches = [r for r in runs if r.run_id.endswith(run_id) or run_id in r.run_id]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise typer.BadParameter(f"no run matching '{run_id}' under experiments/")
    raise typer.BadParameter(f"ambiguous run '{run_id}'; matches: {[m.run_id for m in matches]}")


@app.command("train")
def train(
    trainer: str = typer.Option("alphazero", help=f"Trainer: {sorted(TRAINERS)}"),
    game: str = typer.Option("boop", help="Game to train on."),
    name: str = typer.Option("", help="Optional human label appended to run id."),
    iters: int = typer.Option(5, help="Number of training iterations."),
    # Dummy-trainer option (ignored by alphazero)
    iter_seconds: float = typer.Option(1.0, help="[dummy only] seconds per fake iteration."),
    # AlphaZero-trainer options (ignored by dummy)
    selfplay_games: int = typer.Option(20, help="[az] games per self-play batch."),
    selfplay_sims: int = typer.Option(100, help="[az] PUCT sims per move in self-play."),
    eval_games: int = typer.Option(20, help="[az] games per eval match."),
    eval_sims: int = typer.Option(100, help="[az] PUCT sims per move in eval."),
    baseline_mcts_sims: int = typer.Option(200, help="[az] sims for the fixed MCTS baseline."),
    train_batches: int = typer.Option(100, help="[az] training minibatches per iteration."),
    batch_size: int = typer.Option(128, help="[az] training batch size."),
    lr: float = typer.Option(1e-3, help="[az] learning rate."),
    warmup_samples: int = typer.Option(256, help="[az] min buffer size before training starts."),
    net_blocks: int = typer.Option(4, help="[az] residual blocks."),
    net_channels: int = typer.Option(64, help="[az] channels per block."),
    device: str = typer.Option("mps", help="[az] torch device for training (mps|cpu|cuda)."),
    samples_per_iter: int = typer.Option(2, help="[az] self-play games saved per iteration."),
    no_augment: bool = typer.Option(False, help="[az] disable D4 symmetry augmentation."),
    n_workers: int = typer.Option(
        1, help="[az] worker processes for self-play/eval parallelism (1 = sequential)."
    ),
    mcts_eval_every: int = typer.Option(1, help="[az] run the mcts baseline eval every N iters."),
    random_eval_every: int = typer.Option(
        1, help="[az] run the random baseline eval every N iters."
    ),
    resume_from: str = typer.Option(
        "", help="[az] path to a prior checkpoint; warm-start net weights."
    ),
    baseline_ckpt: str = typer.Option(
        "", help="[az] path to a fixed opponent checkpoint for periodic eval."
    ),
    baseline_ckpt_every: int = typer.Option(5, help="[az] iters between baseline-ckpt evals."),
    baseline_ckpt_games: int = typer.Option(20, help="[az] games per baseline-ckpt eval match."),
    stop_on_baseline_win_rate: float = typer.Option(
        0.0,
        help="[az] stop when baseline-ckpt win rate >= this value "
        "(held for stop-requires-consecutive evals). 0 = disabled.",
    ),
    stop_requires_consecutive: int = typer.Option(
        1, help="[az] consecutive above-threshold baseline evals required to stop."
    ),
    stop_on_baseline_pvalue: float = typer.Option(
        0.0,
        help="[az] stop when one-sided binomial p-value of baseline-ckpt eval "
        "(H0: true_winrate=0.5) <= this. 0 = disabled. Recommended: 0.05 with "
        "--baseline-ckpt-games 40 for honest evidence-based stopping.",
    ),
    seed: int = typer.Option(0, help="Trainer RNG seed."),
) -> None:
    """Launch a training run as a detached subprocess and return immediately."""
    if trainer not in TRAINERS:
        raise typer.BadParameter(f"unknown trainer '{trainer}'. Options: {sorted(TRAINERS)}")

    root = _workspace_root()
    run_id = new_run_id(name or trainer)
    layout = RunLayout(root=root, game=game, run_id=run_id)

    trainer_args: list[str] = ["--iters", str(iters), "--seed", str(seed)]
    config_summary: dict = {
        "trainer": trainer,
        "game": game,
        "iters": iters,
        "seed": seed,
    }

    if trainer == "dummy":
        trainer_args += ["--iter-seconds", str(iter_seconds)]
        config_summary["iter_seconds"] = iter_seconds
    elif trainer == "alphazero":
        trainer_args += [
            "--selfplay-games",
            str(selfplay_games),
            "--selfplay-sims",
            str(selfplay_sims),
            "--eval-games",
            str(eval_games),
            "--eval-sims",
            str(eval_sims),
            "--baseline-mcts-sims",
            str(baseline_mcts_sims),
            "--train-batches",
            str(train_batches),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--warmup-samples",
            str(warmup_samples),
            "--net-blocks",
            str(net_blocks),
            "--net-channels",
            str(net_channels),
            "--device",
            device,
            "--samples-per-iter",
            str(samples_per_iter),
        ]
        if no_augment:
            trainer_args.append("--no-augment")
        trainer_args += [
            "--n-workers",
            str(n_workers),
            "--mcts-eval-every",
            str(mcts_eval_every),
            "--random-eval-every",
            str(random_eval_every),
            "--baseline-ckpt-every",
            str(baseline_ckpt_every),
            "--baseline-ckpt-games",
            str(baseline_ckpt_games),
            "--stop-on-baseline-win-rate",
            str(stop_on_baseline_win_rate),
            "--stop-requires-consecutive",
            str(stop_requires_consecutive),
            "--stop-on-baseline-pvalue",
            str(stop_on_baseline_pvalue),
        ]
        if resume_from:
            trainer_args += ["--resume-from", resume_from]
        if baseline_ckpt:
            trainer_args += ["--baseline-ckpt", baseline_ckpt]
        config_summary.update(
            {
                "selfplay_games": selfplay_games,
                "selfplay_sims": selfplay_sims,
                "eval_games": eval_games,
                "eval_sims": eval_sims,
                "baseline_mcts_sims": baseline_mcts_sims,
                "train_batches": train_batches,
                "batch_size": batch_size,
                "lr": lr,
                "net_blocks": net_blocks,
                "net_channels": net_channels,
                "device": device,
                "augment": not no_augment,
                "n_workers": n_workers,
                "mcts_eval_every": mcts_eval_every,
                "random_eval_every": random_eval_every,
                "resume_from": resume_from or None,
                "baseline_ckpt": baseline_ckpt or None,
                "baseline_ckpt_every": baseline_ckpt_every,
                "baseline_ckpt_games": baseline_ckpt_games,
                "stop_on_baseline_win_rate": stop_on_baseline_win_rate,
                "stop_requires_consecutive": stop_requires_consecutive,
                "stop_on_baseline_pvalue": stop_on_baseline_pvalue,
            }
        )

    pid = launch_run(
        layout,
        trainer_module=TRAINERS[trainer],
        trainer_args=trainer_args,
        config_summary=config_summary,
    )
    console.print(f"[green]started[/green] run [bold]{run_id}[/bold] (pid {pid})")
    console.print(f"  dir:   {layout.dir}")
    console.print(f"  watch: [cyan]mlfactory watch {run_id}[/cyan]")
    console.print(f"  stop:  [cyan]mlfactory stop {run_id}[/cyan]")
    console.print(f"  tail:  [dim]tail -f {layout.log_path}[/dim]")


@app.command("train-mandala")
def train_mandala(
    name: str = typer.Option("", help="Optional human label appended to run id."),
    iters: int = typer.Option(5, help="Number of training iterations."),
    selfplay_games: int = typer.Option(20, help="Self-play games per iteration."),
    selfplay_sims: int = typer.Option(50, help="PUCT sims per move in self-play."),
    eval_games: int = typer.Option(10, help="Games per eval match."),
    eval_sims: int = typer.Option(50, help="PUCT sims per move in eval."),
    train_batches: int = typer.Option(100, help="Training minibatches per iter."),
    batch_size: int = typer.Option(128, help="Training batch size."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
    warmup_samples: int = typer.Option(256, help="Min buffer size before training."),
    hidden: int = typer.Option(256, help="MLP hidden width."),
    n_blocks: int = typer.Option(4, help="MLP residual block count."),
    value_hidden: int = typer.Option(128, help="Value head hidden width."),
    device: str = typer.Option("mps", help="Training device."),
    samples_per_iter: int = typer.Option(2, help="Self-play games saved per iter."),
    n_workers: int = typer.Option(
        1, help="Parallel worker processes for self-play (1 = sequential)."
    ),
    resume_from: str = typer.Option("", help="Checkpoint path to warm-start from."),
    baseline_ckpt: str = typer.Option("", help="Fixed baseline checkpoint for periodic eval."),
    baseline_ckpt_every: int = typer.Option(5, help="Iters between baseline evals."),
    baseline_ckpt_games: int = typer.Option(20, help="Games per baseline eval match."),
    stop_on_baseline_pvalue: float = typer.Option(
        0.0, help="Early-stop when baseline eval p-value <= this. 0 = disabled."
    ),
    seed: int = typer.Option(0, help="RNG seed."),
) -> None:
    """Launch a Mandala training run as a detached subprocess."""
    game = "mandala"
    root = _workspace_root()
    run_id = new_run_id(name or "alphazero_mandala")
    layout = RunLayout(root=root, game=game, run_id=run_id)

    trainer_args: list[str] = [
        "--iters",
        str(iters),
        "--selfplay-games",
        str(selfplay_games),
        "--selfplay-sims",
        str(selfplay_sims),
        "--eval-games",
        str(eval_games),
        "--eval-sims",
        str(eval_sims),
        "--train-batches",
        str(train_batches),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--warmup-samples",
        str(warmup_samples),
        "--hidden",
        str(hidden),
        "--n-blocks",
        str(n_blocks),
        "--value-hidden",
        str(value_hidden),
        "--device",
        device,
        "--samples-per-iter",
        str(samples_per_iter),
        "--n-workers",
        str(n_workers),
        "--baseline-ckpt-every",
        str(baseline_ckpt_every),
        "--baseline-ckpt-games",
        str(baseline_ckpt_games),
        "--stop-on-baseline-pvalue",
        str(stop_on_baseline_pvalue),
        "--seed",
        str(seed),
    ]
    if resume_from:
        trainer_args += ["--resume-from", resume_from]
    if baseline_ckpt:
        trainer_args += ["--baseline-ckpt", baseline_ckpt]

    config_summary = {
        "trainer": "alphazero_mandala",
        "game": game,
        "iters": iters,
        "selfplay_games": selfplay_games,
        "selfplay_sims": selfplay_sims,
        "lr": lr,
        "hidden": hidden,
        "n_blocks": n_blocks,
        "n_workers": n_workers,
        "resume_from": resume_from or None,
        "baseline_ckpt": baseline_ckpt or None,
    }

    pid = launch_run(
        layout,
        trainer_module="mlfactory.training.trainer_mandala",
        trainer_args=trainer_args,
        config_summary=config_summary,
    )
    console.print(f"[green]started[/green] run [bold]{run_id}[/bold] (pid {pid})")
    console.print(f"  dir:   {layout.dir}")
    console.print(f"  watch: [cyan]mlfactory watch {run_id}[/cyan]")
    console.print(f"  stop:  [cyan]mlfactory stop {run_id}[/cyan]")
    console.print(f"  tail:  [dim]tail -f {layout.log_path}[/dim]")


@app.command("list")
def list_cmd(
    game: str = typer.Option("", help="Filter by game; empty = all."),
    limit: int = typer.Option(20, help="Max runs to show (most recent first)."),
) -> None:
    """List all training runs in the current workspace."""
    root = _workspace_root()
    runs = list_runs(root, game=game or None)
    if not runs:
        console.print("[dim]no runs found under experiments/[/dim]")
        return

    # Most recent first (run ids are timestamp-prefixed).
    runs = sorted(runs, key=lambda r: r.run_id, reverse=True)[:limit]

    tbl = Table(title=f"Runs ({len(runs)})")
    tbl.add_column("run_id")
    tbl.add_column("game")
    tbl.add_column("status")
    tbl.add_column("proc")
    tbl.add_column("events", justify="right")
    tbl.add_column("samples", justify="right")
    for r in runs:
        status = r.read_status()
        alive = "[green]alive[/green]" if run_is_alive(r) else "[dim]—[/dim]"
        # cheap event/sample counts
        n_events = 0
        if r.events_path.exists():
            with r.events_path.open("rb") as f:
                n_events = sum(1 for _ in f)
        n_samples = sum(1 for _ in r.samples_dir.rglob("*.json")) if r.samples_dir.exists() else 0
        tbl.add_row(r.run_id, r.game, status, alive, str(n_events), str(n_samples))
    console.print(tbl)


@app.command("watch")
def watch_cmd(
    run_id: str = typer.Argument(..., help="Run id (or trailing slug)."),
    game: str = typer.Option("", help="Disambiguate by game."),
    refresh: float = typer.Option(4.0, help="Refresh Hz."),
) -> None:
    """Attach a live TUI to a running (or finished) run. Ctrl-C to detach."""
    layout = _resolve_run(run_id, game=game or None)
    watch_run(layout, refresh_hz=refresh)


@app.command("active")
def active_cmd(
    log_paths: str = typer.Option(
        "/tmp/hp-distill*.log,/tmp/eval-*.log,/tmp/hp-vs-rand*.log,/tmp/notify-*.log",
        help=(
            "Comma-separated glob patterns to scan for ad-hoc job logs. "
            "Defaults cover the conventions used by recent ad-hoc scripts."
        ),
    ),
) -> None:
    """Show ALL active MLFactory work — registered training runs PLUS
    any ad-hoc detached scripts (eval matches, distillations, etc).

    Two sources:
      1. Registered training runs under experiments/ (same as
         `watch-active` searches). Shown with status + alive/dead.
      2. Python subprocesses currently running an mlfactory.* module,
         discovered via `ps`. Shown with pid, age, last log line.

    Use this when you've lost track of what's going. Quick snapshot only,
    not a live TUI; for live updates `tail -f <log>` or `mlfactory watch
    <run-id>` for registered runs.
    """
    import glob
    import os
    import subprocess
    from datetime import datetime

    # ---- Registered training runs ----
    root = _workspace_root()
    runs = list_runs(root, game=None)
    alive_runs = [r for r in runs if run_is_alive(r)]
    finished_runs = sorted(
        [r for r in runs if not run_is_alive(r)],
        key=lambda r: r.run_id,
        reverse=True,
    )

    console.print("[bold]Active runs (training, registered):[/bold]")
    if not alive_runs:
        console.print("  [dim](none)[/dim]")
    else:
        for r in sorted(alive_runs, key=lambda r: r.run_id, reverse=True):
            pid = r.read_pid()
            console.print(
                f"  [green]{r.run_id}[/green]  game={r.game}  pid={pid}  "
                f"watch: [cyan]mlfactory watch {r.run_id}[/cyan]"
            )

    # ---- Ad-hoc python subprocesses running mlfactory.* modules ----
    console.print("\n[bold]Active ad-hoc scripts (mlfactory.*):[/bold]")
    try:
        ps_out = subprocess.check_output(
            ["ps", "-Ao", "pid,etime,command"],
            text=True,
            timeout=5,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"  [red]ps failed: {e}[/red]")
        ps_out = ""

    adhoc = []
    for line in ps_out.splitlines()[1:]:
        # Skip our own line, ps itself, the typer process, and `grep` matches.
        if "mlfactory" not in line:
            continue
        if "ps -Ao" in line or "active_cmd" in line or "/cli.py" in line:
            continue
        # Skip the typer process running this very command.
        if "mlfactory active" in line or "mlfactory.cli" in line:
            continue
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, etime, cmd = parts
        # Identify the module from the command if possible.
        adhoc.append((pid, etime, cmd))

    if not adhoc:
        console.print("  [dim](none)[/dim]")
    else:
        for pid, etime, cmd in adhoc:
            # Truncate command for display.
            short = cmd if len(cmd) <= 100 else cmd[:97] + "..."
            console.print(f"  pid={pid}  age={etime}  cmd=[cyan]{short}[/cyan]")

    # ---- Recently-active log files ----
    console.print("\n[bold]Recent ad-hoc job logs (most recent first):[/bold]")
    patterns = [p.strip() for p in log_paths.split(",") if p.strip()]
    files: list[tuple[float, str]] = []
    for pat in patterns:
        for fp in glob.glob(os.path.expanduser(pat)):
            try:
                files.append((os.path.getmtime(fp), fp))
            except OSError:
                continue
    files.sort(reverse=True)
    if not files:
        console.print("  [dim](no matching log files)[/dim]")
    else:
        for mtime, fp in files[:10]:
            mtstr = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
            try:
                with open(fp, "r") as f:
                    lines = f.readlines()
                last = lines[-1].rstrip() if lines else "(empty)"
            except OSError as e:
                last = f"(read error: {e})"
            short_last = last if len(last) <= 80 else last[:77] + "..."
            console.print(f"  [dim]{mtstr}[/dim]  {fp}  [dim]| {short_last}[/dim]")

    # ---- Most recent finished run, as a hint ----
    if not alive_runs and finished_runs:
        console.print(
            f"\n[dim]most recent finished run: [bold]{finished_runs[0].run_id}[/bold]  "
            f"status={finished_runs[0].read_status()}[/dim]"
        )


@app.command("watch-active")
def watch_active_cmd(
    game: str = typer.Option("", help="Optionally filter by game."),
    refresh: float = typer.Option(4.0, help="Refresh Hz."),
) -> None:
    """Attach the watch TUI to the currently-running training run.

    Finds the newest run (by run_id timestamp) whose process is still
    alive. If several are alive, lists them and attaches to the newest.
    Exits with an error if none are alive.

    This is the "I don't want to type the run id" command: just
    `mlfactory watch-active` and it does the right thing.
    """
    root = _workspace_root()
    runs = list_runs(root, game=game or None)
    if not runs:
        console.print("[dim]no runs found under experiments/[/dim]")
        raise typer.Exit(code=1)

    alive_runs = [r for r in runs if run_is_alive(r)]
    if not alive_runs:
        console.print("[yellow]no alive runs[/yellow]")
        # Helpful: show the most recent finished/crashed run so the user
        # knows where to look next.
        runs_sorted = sorted(runs, key=lambda r: r.run_id, reverse=True)
        most_recent = runs_sorted[0]
        console.print(
            f"  (most recent: [bold]{most_recent.run_id}[/bold] "
            f"status=[bold]{most_recent.read_status()}[/bold])"
        )
        console.print(f"  watch it explicitly: [cyan]mlfactory watch {most_recent.run_id}[/cyan]")
        raise typer.Exit(code=1)

    alive_runs.sort(key=lambda r: r.run_id, reverse=True)
    if len(alive_runs) > 1:
        console.print(f"[dim]{len(alive_runs)} alive runs:[/dim]")
        for r in alive_runs:
            console.print(f"  [cyan]{r.run_id}[/cyan]  [dim]({r.game})[/dim]")
        console.print(f"[dim]attaching to newest: {alive_runs[0].run_id}[/dim]\n")

    watch_run(alive_runs[0], refresh_hz=refresh)


@app.command("stop")
def stop_cmd(
    run_id: str = typer.Argument(..., help="Run id (or trailing slug)."),
    game: str = typer.Option("", help="Disambiguate by game."),
    timeout: float = typer.Option(30.0, help="Graceful SIGTERM timeout before SIGKILL."),
) -> None:
    """Request a graceful stop; escalate to SIGKILL after timeout."""
    layout = _resolve_run(run_id, game=game or None)
    result = stop_run(layout, timeout=timeout)
    color = {"stopped": "green", "killed": "yellow", "not_running": "dim"}.get(result, "white")
    console.print(f"[{color}]{result}[/{color}] {layout.run_id}")


@app.command("evaluate")
def evaluate_cmd(
    checkpoint: Path = typer.Argument(..., help="Path to a saved .pt checkpoint."),
    game: str = typer.Option("boop", help="Game name (boop, connect4)."),
    opponents: str = typer.Option(
        "mcts:200,mcts:800,mcts:1600",
        help="Comma-separated opponents, e.g. 'mcts:800,random,mcts:1600'.",
    ),
    n_games: int = typer.Option(40, help="Games per match (colour-balanced)."),
    az_sims: int = typer.Option(100, help="PUCT sims per move for the checkpoint net."),
    n_workers: int = typer.Option(12, help="Parallel worker processes."),
    seed: int = typer.Option(0, help="RNG seed base."),
) -> None:
    """Evaluate a saved AlphaZero checkpoint against one or more opponents.

    Runs colour-balanced matches in parallel workers and prints a summary
    table. Useful for post-hoc evaluation at higher game counts than the
    trainer's per-iter eval (which is only 10 games).
    """
    from mlfactory.agents.alphazero.net import AlphaZeroNet
    from mlfactory.training.parallel import (
        EvalJob,
        parallel_eval,
        serialise_net,
    )

    if not checkpoint.exists():
        raise typer.BadParameter(f"checkpoint not found: {checkpoint}")

    # Parse opponents
    specs: list[tuple[str, int | None]] = []
    for raw in opponents.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if raw == "random":
            specs.append(("random", None))
        elif raw.startswith("mcts:"):
            try:
                sims = int(raw.split(":", 1)[1])
            except ValueError:
                raise typer.BadParameter(f"bad mcts spec: {raw}")
            specs.append(("mcts", sims))
        else:
            raise typer.BadParameter(f"unknown opponent '{raw}'. Use 'random' or 'mcts:<N>'")

    console.print(f"[bold]Evaluating[/bold] {checkpoint.name}")
    console.print(f"  game: {game}  az_sims: {az_sims}  n_games/match: {n_games}")

    net, extra = AlphaZeroNet.load(checkpoint)
    net = net.cpu().eval()
    console.print(f"  net: {net.param_count():,} params; extra={extra}")

    az_bytes, az_cfg = serialise_net(net)

    results_tbl = Table(title=f"{checkpoint.name} vs ... ({n_games} games each)")
    results_tbl.add_column("opponent")
    results_tbl.add_column("wins", justify="right")
    results_tbl.add_column("losses", justify="right")
    results_tbl.add_column("draws", justify="right")
    results_tbl.add_column("score", justify="right")
    results_tbl.add_column("wall", justify="right")

    for opp_kind, opp_sims in specs:
        tag = f"mcts{opp_sims}" if opp_kind == "mcts" else "random"
        jobs = [
            EvalJob(
                game_name=game,
                a_net_state_dict_bytes=az_bytes,
                a_net_config_dict=az_cfg,
                a_sims=az_sims,
                a_temperature_moves=4,
                a_name=f"az({checkpoint.stem})",
                a_seed=seed + i,
                opponent_kind=opp_kind,  # type: ignore[arg-type]
                b_sims=opp_sims,
                b_name=tag,
                b_seed=seed + 10_000 + i,
                a_is_player_0=(i % 2 == 0),
            )
            for i in range(n_games)
        ]
        t0 = time.monotonic()
        gr = parallel_eval(jobs, n_workers=n_workers)
        dt = time.monotonic() - t0
        wins = sum(1 for r in gr if r.a_won)
        losses = sum(1 for r in gr if r.b_won)
        draws = sum(1 for r in gr if r.drew)
        total = wins + losses + draws
        score = wins / max(total, 1)
        results_tbl.add_row(
            tag,
            str(wins),
            str(losses),
            str(draws),
            f"{score:.2f}",
            f"{dt:.0f}s",
        )
    console.print(results_tbl)


@app.command("analyze-game")
def analyze_game_cmd(
    game: str = typer.Argument(..., help="Game name. Currently supported: 'boop' | 'mandala'."),
    n_random_games: int = typer.Option(
        50, help="Random self-play games for branching/length stats."
    ),
    n_heuristic_games: int = typer.Option(
        30, help="Heuristic-vs-random games to measure heuristic strength."
    ),
) -> None:
    """Profile a game's structural dimensions and recommend an ML pipeline.

    Runs the analyzer in `mlfactory.analysis.game_classifier` against the
    game's probe (in `mlfactory.analysis.probes`) and prints a
    human-readable report. The recommendation is what the framework
    chooser will use to set up training (manual for now; automated path
    coming).
    """
    from mlfactory.analysis import classify, pretty_print
    from mlfactory.analysis.probes import BoopProbe, MandalaProbe

    probes = {"boop": BoopProbe, "mandala": MandalaProbe}
    if game not in probes:
        raise typer.BadParameter(f"unknown game '{game}'. Available: {sorted(probes)}")
    probe = probes[game]()
    profile = classify(
        probe,
        n_random_games=n_random_games,
        n_heuristic_games=n_heuristic_games,
    )
    console.print(pretty_print(profile))


@app.command("replay")
def replay_cmd(
    path: Path = typer.Argument(..., help="Path to a sample-game JSON file."),
    step: bool = typer.Option(False, help="Pause for Enter between moves."),
) -> None:
    """Render a saved sample game to the terminal."""
    replay_file(path, console=console, step=step)


if __name__ == "__main__":
    app()
