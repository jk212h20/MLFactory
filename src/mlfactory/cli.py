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


@app.command("replay")
def replay_cmd(
    path: Path = typer.Argument(..., help="Path to a sample-game JSON file."),
    step: bool = typer.Option(False, help="Pause for Enter between moves."),
) -> None:
    """Render a saved sample game to the terminal."""
    replay_file(path, console=console, step=step)


if __name__ == "__main__":
    app()
