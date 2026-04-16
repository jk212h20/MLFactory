"""MLFactory CLI."""

from __future__ import annotations

import sys
import time

import typer
from rich.console import Console
from rich.table import Table

from mlfactory import __version__
from mlfactory.agents.base import Agent
from mlfactory.agents.mcts import MCTSAgent
from mlfactory.agents.random_agent import RandomAgent
from mlfactory.core.env import Env
from mlfactory.games.connect4 import Connect4
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


if __name__ == "__main__":
    app()
