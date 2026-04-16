"""Replay a saved sample game in the terminal.

For phase 3a (runner validation) this is a generic JSON pretty-printer.
When the real trainer lands in 3c, we'll add game-specific board rendering
by dispatching on the game name stored in the sample.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from mlfactory.games.boop import Boop


def replay_file(path: Path, *, console: Console | None = None, step: bool = False) -> None:
    """Render a saved sample-game JSON file to the console."""
    console = console or Console()
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text())
    console.print(f"[bold]Replay[/bold]: {path}")
    game = data.get("game", "?")
    console.print(
        f"game={game}  iter={data.get('iter', '?')}  kind={data.get('kind', '?')}  "
        f"result={data.get('result', '?')}  winner={data.get('winner', '?')}"
    )
    console.print(f"agents: A={data.get('agent_a', '?')}  B={data.get('agent_b', '?')}")
    moves = data.get("moves", [])
    states = data.get("states", [])
    console.print(f"moves ({len(moves)}), states ({len(states)}):")

    if game == "boop" and states:
        _replay_boop(data, console, step=step)
    else:
        # Generic fallback — just print the moves.
        console.print(json.dumps(moves, indent=2))


def _replay_boop(data: dict, console: Console, *, step: bool) -> None:
    from mlfactory.training.sample_game import _state_from_dict

    env = Boop()
    states = data["states"]
    moves = data.get("moves", [])
    for i, raw in enumerate(states):
        state = _state_from_dict("boop", raw)
        console.rule(f"state {i} / {len(states) - 1}")
        if state is not None:
            console.print(env.render(state))
        else:
            console.print(str(raw)[:200])
        if i < len(moves):
            m = moves[i]
            console.print(
                f"[dim]next: player {m.get('to_play')} plays action {m.get('action')}"
                + (f"  root_v={m['root_value']:+.2f}" if m.get("root_value") is not None else "")
                + "[/dim]"
            )
        if step:
            try:
                input("(enter to continue, ctrl-c to quit) ")
            except KeyboardInterrupt:
                return
