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
from mlfactory.games.boop.rules import BoopState
from mlfactory.games.connect4 import Connect4


_BOARD_ENV = {
    "boop": Boop,
    "connect4": Connect4,
}


def replay_file(path: Path, *, console: Console | None = None, step: bool = False) -> None:
    """Render a saved sample-game JSON file to the console."""
    console = console or Console()
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text())
    console.print(f"[bold]Replay[/bold]: {path}")
    game = data.get("game", "?")
    console.print(f"game={game}  iter={data.get('iter', '?')}  kind={data.get('kind', '?')}")
    if "result" in data:
        console.print(f"result={data['result']}")
    moves = data.get("moves", [])
    console.print(f"moves ({len(moves)}):")

    # If we can identify the game and states were saved, render them.
    if game == "boop" and "states" in data:
        _replay_boop(data, console, step=step)
    elif game == "connect4" and "states" in data:
        _replay_connect4(data, console, step=step)
    else:
        # Generic fallback — just print the moves.
        console.print(json.dumps(moves, indent=2))


def _replay_boop(data: dict, console: Console, *, step: bool) -> None:
    states = data["states"]
    moves = data.get("moves", [])
    env = Boop()
    for i, raw in enumerate(states):
        state = BoopState.from_dict(raw) if hasattr(BoopState, "from_dict") else None
        console.rule(f"move {i} / {len(states) - 1}")
        if state is not None:
            console.print(_render_boop_board(state))
        else:
            console.print(str(raw)[:200])
        if i < len(moves):
            console.print(f"next move: {moves[i]}")
        if step:
            try:
                input("(enter to continue, ctrl-c to quit) ")
            except KeyboardInterrupt:
                return


def _render_boop_board(state: "BoopState") -> str:
    from mlfactory.games.boop.rules import G_CAT, G_KITTEN, O_CAT, O_KITTEN

    glyph = {0: ".", O_KITTEN: "o", O_CAT: "O", G_KITTEN: "g", G_CAT: "G"}
    rows = []
    for r in range(6):
        cells = " ".join(glyph.get(state.board[r * 6 + c], "?") for c in range(6))
        rows.append(f"{r}  {cells}")
    header = "   0 1 2 3 4 5"
    pool = (
        f"orange pool: kittens={state.orange_pool[0]} cats={state.orange_pool[1]} "
        f"retired={state.orange_pool[2]}\n"
        f"gray pool:   kittens={state.gray_pool[0]} cats={state.gray_pool[1]} "
        f"retired={state.gray_pool[2]}\n"
        f"to_play={'orange' if state.to_play == 0 else 'gray'}  phase={state.phase}  "
        f"winner={state.winner}"
    )
    return header + "\n" + "\n".join(rows) + "\n" + pool


def _replay_connect4(data: dict, console: Console, *, step: bool) -> None:  # noqa: ARG001
    # Placeholder — will flesh out when connect4 self-play samples are added.
    console.print("(connect4 replay not yet implemented; raw moves below)")
    console.print(json.dumps(data.get("moves", []), indent=2))
