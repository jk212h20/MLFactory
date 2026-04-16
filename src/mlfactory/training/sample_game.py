"""Serialise a played game to disk for later replay/inspection.

Every saved game includes:
- the sequence of actions taken,
- the full state after each move (so `mlfactory replay` can render boards
  without re-simulating from scratch),
- MCTS visit counts + Q values + chosen action per move (when available),
- the final result.

Two files are written alongside each other:
- <name>.json  : machine-readable full record (used by `mlfactory replay`)
- <name>.txt   : human-readable ASCII board replay (used when you just
                 want to read through a game without launching the TUI)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mlfactory.core.env import Env


@dataclass
class MoveRecord:
    """One move in a saved game."""

    ply: int
    to_play: int
    action: int
    # Optional MCTS introspection (only when the mover was a search-based agent):
    visits: dict[int, int] | None = None  # action -> visit count at root
    q_values: dict[int, float] | None = None  # action -> Q at root
    root_value: float | None = None  # PUCT's value at root


@dataclass
class GameRecord:
    """Full saved game."""

    game: str  # "boop" | "connect4"
    iter: int | None  # training iteration that produced this game
    kind: str  # "selfplay" | "eval" | "champion"
    agent_a: str
    agent_b: str  # for self-play both are the same name; differentiated here as a label
    seed: int | None
    result: str  # "a_win" | "b_win" | "draw"
    winner: int | None  # 0, 1, or None
    moves: list[MoveRecord]
    states: list[dict]  # serialised state after each move, inclusive of initial
    notes: dict  # any extra metadata (config summary, etc.)


def state_to_dict(state) -> dict:
    """Serialise a state to a json-safe dict. Dispatches on state shape."""
    if hasattr(state, "board") and hasattr(state, "orange_pool"):
        # Boop
        return {
            "kind": "boop",
            "board": list(state.board),
            "orange_pool": list(state.orange_pool),
            "gray_pool": list(state.gray_pool),
            "to_play": state.to_play,
            "phase": getattr(state, "phase", "playing"),
            "pending_options": [list(o) for o in getattr(state, "pending_options", ())],
            "winner": state.winner,
            "move_number": getattr(state, "move_number", 0),
            "is_terminal": state.is_terminal,
        }
    # Generic fallback
    return {
        "kind": "unknown",
        "to_play": state.to_play,
        "winner": state.winner,
        "is_terminal": state.is_terminal,
        "repr": repr(state),
    }


def write_game(
    path: Path,
    *,
    env: Env,
    record: GameRecord,
) -> None:
    """Write the game to `path` (json) and `path.with_suffix('.txt')` (human)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "game": record.game,
        "iter": record.iter,
        "kind": record.kind,
        "agent_a": record.agent_a,
        "agent_b": record.agent_b,
        "seed": record.seed,
        "result": record.result,
        "winner": record.winner,
        "moves": [asdict(m) for m in record.moves],
        "states": record.states,
        "notes": record.notes,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    # Human readable
    txt_path = path.with_suffix(".txt")
    lines: list[str] = []
    lines.append(
        f"game={record.game}  iter={record.iter}  kind={record.kind}  "
        f"result={record.result}  winner={record.winner}"
    )
    lines.append(f"agents: A={record.agent_a}  B={record.agent_b}")
    if record.notes:
        lines.append(f"notes: {record.notes}")
    lines.append("")
    # Render initial state
    try:
        initial_state = _state_from_dict(record.game, record.states[0])
    except Exception:  # noqa: BLE001
        initial_state = None
    if initial_state is not None:
        lines.append("-- initial --")
        lines.append(env.render(initial_state))
        lines.append("")
    for i, move in enumerate(record.moves):
        lines.append(
            f"move {i + 1}: player {move.to_play} plays action {move.action}"
            + (f"  root_value={move.root_value:+.2f}" if move.root_value is not None else "")
        )
        # Show the state AFTER this move (index i+1 in states list)
        if i + 1 < len(record.states):
            try:
                s = _state_from_dict(record.game, record.states[i + 1])
                if s is not None:
                    lines.append(env.render(s))
                    lines.append("")
            except Exception:  # noqa: BLE001
                pass
    lines.append(f"FINAL: winner={record.winner}  result={record.result}")
    txt_path.write_text("\n".join(lines) + "\n")


def _state_from_dict(game: str, d: dict):
    """Reconstruct a state object for rendering. Returns None if we can't."""
    if game == "boop":
        from mlfactory.games.boop.rules import BoopState

        return BoopState(
            board=tuple(d["board"]),
            orange_pool=tuple(d["orange_pool"]),
            gray_pool=tuple(d["gray_pool"]),
            to_play=d["to_play"],
            phase=d.get("phase", "playing"),
            winner=d.get("winner"),
            move_number=d.get("move_number", 0),
            pending_options=tuple(tuple(o) for o in d.get("pending_options", [])),
            _is_terminal=d.get("is_terminal", False),
        )
    return None
