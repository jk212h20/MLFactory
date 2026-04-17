"""Convert between Boop-TS JSON game state and Python BoopState.

The Boop TS server's GameState.getState() (server/src/game/GameState.ts:625)
produces JSON of this shape:

    {
      "board":   <6x6 array of {color, type} | null>,
      "players": {
        "orange": {kittensInPool, catsInPool, kittensRetired, ...} | null,
        "gray":   {...}
      },
      "currentTurn":  "orange" | "gray",
      "phase":        "waiting" | "playing" | "selecting_graduation" | "finished",
      "winner":       "orange" | "gray" | null,
      "lastMove":     {...} | null,
      "boopedPieces": [...],
      "graduatedPieces": [...],
      "pendingGraduationOptions"?: Cell[][],
      "pendingGraduationPlayer"?: "orange" | "gray"
    }

We convert this into our internal `BoopState` (an immutable dataclass with
integer-coded board). Actions are returned in the TS wire format.
"""

from __future__ import annotations

from typing import Any

from mlfactory.games.boop.rules import (
    EMPTY,
    G_CAT,
    G_KITTEN,
    N_CELLS,
    N_PLACE_ACTIONS,
    O_CAT,
    O_KITTEN,
    BoopState,
)


# -- JSON -> BoopState ------------------------------------------------------


def _piece_to_int(piece_json: dict | None) -> int:
    """Map `{color, type}` piece to our single-byte encoding."""
    if piece_json is None:
        return EMPTY
    color = piece_json.get("color")
    ptype = piece_json.get("type")
    if color == "orange":
        return O_KITTEN if ptype == "kitten" else O_CAT
    if color == "gray":
        return G_KITTEN if ptype == "kitten" else G_CAT
    raise ValueError(f"unknown piece: {piece_json!r}")


def _cell_list_to_indices(cells: list[dict]) -> tuple[int, int, int]:
    """Convert list of `{row, col}` cells (used in pendingGraduationOptions)
    into a triple of linearised indices.
    """
    indices = [c["row"] * 6 + c["col"] for c in cells]
    # Our BoopState.pending_options expects tuples of 3 indices.
    # If fewer than 3 are provided (malformed) we pad with -1 to match shape.
    while len(indices) < 3:
        indices.append(-1)
    return (indices[0], indices[1], indices[2])


def parse_boop_state(payload: dict[str, Any]) -> BoopState:
    """Parse a Boop-TS JSON state into our internal BoopState dataclass.

    Tolerant of missing keys on quiescent states; strict on core required
    fields. Returns a fully-populated BoopState that our agent and PUCT
    can operate on.
    """
    board_json = payload["board"]
    assert len(board_json) == 6 and len(board_json[0]) == 6, "board must be 6x6"

    flat_board: list[int] = []
    for row in board_json:
        for cell in row:
            flat_board.append(_piece_to_int(cell))
    assert len(flat_board) == N_CELLS

    players = payload.get("players", {}) or {}
    orange = players.get("orange") or {}
    gray = players.get("gray") or {}

    orange_pool = (
        int(orange.get("kittensInPool", 0)),
        int(orange.get("catsInPool", 0)),
        int(orange.get("kittensRetired", 0)),
    )
    gray_pool = (
        int(gray.get("kittensInPool", 0)),
        int(gray.get("catsInPool", 0)),
        int(gray.get("kittensRetired", 0)),
    )

    current_turn = payload.get("currentTurn", "orange")
    to_play = 0 if current_turn == "orange" else 1

    phase = payload.get("phase", "playing")
    winner_str = payload.get("winner")
    winner = None
    if winner_str == "orange":
        winner = 0
    elif winner_str == "gray":
        winner = 1

    pending_options_json = payload.get("pendingGraduationOptions") or []
    pending_options = tuple(_cell_list_to_indices(opt) for opt in pending_options_json)

    # Move counter isn't sent by the server; it's used only by our random
    # seeding and doesn't affect legality or value. Default to 0.
    move_number = 0

    is_terminal = phase == "finished"

    return BoopState(
        board=tuple(flat_board),
        orange_pool=orange_pool,
        gray_pool=gray_pool,
        to_play=to_play,
        phase=phase,
        winner=winner,
        move_number=move_number,
        pending_options=pending_options,
        _is_terminal=is_terminal,
    )


# -- Action -> TS wire format ----------------------------------------------


def action_to_wire(action: int) -> dict[str, Any]:
    """Encode our integer action as a Boop-TS move payload.

    Returns either:
      {"kind": "place",       "row": r, "col": c, "pieceType": "kitten"|"cat"}
      {"kind": "graduation",  "optionIndex": i}
    """
    if 0 <= action < N_PLACE_ACTIONS:
        piece_kind = action // N_CELLS  # 0 = kitten, 1 = cat
        cell_idx = action % N_CELLS
        row, col = divmod(cell_idx, 6)
        return {
            "kind": "place",
            "row": row,
            "col": col,
            "pieceType": "kitten" if piece_kind == 0 else "cat",
        }
    # Graduation option
    option_index = action - N_PLACE_ACTIONS
    return {"kind": "graduation", "optionIndex": option_index}
