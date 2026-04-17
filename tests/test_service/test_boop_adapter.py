"""Tests for the Boop-TS JSON <-> BoopState adapter."""

from __future__ import annotations

from mlfactory.games.boop import Boop
from mlfactory.games.boop.rules import (
    EMPTY,
    G_CAT,
    G_KITTEN,
    N_PLACE_ACTIONS,
    O_CAT,
    O_KITTEN,
)
from mlfactory.service.boop_adapter import (
    action_to_wire,
    parse_boop_state,
)


def _empty_ts_state() -> dict:
    """Match the shape of BoopGame.getState() at the start of a game."""
    return {
        "board": [[None] * 6 for _ in range(6)],
        "players": {
            "orange": {
                "color": "orange",
                "kittensInPool": 8,
                "catsInPool": 0,
                "kittensRetired": 0,
                "socketId": "x",
                "name": "P",
                "connected": True,
            },
            "gray": {
                "color": "gray",
                "kittensInPool": 8,
                "catsInPool": 0,
                "kittensRetired": 0,
                "socketId": "y",
                "name": "Q",
                "connected": True,
            },
        },
        "currentTurn": "orange",
        "phase": "playing",
        "winner": None,
        "lastMove": None,
        "boopedPieces": [],
        "graduatedPieces": [],
    }


def test_parse_initial_state_matches_boop_initial() -> None:
    ts = _empty_ts_state()
    parsed = parse_boop_state(ts)
    initial = Boop().initial_state()
    assert parsed.board == initial.board
    assert parsed.orange_pool == initial.orange_pool
    assert parsed.gray_pool == initial.gray_pool
    assert parsed.to_play == initial.to_play
    assert parsed.phase == initial.phase
    assert parsed.is_terminal == initial.is_terminal


def test_parse_piece_colours_and_types() -> None:
    ts = _empty_ts_state()
    ts["board"][0][0] = {"color": "orange", "type": "kitten"}
    ts["board"][0][1] = {"color": "orange", "type": "cat"}
    ts["board"][1][0] = {"color": "gray", "type": "kitten"}
    ts["board"][1][1] = {"color": "gray", "type": "cat"}
    parsed = parse_boop_state(ts)
    assert parsed.board[0] == O_KITTEN
    assert parsed.board[1] == O_CAT
    assert parsed.board[6] == G_KITTEN
    assert parsed.board[7] == G_CAT


def test_parse_gray_turn() -> None:
    ts = _empty_ts_state()
    ts["currentTurn"] = "gray"
    parsed = parse_boop_state(ts)
    assert parsed.to_play == 1


def test_parse_graduation_phase() -> None:
    ts = _empty_ts_state()
    ts["phase"] = "selecting_graduation"
    ts["pendingGraduationPlayer"] = "orange"
    ts["pendingGraduationOptions"] = [
        [{"row": 0, "col": 0}, {"row": 0, "col": 1}, {"row": 0, "col": 2}],
        [{"row": 1, "col": 0}, {"row": 1, "col": 1}, {"row": 1, "col": 2}],
    ]
    parsed = parse_boop_state(ts)
    assert parsed.phase == "selecting_graduation"
    assert len(parsed.pending_options) == 2
    # First option: cells (0,0)=0, (0,1)=1, (0,2)=2
    assert parsed.pending_options[0] == (0, 1, 2)
    # Second option: cells (1,0)=6, (1,1)=7, (1,2)=8
    assert parsed.pending_options[1] == (6, 7, 8)


def test_parse_winner_finished() -> None:
    ts = _empty_ts_state()
    ts["phase"] = "finished"
    ts["winner"] = "gray"
    parsed = parse_boop_state(ts)
    assert parsed.winner == 1
    assert parsed.is_terminal is True


def test_parse_missing_players_tolerated() -> None:
    """Defensive parsing: missing players object doesn't crash."""
    ts = _empty_ts_state()
    ts["players"] = {"orange": None, "gray": None}
    parsed = parse_boop_state(ts)
    # Pools default to zeros
    assert parsed.orange_pool == (0, 0, 0)
    assert parsed.gray_pool == (0, 0, 0)


def test_action_to_wire_kitten_placement() -> None:
    # Action 0 = kitten at cell 0 = (0,0)
    wire = action_to_wire(0)
    assert wire == {"kind": "place", "row": 0, "col": 0, "pieceType": "kitten"}


def test_action_to_wire_kitten_last_cell() -> None:
    # Action 35 = kitten at cell 35 = (5,5)
    wire = action_to_wire(35)
    assert wire == {"kind": "place", "row": 5, "col": 5, "pieceType": "kitten"}


def test_action_to_wire_cat_placement() -> None:
    # Action 36 = cat at cell 0 = (0,0)
    wire = action_to_wire(36)
    assert wire == {"kind": "place", "row": 0, "col": 0, "pieceType": "cat"}


def test_action_to_wire_graduation_option() -> None:
    wire = action_to_wire(N_PLACE_ACTIONS)
    assert wire == {"kind": "graduation", "optionIndex": 0}
    wire = action_to_wire(N_PLACE_ACTIONS + 5)
    assert wire == {"kind": "graduation", "optionIndex": 5}


def test_roundtrip_initial_state_action_legal() -> None:
    """Parse initial state, get legal actions, round-trip one."""
    env = Boop()
    parsed = parse_boop_state(_empty_ts_state())
    legal = env.legal_actions(parsed)
    assert len(legal) > 0
    # Every legal action at initial state is a placement (no graduation)
    for a in legal:
        wire = action_to_wire(a)
        assert wire["kind"] == "place"
