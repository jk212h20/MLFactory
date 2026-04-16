"""Tests for sample-game serialisation."""

from __future__ import annotations

import json
from pathlib import Path

from mlfactory.games.boop import Boop
from mlfactory.training.sample_game import (
    GameRecord,
    MoveRecord,
    _state_from_dict,
    state_to_dict,
    write_game,
)


def test_state_to_dict_boop_roundtrip() -> None:
    env = Boop()
    s = env.initial_state()
    d = state_to_dict(s)
    # JSON-serialisable
    json.dumps(d)
    s2 = _state_from_dict("boop", d)
    assert s2 is not None
    # Initial states are equal.
    assert s2.board == s.board
    assert s2.orange_pool == s.orange_pool
    assert s2.gray_pool == s.gray_pool
    assert s2.to_play == s.to_play
    assert s2.phase == s.phase
    assert s2.winner == s.winner
    assert s2.is_terminal == s.is_terminal


def test_write_game_produces_json_and_txt(tmp_path: Path) -> None:
    env = Boop()
    s0 = env.initial_state()
    s1 = env.step(s0, 0)  # kitten at (0,0)

    record = GameRecord(
        game="boop",
        iter=1,
        kind="selfplay",
        agent_a="az-test",
        agent_b="az-test",
        seed=0,
        result="draw",
        winner=None,
        moves=[MoveRecord(ply=0, to_play=0, action=0, root_value=0.1)],
        states=[state_to_dict(s0), state_to_dict(s1)],
        notes={"n_simulations": 20},
    )

    json_path = tmp_path / "game.json"
    write_game(json_path, env=env, record=record)

    # Both files written
    assert json_path.exists()
    assert json_path.with_suffix(".txt").exists()

    # JSON is valid and contains expected keys
    data = json.loads(json_path.read_text())
    assert data["game"] == "boop"
    assert data["iter"] == 1
    assert data["kind"] == "selfplay"
    assert len(data["moves"]) == 1
    assert len(data["states"]) == 2

    # Text file contains the final-result line
    txt = json_path.with_suffix(".txt").read_text()
    assert "FINAL" in txt
    assert "move 1" in txt
