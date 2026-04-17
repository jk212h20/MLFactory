"""Mandala game engine (ported from mandala-web's game.js).

Reference implementation: ~/ActiveProjects/mandala-web/game.js (715 lines,
MIT-licensed, commit 1d84c5b as of 2026-04-17).

Port rules:
- Public API matches the JS module's exports (createGame, performAction,
  getValidActions, getPlayerView, getWinner, calculateScore).
- Card identity is preserved (every card has an id string), but only
  color is semantically meaningful for game legality. This makes
  parity-testing against the JS engine trivial (id strings match).
- All mutation goes through copy.deepcopy (== structuredClone in JS).
- Reshuffle uses an injectable random.Random so JS↔Python parity tests
  can line up the RNG streams.

See tests/test_mandala/test_parity.py for the 10k-game parity harness.
"""

from mlfactory.games.mandala.rules import (
    COLORS,
    CARDS_PER_COLOR,
    INITIAL_CUP_SIZE,
    INITIAL_HAND_SIZE,
    INITIAL_MOUNTAIN_SIZE,
    MAX_HAND_SIZE,
    RIVER_SIZE,
    calculate_score,
    create_deck,
    create_game,
    get_player_view,
    get_valid_actions,
    get_winner,
    perform_action,
)

__all__ = [
    "COLORS",
    "CARDS_PER_COLOR",
    "INITIAL_CUP_SIZE",
    "INITIAL_HAND_SIZE",
    "INITIAL_MOUNTAIN_SIZE",
    "MAX_HAND_SIZE",
    "RIVER_SIZE",
    "calculate_score",
    "create_deck",
    "create_game",
    "get_player_view",
    "get_valid_actions",
    "get_winner",
    "perform_action",
]
