"""Boop game implementation for MLFactory.

Boop is a 2-player, perfect-information, deterministic board game by Smirk & Dagger.
We port the rules from `Boop/server/src/game/GameState.ts` and cross-verify for parity.
"""

from mlfactory.games.boop.rules import (
    BOARD_SIZE,
    MAX_CATS,
    STARTING_KITTENS,
    Boop,
    BoopState,
)

__all__ = ["Boop", "BoopState", "BOARD_SIZE", "STARTING_KITTENS", "MAX_CATS"]
