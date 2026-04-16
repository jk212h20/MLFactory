"""Tensor encoding for Boop states, ready for the Phase-3 AlphaZero network.

We emit a (C, 6, 6) float tensor with planes:

   0: mover's kittens            (spatial, 0/1)
   1: mover's cats               (spatial, 0/1)
   2: opponent's kittens         (spatial, 0/1)
   3: opponent's cats            (spatial, 0/1)
   4: legal placement locations  (spatial, 0/1) — any cell where the mover can place
   5: mover pool: kittens / 8    (constant plane)
   6: mover pool: cats / 8       (constant plane)
   7: opp pool: kittens / 8      (constant plane)
   8: opp pool: cats / 8         (constant plane)
   9: mover cats retired / 8     (constant plane; approximate "progress")
  10: opp cats retired / 8       (constant plane)

Design notes:
- All planes are from the **mover's perspective** so the network never has to
  learn the orange/gray labels — cuts effective state space in half.
- Constant planes are cheap; the net learns to ignore what it doesn't need.
- No history planes in v1. Boop is short and Markovian enough that we expect
  the current state to suffice; adding 1-2 history planes is an easy ablation
  in Phase 3.
- Legal-action mask is a separate vector, produced by `legal_mask()`, with
  shape (104,) bool: True where the action is legal. The policy head's output
  will be softmaxed over this mask.

This file is Phase-2 infrastructure; no network uses it yet.
"""

from __future__ import annotations

import numpy as np

from mlfactory.games.boop.rules import (
    BOARD_SIZE,
    EMPTY,
    G_CAT,
    G_KITTEN,
    N_ACTIONS,
    N_CELLS,
    O_CAT,
    O_KITTEN,
    Boop,
    BoopState,
)

N_PLANES = 11


def encode_state(state: BoopState, env: Boop | None = None) -> np.ndarray:
    """Encode a BoopState into a (N_PLANES, 6, 6) float32 array."""
    if env is None:
        env = Boop()
    planes = np.zeros((N_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    mover = state.to_play
    # Mover is orange (0) -> their kittens are O_KITTEN (1), cats O_CAT (2).
    # Opponent is the other color.
    if mover == 0:
        mover_kitten, mover_cat = O_KITTEN, O_CAT
        opp_kitten, opp_cat = G_KITTEN, G_CAT
        mover_pool = state.orange_pool
        opp_pool = state.gray_pool
    else:
        mover_kitten, mover_cat = G_KITTEN, G_CAT
        opp_kitten, opp_cat = O_KITTEN, O_CAT
        mover_pool = state.gray_pool
        opp_pool = state.orange_pool

    for idx, piece in enumerate(state.board):
        r, c = divmod(idx, BOARD_SIZE)
        if piece == mover_kitten:
            planes[0, r, c] = 1.0
        elif piece == mover_cat:
            planes[1, r, c] = 1.0
        elif piece == opp_kitten:
            planes[2, r, c] = 1.0
        elif piece == opp_cat:
            planes[3, r, c] = 1.0

    # Legal placement locations (any empty cell where mover has at least one piece
    # in pool). In selecting_graduation phase, this is all zeros.
    if state.phase == "playing":
        has_kitten = mover_pool[0] > 0
        has_cat = mover_pool[1] > 0
        if has_kitten or has_cat:
            for idx, piece in enumerate(state.board):
                if piece == EMPTY:
                    r, c = divmod(idx, BOARD_SIZE)
                    planes[4, r, c] = 1.0

    # Constant (pool) planes, normalized by starting kitten count (8).
    planes[5, :, :] = mover_pool[0] / 8.0
    planes[6, :, :] = mover_pool[1] / 8.0
    planes[7, :, :] = opp_pool[0] / 8.0
    planes[8, :, :] = opp_pool[1] / 8.0
    planes[9, :, :] = mover_pool[2] / 8.0
    planes[10, :, :] = opp_pool[2] / 8.0

    return planes


def legal_mask(state: BoopState, env: Boop | None = None) -> np.ndarray:
    """Return a (N_ACTIONS,) bool array where True = legal action."""
    if env is None:
        env = Boop()
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for a in env.legal_actions(state):
        mask[a] = True
    return mask


__all__ = ["encode_state", "legal_mask", "N_PLANES"]
