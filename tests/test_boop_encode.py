"""Quick tests for the Boop tensor encoding.

We check:
- Shapes are correct.
- Mover-relative perspective flips correctly between players.
- Legal-mask matches legal_actions.
- Pool planes are constant per channel.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlfactory.games.boop.encode import N_PLANES, encode_state, legal_mask
from mlfactory.games.boop.rules import BOARD_SIZE, N_ACTIONS, Boop


@pytest.fixture
def env() -> Boop:
    return Boop()


def test_encode_shape(env: Boop) -> None:
    planes = encode_state(env.initial_state(), env)
    assert planes.shape == (N_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert planes.dtype == np.float32


def test_initial_state_has_empty_piece_planes(env: Boop) -> None:
    planes = encode_state(env.initial_state(), env)
    # Piece-location planes (0..3) should be all zero.
    assert np.all(planes[0:4] == 0.0)
    # Legal placement plane should be all 1s (36 empty cells, kittens available).
    assert np.all(planes[4] == 1.0)
    # Pool planes: mover kittens 8/8 = 1.0, cats 0/8 = 0.0.
    assert np.all(planes[5] == 1.0)
    assert np.all(planes[6] == 0.0)
    assert np.all(planes[7] == 1.0)
    assert np.all(planes[8] == 0.0)
    assert np.all(planes[9] == 0.0)
    assert np.all(planes[10] == 0.0)


def test_mover_perspective_swaps(env: Boop) -> None:
    """After a move, the mover flips; the same piece should appear in a
    different plane depending on who's to move next.
    """
    s = env.initial_state()
    # Orange plays a kitten at (3, 3). Action: kind=0 * 36 + 3*6 + 3 = 21.
    s2 = env.step(s, 21)
    # Now gray is to_play. The orange kitten at (3,3) should appear in plane 2
    # (opponent's kittens) from gray's perspective.
    planes = encode_state(s2, env)
    # Mover planes (0,1) should be zero except for any pieces gray owns (none).
    assert planes[0].sum() == 0.0
    assert planes[1].sum() == 0.0
    # Opponent plane 2 (opponent's kittens) should have exactly 1.0 at (3, 3).
    assert planes[2, 3, 3] == 1.0
    assert planes[2].sum() == 1.0


def test_legal_mask_matches_legal_actions(env: Boop) -> None:
    s = env.initial_state()
    mask = legal_mask(s, env)
    legal = set(env.legal_actions(s))
    for a in range(N_ACTIONS):
        assert mask[a] == (a in legal)


def test_pool_planes_are_constant(env: Boop) -> None:
    """Constant planes must have the same value in every cell."""
    s = env.initial_state()
    # Advance a few moves.
    import random

    rng = random.Random(0)
    for _ in range(5):
        a = rng.choice(env.legal_actions(s))
        s = env.step(s, a)
    planes = encode_state(s, env)
    for i in (5, 6, 7, 8, 9, 10):
        assert planes[i].std() < 1e-9, f"plane {i} not constant"
