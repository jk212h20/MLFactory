"""D4 dihedral symmetry group for the 6x6 Boop board.

The D4 group has 8 elements: 4 rotations (0°, 90°, 180°, 270°) and 4 reflections
(horizontal, vertical, main diagonal, anti-diagonal). Boop's rules are invariant
under D4 — the game played on a rotated/reflected board is the same game, with
positions and actions mapped accordingly. This gives free 8× data augmentation
for training neural networks.

Two things must transform in lockstep:
- The **board** (6x6 grid of piece IDs).
- The **action** (place at row/col with a piece type). The piece type is
  symmetry-invariant; only the spatial position transforms.

Graduation-choice actions (>= 72) are harder: the option indices refer to
specific trios whose cells also transform. But we never augment in the
graduation-choice state (it's transient), so we simply refuse to symmetrize
those — callers should only apply symmetry in `playing` phase.

One caveat — the **stranded-graduation fallback** (TS `BoopGame.checkAndExecuteGraduation`
branch when `piecesOnBoard >= 8 && no 3-in-a-row`) picks the *first kitten in
row-major order*. That tie-breaking rule is NOT symmetry-invariant: after a
rotation, the "first kitten" is a different kitten. Training augmentation must
detect transitions that involved a stranded fallback and either skip them or
log them separately. See `was_stranded_fallback()` below.

Exported API:
- `SYMMETRIES`: list of 8 Symmetry objects.
- `Symmetry.apply_state(state)`: returns a new BoopState with the board rotated/reflected.
- `Symmetry.apply_action(action)`: maps a placement action to the transformed action.
- `Symmetry.apply_cell(row, col)`: maps a single cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mlfactory.core.env import Action
from mlfactory.games.boop.rules import (
    BOARD_SIZE,
    N_CELLS,
    N_PLACE_ACTIONS,
    BoopState,
)


def _rot0(r: int, c: int) -> tuple[int, int]:
    return r, c


def _rot90(r: int, c: int) -> tuple[int, int]:
    # 90° counter-clockwise: (r, c) -> (N-1-c, r)
    return BOARD_SIZE - 1 - c, r


def _rot180(r: int, c: int) -> tuple[int, int]:
    return BOARD_SIZE - 1 - r, BOARD_SIZE - 1 - c


def _rot270(r: int, c: int) -> tuple[int, int]:
    return c, BOARD_SIZE - 1 - r


def _flip_h(r: int, c: int) -> tuple[int, int]:
    # Reflect across horizontal midline (swap rows)
    return BOARD_SIZE - 1 - r, c


def _flip_v(r: int, c: int) -> tuple[int, int]:
    # Reflect across vertical midline (swap cols)
    return r, BOARD_SIZE - 1 - c


def _flip_diag(r: int, c: int) -> tuple[int, int]:
    # Reflect across main diagonal (swap r, c)
    return c, r


def _flip_anti(r: int, c: int) -> tuple[int, int]:
    # Reflect across anti-diagonal
    return BOARD_SIZE - 1 - c, BOARD_SIZE - 1 - r


@dataclass(frozen=True)
class Symmetry:
    """One element of D4. Maps (row, col) to (row', col')."""

    name: str
    fn: Callable[[int, int], tuple[int, int]]

    def apply_cell(self, row: int, col: int) -> tuple[int, int]:
        return self.fn(row, col)

    def apply_cell_index(self, idx: int) -> int:
        r, c = divmod(idx, BOARD_SIZE)
        rr, cc = self.fn(r, c)
        return rr * BOARD_SIZE + cc

    def apply_action(self, action: Action) -> Action:
        """Map a PLACEMENT action to its symmetric counterpart.

        Graduation-choice actions (action >= 72) are returned unchanged because
        their option-index semantics do not transform meaningfully under
        board symmetry — the option list would need to be recomputed on the
        transformed state. Callers should only apply symmetry in `playing` phase.
        """
        if action >= N_PLACE_ACTIONS:
            return action  # see docstring
        kind = action // N_CELLS
        cell_idx = action % N_CELLS
        new_cell = self.apply_cell_index(cell_idx)
        return kind * N_CELLS + new_cell

    def apply_state(self, state: BoopState) -> BoopState:
        """Return a symmetry-transformed copy of `state`.

        Only transforms the spatial board. All non-spatial fields (pools,
        to_play, phase, winner, move_number, pending_options) are preserved
        untouched. `pending_options` is zeroed out; if you need to symmetrize a
        graduation-pending state, don't — use this only during `playing`.
        """
        new_board = [0] * N_CELLS
        for idx, piece in enumerate(state.board):
            new_idx = self.apply_cell_index(idx)
            new_board[new_idx] = piece
        return BoopState(
            board=tuple(new_board),
            orange_pool=state.orange_pool,
            gray_pool=state.gray_pool,
            to_play=state.to_play,
            phase=state.phase,
            winner=state.winner,
            move_number=state.move_number,
            pending_options=(),  # dropped
            _is_terminal=state._is_terminal,
        )


SYMMETRIES: list[Symmetry] = [
    Symmetry("identity", _rot0),
    Symmetry("rot90", _rot90),
    Symmetry("rot180", _rot180),
    Symmetry("rot270", _rot270),
    Symmetry("flip_h", _flip_h),
    Symmetry("flip_v", _flip_v),
    Symmetry("flip_diag", _flip_diag),
    Symmetry("flip_anti", _flip_anti),
]


assert len(SYMMETRIES) == 8


def was_stranded_fallback(before: BoopState, action: Action) -> bool:
    """Return True if `action` in `before` triggered the stranded-graduation
    fallback (which breaks symmetry; see the `Boop.would_trigger_stranded_fallback`
    docstring). This is just a delegating wrapper for symmetry test/augmentation
    callers who don't want to carry an env reference.
    """
    # Late import to avoid a circular dep at module load.
    from mlfactory.games.boop.rules import Boop

    return Boop().would_trigger_stranded_fallback(before, action)
