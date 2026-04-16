"""Connect 4 on the standard 7-wide x 6-tall board.

Implementation notes:
- **Bitboard representation** for speed. Each player has a 49-bit integer; bit `col*7 + row`
  is set when that player has a piece at (col, row), row counted from the bottom.
  The 7-bit column layout (with one sentinel row per column, so 42 cells span 49 bits)
  is the standard Connect 4 bitboard trick — win detection is 4 bitwise AND/shifts total.
- States are immutable `Position` dataclasses; `step()` returns a new one.
- Actions are integers 0..6, representing the column to drop in.
- `to_play` is 0 (first player) or 1; on an empty board `to_play = 0`.

Win check (shift-and-AND):
    The 49-bit layout has 7 columns × 7 bits (col*7 is bottom of column c).
    Horizontal 4: p & (p>>7) & (p>>14) & (p>>21)
    Vertical   4: p & (p>>1) & (p>>2)  & (p>>3)
    Diag  \\   4: p & (p>>6) & (p>>12) & (p>>18)   # up-left
    Diag  /    4: p & (p>>8) & (p>>16) & (p>>24)   # up-right
    If any is non-zero, the player has a 4-in-a-row.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlfactory.core.env import Action, Player

NUM_COLS = 7
NUM_ROWS = 6
# Bit layout: bit index = col * 7 + row. Row 0 = bottom.
# Column heights are tracked separately for O(1) drop.
BOTTOM_MASK = 0
for c in range(NUM_COLS):
    BOTTOM_MASK |= 1 << (c * 7)
BOARD_MASK = 0
for c in range(NUM_COLS):
    for r in range(NUM_ROWS):
        BOARD_MASK |= 1 << (c * 7 + r)


def _has_four(bits: int) -> bool:
    """Return True if the given 49-bit bitboard contains a 4-in-a-row."""
    # vertical
    m = bits & (bits >> 1)
    if m & (m >> 2):
        return True
    # horizontal
    m = bits & (bits >> 7)
    if m & (m >> 14):
        return True
    # diagonal up-right (/)
    m = bits & (bits >> 8)
    if m & (m >> 16):
        return True
    # diagonal up-left (\)
    m = bits & (bits >> 6)
    if m & (m >> 12):
        return True
    return False


@dataclass(frozen=True, slots=True)
class Connect4State:
    # Each player's pieces as a 49-bit int.
    p0: int
    p1: int
    # Heights[c] = row where next piece lands in column c (0..NUM_ROWS).
    heights: tuple[int, ...]
    # Whose turn (0 or 1).
    to_play: Player
    # Move number (for tie-break / draw detection). 0 on empty board.
    move_number: int
    # Cached terminal info.
    _winner: Player | None
    _is_terminal: bool

    @property
    def winner(self) -> Player | None:
        return self._winner

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    def pieces_of(self, player: Player) -> int:
        return self.p0 if player == 0 else self.p1


class Connect4:
    """Connect 4 Env. Stateless; all state is in the `Connect4State`."""

    name = "connect4"
    num_actions = NUM_COLS

    def initial_state(self) -> Connect4State:
        return Connect4State(
            p0=0,
            p1=0,
            heights=(0,) * NUM_COLS,
            to_play=0,
            move_number=0,
            _winner=None,
            _is_terminal=False,
        )

    def legal_actions(self, state: Connect4State) -> list[Action]:
        if state._is_terminal:
            return []
        return [c for c in range(NUM_COLS) if state.heights[c] < NUM_ROWS]

    def step(self, state: Connect4State, action: Action) -> Connect4State:
        if state._is_terminal:
            raise ValueError("step() called on terminal state")
        if not 0 <= action < NUM_COLS:
            raise ValueError(f"illegal action {action}")
        h = state.heights[action]
        if h >= NUM_ROWS:
            raise ValueError(f"column {action} is full")

        bit = 1 << (action * 7 + h)
        if state.to_play == 0:
            new_p0 = state.p0 | bit
            new_p1 = state.p1
            mover_bits = new_p0
        else:
            new_p0 = state.p0
            new_p1 = state.p1 | bit
            mover_bits = new_p1

        new_heights = list(state.heights)
        new_heights[action] = h + 1
        new_move_number = state.move_number + 1

        # Win check
        if _has_four(mover_bits):
            return Connect4State(
                p0=new_p0,
                p1=new_p1,
                heights=tuple(new_heights),
                to_play=1 - state.to_play,
                move_number=new_move_number,
                _winner=state.to_play,
                _is_terminal=True,
            )

        # Draw check
        if new_move_number >= NUM_COLS * NUM_ROWS:
            return Connect4State(
                p0=new_p0,
                p1=new_p1,
                heights=tuple(new_heights),
                to_play=1 - state.to_play,
                move_number=new_move_number,
                _winner=None,
                _is_terminal=True,
            )

        return Connect4State(
            p0=new_p0,
            p1=new_p1,
            heights=tuple(new_heights),
            to_play=1 - state.to_play,
            move_number=new_move_number,
            _winner=None,
            _is_terminal=False,
        )

    def terminal_value(self, state: Connect4State) -> float:
        """Value from perspective of state.to_play (the side that would move next)."""
        if not state._is_terminal:
            raise ValueError("terminal_value() on non-terminal state")
        if state._winner is None:
            return 0.0
        # The player who just made the winning move is (1 - state.to_play).
        # So from state.to_play's perspective, they LOST: value = -1.
        return -1.0

    def render(self, state: Connect4State) -> str:
        """ASCII rendering, top row first."""
        lines = []
        for r in range(NUM_ROWS - 1, -1, -1):
            row = []
            for c in range(NUM_COLS):
                bit = 1 << (c * 7 + r)
                if state.p0 & bit:
                    row.append("X")
                elif state.p1 & bit:
                    row.append("O")
                else:
                    row.append(".")
            lines.append("|" + " ".join(row) + "|")
        lines.append("+" + "-" * (NUM_COLS * 2 - 1) + "+")
        lines.append(" " + " ".join(str(c) for c in range(NUM_COLS)))
        status = (
            f"turn={state.move_number} to_play={state.to_play} "
            f"terminal={state._is_terminal} winner={state._winner}"
        )
        return "\n".join([status] + lines)
