"""Boop rules, ported from Boop/server/src/game/GameState.ts.

Differences from the TS source:

- **No networking / socket / player registration**. Just pure rules.
- **Players are integer IDs** 0 (orange, first) and 1 (gray).
- **Pieces encoded** as small integers: 0 = empty, 1 = orange kitten, 2 = orange cat,
  3 = gray kitten, 4 = gray cat. This lets us store the board as a flat tuple[int].
- **Action space** is flat, 104 integers wide:
    - 0..71  : place a piece. `action // 36` = piece type (0 kitten, 1 cat);
               `(action % 36) // 6` = row, `action % 6` = col.
    - 72..103: choose graduation option index (0..31). Only legal in the
               `selecting_graduation` phase; up to 32 options allowed (hard
               upper bound for a 6x6 board — in practice far fewer).
- **State is immutable**. `step()` returns a new state.
- **Terminal value** follows MLFactory convention: from perspective of `to_play`
  (the player who would move next). If opponent just won, to_play sees -1.

Rules summary (mirrors the TS):

- 6x6 board. Each player starts with 8 kittens, 0 cats. Max 8 cats.
- On your turn, place any piece you own (from pool) on an empty cell.
- After placement, BOOP all 8 adjacent cells: for each, try to push the piece
  one square further away. Kittens cannot boop cats. Cannot push into an
  occupied square. Pushed off the board -> returned to owner's pool.
- Then check for GRADUATION: any 3-in-a-row of current player's pieces
  (ortho or diag), as long as the trio contains at least one kitten.
  - Overlapping subsets of a line of 4+ are distinct options.
  - Exactly 1 option -> auto-execute: remove all 3, kittens become cats
    (incrementing `kittensRetired` for the line-of-kittens win condition
    semantics — here `cats_retired` just counts cats in pool and off-board).
    Cats in the line just return to pool.
  - >1 options -> game enters `selecting_graduation` phase; same player must
    pick an option via actions 72..71+N.
  - 0 options AND player has all 8 pieces on board AND cats earned so far < 8:
    graduate a single arbitrary kitten (first one found in row-major order) to
    unstick the game.
- Win:
  - 3 cats in a row (ortho or diag), OR
  - all 8 cats on the board simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from mlfactory.core.env import Action, Player

BOARD_SIZE = 6
STARTING_KITTENS = 8
MAX_CATS = 8

# Piece encoding
EMPTY = 0
O_KITTEN = 1
O_CAT = 2
G_KITTEN = 3
G_CAT = 4

# Directions for the boop check (all 8 neighbours)
DIRECTIONS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

# Directions for line-of-3 checks (each covers a line — the reverse is included
# implicitly because we scan every starting cell).
LINE_DIRECTIONS = [
    (0, 1),  # horizontal
    (1, 0),  # vertical
    (1, 1),  # diagonal down-right
    (1, -1),  # diagonal down-left
]

N_CELLS = BOARD_SIZE * BOARD_SIZE  # 36
N_PLACE_ACTIONS = 2 * N_CELLS  # 72
MAX_GRAD_OPTIONS = 32  # hard cap; real max is far smaller
N_ACTIONS = N_PLACE_ACTIONS + MAX_GRAD_OPTIONS  # 104


# --- helpers on the packed board ---------------------------------------


def _idx(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def _piece_color(p: int) -> int:
    """0 for orange, 1 for gray. Caller must ensure p != EMPTY."""
    if p == O_KITTEN or p == O_CAT:
        return 0
    return 1


def _piece_kind(p: int) -> int:
    """0 for kitten, 1 for cat. Caller must ensure p != EMPTY."""
    if p == O_KITTEN or p == G_KITTEN:
        return 0
    return 1


def _make_piece(color: int, kind: int) -> int:
    """0/1 color, 0/1 kind -> encoded piece."""
    if color == 0:
        return O_KITTEN if kind == 0 else O_CAT
    return G_KITTEN if kind == 0 else G_CAT


# --- state -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoopState:
    # 36 cells, row-major
    board: tuple[int, ...]
    # (kittens_in_pool, cats_in_pool, kittens_retired) for each player
    # kittens_retired is the TS 'kittensRetired' counter used for the 'all pieces on board'
    # edge case; it does not affect win conditions directly.
    orange_pool: tuple[int, int, int]
    gray_pool: tuple[int, int, int]
    # to_play: 0 (orange) or 1 (gray)
    to_play: Player
    # 'playing' or 'selecting_graduation' or 'finished'
    phase: str
    # Winner: None, 0, or 1
    winner: Player | None
    # Move counter
    move_number: int
    # In 'selecting_graduation' phase, the list of options. Each option is a
    # tuple of 3 (row, col) cell indices (as linearised ints). Empty otherwise.
    pending_options: tuple[tuple[int, int, int], ...]
    # Convenience flag; True iff phase == 'finished'
    _is_terminal: bool

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    def cell(self, row: int, col: int) -> int:
        return self.board[_idx(row, col)]

    def pool_of(self, player: Player) -> tuple[int, int, int]:
        return self.orange_pool if player == 0 else self.gray_pool


# --- rule implementation -----------------------------------------------


class Boop:
    """Boop Env, matching the core.env.Env protocol."""

    name = "boop"
    num_actions = N_ACTIONS

    def initial_state(self) -> BoopState:
        return BoopState(
            board=tuple([EMPTY] * N_CELLS),
            orange_pool=(STARTING_KITTENS, 0, 0),
            gray_pool=(STARTING_KITTENS, 0, 0),
            to_play=0,
            phase="playing",
            winner=None,
            move_number=0,
            pending_options=(),
            _is_terminal=False,
        )

    def _terminal_if_stuck(self, state: BoopState) -> BoopState | None:
        """If the current player has no legal actions in 'playing' phase, the
        game terminates with the opponent as winner.

        This is a rare edge case: the TS authoritative rules don't explicitly
        cover it (placePiece just returns valid:false for every action). We
        make it terminal to avoid infinite-loop hazards in agents and rollouts.
        Flagged as an MLFactory extension beyond strict TS parity. The
        condition is extremely rare under normal play (never observed in 10,000
        random parity games) because both players start with 8 kittens and
        replenish via booping.
        """
        if state._is_terminal or state.phase != "playing":
            return None
        pool = state.pool_of(state.to_play)
        # If mover has any pieces in pool (kittens or cats), they have a move
        # (because there's always at least one empty cell unless every cell is
        # filled, which implies 12 pieces on board and should be terminal via
        # other routes).
        if pool[0] > 0 or pool[1] > 0:
            return None
        # Zero pool: mover cannot place. Opponent wins by forfeit.
        winner = 1 - state.to_play
        return BoopState(
            board=state.board,
            orange_pool=state.orange_pool,
            gray_pool=state.gray_pool,
            to_play=state.to_play,
            phase="finished",
            winner=winner,
            move_number=state.move_number,
            pending_options=(),
            _is_terminal=True,
        )

    # -- legal action computation --------------------------------------

    def legal_actions(self, state: BoopState) -> list[Action]:
        if state._is_terminal:
            return []
        if state.phase == "selecting_graduation":
            return [N_PLACE_ACTIONS + i for i in range(len(state.pending_options))]
        # phase == playing: one action per (piece_type, empty cell) combination
        # for every piece type the player has in pool.
        pool = state.pool_of(state.to_play)
        kittens, cats, _ = pool
        legal: list[Action] = []
        # piece_kind 0 = kitten, 1 = cat
        for cell_idx in range(N_CELLS):
            if state.board[cell_idx] != EMPTY:
                continue
            if kittens > 0:
                legal.append(0 * N_CELLS + cell_idx)  # kitten placement
            if cats > 0:
                legal.append(1 * N_CELLS + cell_idx)  # cat placement
        return legal

    # -- step ----------------------------------------------------------

    def step(self, state: BoopState, action: Action) -> BoopState:
        if state._is_terminal:
            raise ValueError("step() on terminal state")
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action out of range: {action}")

        if state.phase == "selecting_graduation":
            return self._step_graduation_choice(state, action)
        return self._step_place(state, action)

    def would_trigger_stranded_fallback(self, state: BoopState, action: Action) -> bool:
        """Return True if taking `action` in `state` would fire the stranded-graduation
        fallback (TS: `checkAndExecuteGraduation` no-options branch with 8 pieces on board).

        The stranded fallback is NOT symmetry-invariant (it tie-breaks by row-major
        order), so symmetry augmentation code uses this to skip transitions.

        Re-runs the placement + boop + options check without committing.
        """
        if state.phase != "playing":
            return False
        if action < 0 or action >= N_PLACE_ACTIONS:
            return False

        piece_kind = action // N_CELLS
        cell_idx = action % N_CELLS
        row, col = divmod(cell_idx, BOARD_SIZE)
        if state.board[cell_idx] != EMPTY:
            return False

        pool = state.pool_of(state.to_play)
        if piece_kind == 0 and pool[0] <= 0:
            return False
        if piece_kind == 1 and pool[1] <= 0:
            return False

        board = list(state.board)
        orange_pool = list(state.orange_pool)
        gray_pool = list(state.gray_pool)
        piece = _make_piece(state.to_play, piece_kind)
        board[cell_idx] = piece
        if state.to_play == 0:
            orange_pool[piece_kind] -= 1
        else:
            gray_pool[piece_kind] -= 1

        self._execute_boop(board, row, col, piece, orange_pool, gray_pool)

        options = self._find_graduation_options(board, state.to_play)
        if options:  # any graduation options -> not the fallback branch
            return False

        # Mirror the _maybe_graduate_stranded preconditions.
        player_color = state.to_play
        on_board = sum(1 for p in board if p != EMPTY and _piece_color(p) == player_color)
        if on_board < 8:
            return False
        target_pool = orange_pool if state.to_play == 0 else gray_pool
        if target_pool[2] + target_pool[1] >= MAX_CATS:
            return False
        return True

    def _step_place(self, state: BoopState, action: Action) -> BoopState:
        if action >= N_PLACE_ACTIONS:
            raise ValueError(f"non-placement action {action} in playing phase")

        piece_kind = action // N_CELLS  # 0 kitten, 1 cat
        cell_idx = action % N_CELLS
        row, col = divmod(cell_idx, BOARD_SIZE)

        if state.board[cell_idx] != EMPTY:
            raise ValueError(f"cell ({row},{col}) occupied")

        pool = state.pool_of(state.to_play)
        kittens, cats, retired = pool
        if piece_kind == 0 and kittens <= 0:
            raise ValueError("no kittens available")
        if piece_kind == 1 and cats <= 0:
            raise ValueError("no cats available")

        # Mutable working copies (will be frozen back into a new dataclass at the end).
        board = list(state.board)
        orange_pool = list(state.orange_pool)
        gray_pool = list(state.gray_pool)

        # Place piece
        piece = _make_piece(state.to_play, piece_kind)
        board[cell_idx] = piece
        if state.to_play == 0:
            if piece_kind == 0:
                orange_pool[0] -= 1
            else:
                orange_pool[1] -= 1
        else:
            if piece_kind == 0:
                gray_pool[0] -= 1
            else:
                gray_pool[1] -= 1

        # Execute boop
        self._execute_boop(board, row, col, piece, orange_pool, gray_pool)

        # Graduation (kitten three-in-a-row mechanic)
        options = self._find_graduation_options(board, state.to_play)
        if len(options) > 1:
            # Multi-option: pause, same player chooses next.
            return BoopState(
                board=tuple(board),
                orange_pool=tuple(orange_pool),  # type: ignore[arg-type]
                gray_pool=tuple(gray_pool),  # type: ignore[arg-type]
                to_play=state.to_play,
                phase="selecting_graduation",
                winner=None,
                move_number=state.move_number + 1,
                pending_options=tuple(options[:MAX_GRAD_OPTIONS]),
                _is_terminal=False,
            )

        if len(options) == 1:
            self._execute_graduation(board, options[0], state.to_play, orange_pool, gray_pool)
        else:
            # No 3-in-a-row. Check the "all 8 on board" fallback.
            self._maybe_graduate_stranded(board, state.to_play, orange_pool, gray_pool)

        # Win check
        winner, _ = self._check_win(board, state.to_play, orange_pool, gray_pool)
        if winner is not None:
            return BoopState(
                board=tuple(board),
                orange_pool=tuple(orange_pool),  # type: ignore[arg-type]
                gray_pool=tuple(gray_pool),  # type: ignore[arg-type]
                to_play=1 - state.to_play,  # conventionally the next player
                phase="finished",
                winner=winner,
                move_number=state.move_number + 1,
                pending_options=(),
                _is_terminal=True,
            )

        return BoopState(
            board=tuple(board),
            orange_pool=tuple(orange_pool),  # type: ignore[arg-type]
            gray_pool=tuple(gray_pool),  # type: ignore[arg-type]
            to_play=1 - state.to_play,
            phase="playing",
            winner=None,
            move_number=state.move_number + 1,
            pending_options=(),
            _is_terminal=False,
        )

    def _step_graduation_choice(self, state: BoopState, action: Action) -> BoopState:
        idx = action - N_PLACE_ACTIONS
        if idx < 0 or idx >= len(state.pending_options):
            raise ValueError(f"illegal graduation option index {idx}")

        board = list(state.board)
        orange_pool = list(state.orange_pool)
        gray_pool = list(state.gray_pool)

        option = state.pending_options[idx]
        self._execute_graduation(board, option, state.to_play, orange_pool, gray_pool)

        winner, _ = self._check_win(board, state.to_play, orange_pool, gray_pool)
        if winner is not None:
            # TS quirk: selectGraduation() does NOT advance currentTurn when the
            # graduation produces a win (it only advances when phase becomes
            # 'playing'). We mirror that to preserve byte-identical parity with
            # the authoritative rules. For the MLFactory terminal_value
            # convention, we handle the sign explicitly based on winner identity
            # rather than assuming to_play == loser.
            return BoopState(
                board=tuple(board),
                orange_pool=tuple(orange_pool),  # type: ignore[arg-type]
                gray_pool=tuple(gray_pool),  # type: ignore[arg-type]
                to_play=state.to_play,
                phase="finished",
                winner=winner,
                move_number=state.move_number + 1,
                pending_options=(),
                _is_terminal=True,
            )

        return BoopState(
            board=tuple(board),
            orange_pool=tuple(orange_pool),  # type: ignore[arg-type]
            gray_pool=tuple(gray_pool),  # type: ignore[arg-type]
            to_play=1 - state.to_play,
            phase="playing",
            winner=None,
            move_number=state.move_number + 1,
            pending_options=(),
            _is_terminal=False,
        )

    # -- mechanics helpers ---------------------------------------------

    def _execute_boop(
        self,
        board: list[int],
        placed_row: int,
        placed_col: int,
        placed_piece: int,
        orange_pool: list[int],
        gray_pool: list[int],
    ) -> None:
        placed_kind = _piece_kind(placed_piece)  # 0 kitten, 1 cat

        for dr, dc in DIRECTIONS:
            adj_r = placed_row + dr
            adj_c = placed_col + dc
            if not (0 <= adj_r < BOARD_SIZE and 0 <= adj_c < BOARD_SIZE):
                continue
            adj_i = _idx(adj_r, adj_c)
            adj_piece = board[adj_i]
            if adj_piece == EMPTY:
                continue

            # Kittens cannot boop cats.
            if placed_kind == 0 and _piece_kind(adj_piece) == 1:
                continue

            dest_r = adj_r + dr
            dest_c = adj_c + dc

            if 0 <= dest_r < BOARD_SIZE and 0 <= dest_c < BOARD_SIZE:
                dest_i = _idx(dest_r, dest_c)
                if board[dest_i] != EMPTY:
                    # Blocked; "line of 2" rule.
                    continue
                # Move piece into dest_i.
                board[dest_i] = adj_piece
                board[adj_i] = EMPTY
            else:
                # Pushed off the board. Return to owner's pool.
                owner = _piece_color(adj_piece)
                kind = _piece_kind(adj_piece)
                target_pool = orange_pool if owner == 0 else gray_pool
                target_pool[kind] += 1  # kittens[0] or cats[1]
                board[adj_i] = EMPTY

    # Get a run of consecutive cells owned by `player_color` starting from (r,c)
    # along (dr,dc); returns a list of linearised indices.
    def _line_from(
        self, board: list[int], r: int, c: int, dr: int, dc: int, player_color: int
    ) -> list[int]:
        line: list[int] = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            i = _idx(r, c)
            p = board[i]
            if p == EMPTY or _piece_color(p) != player_color:
                break
            line.append(i)
            r += dr
            c += dc
        return line

    def _find_graduation_options(
        self, board: list[int], player: Player
    ) -> list[tuple[int, int, int]]:
        options: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for dr, dc in LINE_DIRECTIONS:
                    line = self._line_from(board, r, c, dr, dc, player)
                    if len(line) < 3:
                        continue
                    # For each length-3 window in the run:
                    for i in range(len(line) - 2):
                        trio = (line[i], line[i + 1], line[i + 2])
                        # Must contain at least one kitten.
                        has_kitten = any(_piece_kind(board[j]) == 0 for j in trio)
                        if not has_kitten:
                            continue
                        key = tuple(sorted(trio))  # canonicalise for dedup
                        if key in seen:
                            continue
                        seen.add(key)
                        options.append(trio)
        return options

    def _execute_graduation(
        self,
        board: list[int],
        option: tuple[int, int, int],
        player: Player,
        orange_pool: list[int],
        gray_pool: list[int],
    ) -> None:
        pool = orange_pool if player == 0 else gray_pool
        for cell_i in option:
            piece = board[cell_i]
            if piece == EMPTY:
                continue
            kind = _piece_kind(piece)
            board[cell_i] = EMPTY
            if kind == 0:
                # Kitten graduates to cat: increment retired + cats in pool.
                pool[2] += 1  # kittens_retired
                pool[1] += 1  # cats_in_pool
            else:
                # Cat returns to pool.
                pool[1] += 1

    def _maybe_graduate_stranded(
        self,
        board: list[int],
        player: Player,
        orange_pool: list[int],
        gray_pool: list[int],
    ) -> bool:
        """If the player has all 8 starting pieces on board (no pool, no retired) AND
        no 3-in-a-row, graduate one kitten (first in row-major order) to unstick.

        Matches the TS check: `piecesOnBoard >= 8 && kittensRetired + catsInPool < MAX_CATS`.
        Returns True if the fallback fired, False otherwise.
        """
        pool = orange_pool if player == 0 else gray_pool
        on_board = 0
        player_color = player
        for p in board:
            if p != EMPTY and _piece_color(p) == player_color:
                on_board += 1
        if on_board < 8:
            return False
        if pool[2] + pool[1] >= MAX_CATS:
            return False

        # Find first kitten in row-major order.
        for cell_i, p in enumerate(board):
            if p == EMPTY:
                continue
            if _piece_color(p) != player_color:
                continue
            if _piece_kind(p) != 0:
                continue
            # Graduate this one.
            board[cell_i] = EMPTY
            pool[2] += 1
            pool[1] += 1
            return True
        return False

    def _check_win(
        self, board: list[int], player: Player, orange_pool: list[int], gray_pool: list[int]
    ) -> tuple[Player | None, str | None]:
        # 3 cats in a row
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for dr, dc in LINE_DIRECTIONS:
                    ok = True
                    for i in range(3):
                        rr = r + i * dr
                        cc = c + i * dc
                        if not (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE):
                            ok = False
                            break
                        p = board[_idx(rr, cc)]
                        if p == EMPTY or _piece_color(p) != player or _piece_kind(p) != 1:
                            ok = False
                            break
                    if ok:
                        return player, "three_cats_in_row"

        # All 8 cats on the board
        cats_on_board = sum(
            1 for p in board if p != EMPTY and _piece_color(p) == player and _piece_kind(p) == 1
        )
        if cats_on_board >= 8:
            return player, "all_eight_cats"

        return None, None

    # -- terminal value -----------------------------------------------

    def terminal_value(self, state: BoopState) -> float:
        """Value from perspective of state.to_play.

        Convention: returns +1 if state.to_play is the winner, -1 if the loser,
        0 for draws. Because of the TS quirk documented in `_step_graduation_choice`,
        state.to_play at terminal may equal either the winner (graduation-choice win)
        or the loser (placement win); this method handles both correctly.

        MCTS backpropagation does not use this; it tracks the mover-into-node
        identity directly and compares against `state.winner` without ever asking
        the Env about `to_play`.
        """
        if not state._is_terminal:
            raise ValueError("terminal_value() on non-terminal state")
        if state.winner is None:
            return 0.0
        return 1.0 if state.winner == state.to_play else -1.0

    # -- rendering ----------------------------------------------------

    def render(self, state: BoopState) -> str:
        glyph = {EMPTY: ".", O_KITTEN: "o", O_CAT: "O", G_KITTEN: "g", G_CAT: "G"}
        lines: list[str] = []
        lines.append(
            f"move={state.move_number} to_play={state.to_play} "
            f"phase={state.phase} terminal={state._is_terminal} winner={state.winner}"
        )
        lines.append(f"orange pool (kittens/cats/retired): {state.orange_pool}")
        lines.append(f"gray   pool (kittens/cats/retired): {state.gray_pool}")
        lines.append("  " + " ".join(str(c) for c in range(BOARD_SIZE)))
        for r in range(BOARD_SIZE):
            row_cells = [glyph[state.board[_idx(r, c)]] for c in range(BOARD_SIZE)]
            lines.append(f"{r} " + " ".join(row_cells))
        lines.append("legend: o=orange kitten O=orange cat g=gray kitten G=gray cat")
        if state.phase == "selecting_graduation":
            lines.append(f"pending_options: {state.pending_options}")
        return "\n".join(lines)


# Expose the 'replace' helper for tests that want to construct specific states.
__all__ = [
    "Boop",
    "BoopState",
    "BOARD_SIZE",
    "STARTING_KITTENS",
    "MAX_CATS",
    "N_CELLS",
    "N_PLACE_ACTIONS",
    "N_ACTIONS",
    "EMPTY",
    "O_KITTEN",
    "O_CAT",
    "G_KITTEN",
    "G_CAT",
    "replace",
]
