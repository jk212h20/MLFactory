"""Hand-written tricky-position tests for the Boop rules port.

These tests do NOT rely on the TS bridge — they verify our Python port
against our reading of the rules. The bridge parity test (test_boop_parity)
then validates that reading against the authoritative TS source.
"""

from __future__ import annotations

import random

import pytest

from mlfactory.games.boop.rules import (
    BOARD_SIZE,
    EMPTY,
    G_CAT,
    G_KITTEN,
    N_CELLS,
    N_PLACE_ACTIONS,
    O_CAT,
    O_KITTEN,
    Boop,
    BoopState,
    replace,
)


def _action_place(piece_kind: int, row: int, col: int) -> int:
    return piece_kind * N_CELLS + row * BOARD_SIZE + col


def _set_cell(board: list[int], row: int, col: int, piece: int) -> None:
    board[row * BOARD_SIZE + col] = piece


def _make_state(
    board_map: dict[tuple[int, int], int],
    to_play: int = 0,
    orange_pool: tuple[int, int, int] = (8, 0, 0),
    gray_pool: tuple[int, int, int] = (8, 0, 0),
) -> BoopState:
    """Construct a state with specific pieces placed."""
    board = [EMPTY] * N_CELLS
    for (r, c), piece in board_map.items():
        _set_cell(board, r, c, piece)
    return BoopState(
        board=tuple(board),
        orange_pool=orange_pool,
        gray_pool=gray_pool,
        to_play=to_play,
        phase="playing",
        winner=None,
        move_number=0,
        pending_options=(),
        _is_terminal=False,
    )


@pytest.fixture
def env() -> Boop:
    return Boop()


# --- initial state ----------------------------------------------------


def test_initial_state(env: Boop) -> None:
    s = env.initial_state()
    assert s.to_play == 0
    assert s.phase == "playing"
    assert not s.is_terminal
    assert s.orange_pool == (8, 0, 0)
    assert s.gray_pool == (8, 0, 0)
    assert all(p == EMPTY for p in s.board)
    # At game start, each of 36 cells with a kitten (no cats yet) = 36 legal actions.
    assert len(env.legal_actions(s)) == 36


# --- boop mechanic ---------------------------------------------------


def test_boop_simple_push(env: Boop) -> None:
    """Orange kitten at (3,3), orange places kitten at (3,2) -> pushes (3,3) to (3,4)."""
    s = _make_state({(3, 3): O_KITTEN}, to_play=0, orange_pool=(7, 0, 0))
    # Place kitten at (3, 2).
    s2 = env.step(s, _action_place(0, 3, 2))
    assert s2.cell(3, 2) == O_KITTEN  # placed
    assert s2.cell(3, 3) == EMPTY  # was pushed
    assert s2.cell(3, 4) == O_KITTEN  # landed here


def test_boop_kitten_cannot_boop_cat(env: Boop) -> None:
    """Orange kitten placement next to a gray cat: the cat does not move."""
    s = _make_state({(3, 3): G_CAT}, to_play=0, gray_pool=(8, 0, 0))
    s2 = env.step(s, _action_place(0, 3, 2))  # orange kitten at (3,2)
    assert s2.cell(3, 2) == O_KITTEN
    assert s2.cell(3, 3) == G_CAT  # cat still there
    assert s2.cell(3, 4) == EMPTY


def test_boop_cat_can_boop_kitten(env: Boop) -> None:
    """Orange CAT placement pushes gray kitten."""
    s = _make_state(
        {(3, 3): G_KITTEN},
        to_play=0,
        orange_pool=(7, 1, 0),
        gray_pool=(7, 0, 0),
    )
    s2 = env.step(s, _action_place(1, 3, 2))  # orange cat at (3,2)
    assert s2.cell(3, 2) == O_CAT
    assert s2.cell(3, 3) == EMPTY
    assert s2.cell(3, 4) == G_KITTEN


def test_boop_cat_can_boop_cat(env: Boop) -> None:
    """Orange cat placement pushes gray cat."""
    s = _make_state(
        {(3, 3): G_CAT},
        to_play=0,
        orange_pool=(7, 1, 0),
        gray_pool=(7, 0, 0),
    )
    s2 = env.step(s, _action_place(1, 3, 2))
    assert s2.cell(3, 2) == O_CAT
    assert s2.cell(3, 3) == EMPTY
    assert s2.cell(3, 4) == G_CAT


def test_boop_blocked_by_occupied_destination(env: Boop) -> None:
    """Piece at (3,3), another at (3,4): placing at (3,2) cannot push (3,3)."""
    s = _make_state(
        {(3, 3): G_KITTEN, (3, 4): G_KITTEN},
        to_play=0,
        orange_pool=(8, 0, 0),
        gray_pool=(6, 0, 0),
    )
    s2 = env.step(s, _action_place(0, 3, 2))
    # (3,3) and (3,4) both unchanged
    assert s2.cell(3, 3) == G_KITTEN
    assert s2.cell(3, 4) == G_KITTEN
    # Gray pool unchanged (nothing pushed off)
    assert s2.gray_pool == (6, 0, 0)


def test_boop_push_off_board_returns_to_pool(env: Boop) -> None:
    """Piece at edge gets pushed off; returns to owner's pool."""
    # Gray kitten at (0, 0). Orange kitten placed at (1, 1) boops diagonally.
    # Direction from (1,1) to (0,0) is (-1,-1). Destination is (-1,-1) -> off board.
    s = _make_state(
        {(0, 0): G_KITTEN},
        to_play=0,
        orange_pool=(8, 0, 0),
        gray_pool=(7, 0, 0),
    )
    s2 = env.step(s, _action_place(0, 1, 1))
    assert s2.cell(0, 0) == EMPTY
    assert s2.cell(1, 1) == O_KITTEN
    assert s2.gray_pool == (8, 0, 0)  # kitten returned


def test_boop_multiple_adjacents_all_boop(env: Boop) -> None:
    """All 8 neighbours get booped when placed in centre (if not blocked/cats)."""
    # Ring of gray kittens around (3,3); orange kitten placed at (3,3).
    layout: dict[tuple[int, int], int] = {}
    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        layout[(3 + dr, 3 + dc)] = G_KITTEN
    s = _make_state(layout, to_play=0, orange_pool=(8, 0, 0), gray_pool=(0, 0, 0))
    s2 = env.step(s, _action_place(0, 3, 3))
    # All eight neighbours should have moved 1 step away from center.
    # Expected new positions: offset by (dr, dc) from (3+dr,3+dc), so (3+2dr, 3+2dc).
    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        new_r, new_c = 3 + 2 * dr, 3 + 2 * dc
        assert s2.cell(new_r, new_c) == G_KITTEN, f"expected push to ({new_r},{new_c})"
    # (3,3) has the placed orange kitten; the 8 original adjacent squares are empty.
    assert s2.cell(3, 3) == O_KITTEN
    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        assert s2.cell(3 + dr, 3 + dc) == EMPTY


# --- graduation -------------------------------------------------------


def test_graduation_single_option_auto(env: Boop) -> None:
    """Three orange kittens in a row -> auto graduate, all become cats."""
    # Orange kittens at (2,2), (2,3), (2,4). It is orange's turn, they just placed.
    # We'll simulate by placing the 3rd kitten at (2,4); the first two already exist.
    # But placement may boop neighbours, so put them with no adjacent pieces.
    s = _make_state(
        {(2, 2): O_KITTEN, (2, 3): O_KITTEN},
        to_play=0,
        orange_pool=(6, 0, 0),  # already placed 2
    )
    s2 = env.step(s, _action_place(0, 2, 4))  # orange kitten at (2,4)
    # All three should be cleared; orange pool: kittens_retired 3, cats_in_pool 3.
    assert s2.cell(2, 2) == EMPTY
    assert s2.cell(2, 3) == EMPTY
    assert s2.cell(2, 4) == EMPTY
    k, cat, retired = s2.orange_pool
    assert retired == 3
    assert cat == 3
    # And turn advances to gray.
    assert s2.to_play == 1
    assert s2.phase == "playing"


def test_graduation_multi_option_pauses(env: Boop) -> None:
    """Four in a row -> two possible triples; phase becomes selecting_graduation."""
    # Kittens at (2,1), (2,2), (2,3). Orange places (2,4): line of 4 at (2,1..2,4).
    # The two triples are (2,1)(2,2)(2,3) and (2,2)(2,3)(2,4).
    s = _make_state(
        {(2, 1): O_KITTEN, (2, 2): O_KITTEN, (2, 3): O_KITTEN},
        to_play=0,
        orange_pool=(5, 0, 0),
    )
    s2 = env.step(s, _action_place(0, 2, 4))
    assert s2.phase == "selecting_graduation"
    assert len(s2.pending_options) == 2
    # Still orange's turn (to choose option).
    assert s2.to_play == 0
    legal = env.legal_actions(s2)
    assert legal == [N_PLACE_ACTIONS, N_PLACE_ACTIONS + 1]

    # Pick option 0 and resolve.
    s3 = env.step(s2, N_PLACE_ACTIONS)
    assert s3.phase == "playing"
    # Three cells removed.
    non_empty = sum(1 for p in s3.board if p != EMPTY)
    # Only one kitten should remain: the one NOT in chosen option.
    assert non_empty == 1


def test_graduation_all_kittens_become_cats(env: Boop) -> None:
    """Three kittens graduate -> pool gains 3 cats, 3 kittens retired."""
    s = _make_state(
        {(2, 2): O_KITTEN, (2, 3): O_KITTEN},
        to_play=0,
        orange_pool=(6, 0, 0),
    )
    s2 = env.step(s, _action_place(0, 2, 4))
    assert s2.orange_pool[1] == 3  # cats
    assert s2.orange_pool[2] == 3  # retired counter


def test_graduation_cat_in_line_returns_to_pool(env: Boop) -> None:
    """Line of (kitten, cat, kitten): line is graduated; cat just returns to pool.

    Only the kittens increment kittens_retired.
    """
    s = _make_state(
        {(2, 2): O_KITTEN, (2, 3): O_CAT},
        to_play=0,
        orange_pool=(6, 1, 0),
    )
    s2 = env.step(s, _action_place(0, 2, 4))
    # The cat at (2,3) cannot be booped by a kitten placement. No boops occur.
    # Line (2,2)=kitten, (2,3)=cat, (2,4)=kitten -> graduation.
    # Retired: only the 2 kittens. Cats in pool: 1 (starting) + 2 (from kittens) + 1 (cat
    # returned from the line) = 4.
    assert s2.orange_pool[2] == 2  # retired: only 2 kittens
    assert s2.orange_pool[1] == 4  # cats in pool: 1 starting + 2 new + 1 returned


# --- stranded-piece fallback graduation ------------------------------


def test_stranded_graduation_when_all_eight_on_board(env: Boop) -> None:
    """Player has all 8 kittens on board (no pool, no retired) and no 3-in-a-row:
    one kitten must be graduated after their next move.

    We'll test the simpler invariant: after placing the 8th piece with no line,
    one kitten (first in row-major order) is removed and converted.
    """
    # Place 7 orange kittens scattered so no 3-in-a-row forms.
    layout = {
        (0, 0): O_KITTEN,
        (0, 5): O_KITTEN,
        (2, 0): O_KITTEN,
        (2, 5): O_KITTEN,
        (5, 0): O_KITTEN,
        (5, 2): O_KITTEN,
        (5, 5): O_KITTEN,
    }
    # Orange to place the 8th kitten; pool says 1 left.
    s = _make_state(layout, to_play=0, orange_pool=(1, 0, 0))
    # Place 8th at (0, 3), which is not adjacent to any existing piece (boops nothing).
    s2 = env.step(s, _action_place(0, 0, 3))
    # No 3-in-a-row exists. But all 8 pieces are on the board post-placement.
    # Wait — after placement, but also before boop would change the count. Verify:
    on_board = sum(1 for p in s2.board if p in (O_KITTEN, O_CAT))
    # Expectation: before the stranded check, 8 kittens on board; stranded check
    # graduates one -> 7 remain.
    assert on_board == 7
    # And orange_pool should gain a cat.
    assert s2.orange_pool[1] == 1
    assert s2.orange_pool[2] == 1


# --- win conditions --------------------------------------------------


def test_win_three_cats_in_row(env: Boop) -> None:
    """Three orange cats in a row = orange wins."""
    # Put 2 orange cats at (2,2) and (2,3); orange to place a cat at (2,4).
    s = _make_state(
        {(2, 2): O_CAT, (2, 3): O_CAT},
        to_play=0,
        orange_pool=(0, 1, 0),  # no kittens left, 1 cat in pool
    )
    s2 = env.step(s, _action_place(1, 2, 4))  # orange cat at (2,4)
    assert s2.is_terminal
    assert s2.winner == 0
    # Terminal value from state.to_play's perspective:
    # state.to_play is 1 (gray), loser -> value = -1.
    assert env.terminal_value(s2) == -1.0


def test_win_three_cats_via_graduation_chain(env: Boop) -> None:
    """Graduation produces cats; winning 3-in-a-row can happen via graduation.

    Here we construct a cleaner case: 2 existing orange cats, graduation produces
    a 3rd that ends up in pool (not in a row), so this specific scenario does NOT win.
    Graduation REMOVES pieces from the board — so the winning cat line has to come
    from a placement of a cat, not from a graduation.
    """
    # Confirm: three kittens graduate -> three kittens are removed; no cats placed.
    # So winning via graduation isn't possible at this state. Leave this as a
    # documentation test of the rule: graduation doesn't "place" cats on board.
    s = _make_state(
        {(2, 2): O_KITTEN, (2, 3): O_KITTEN},
        to_play=0,
        orange_pool=(6, 0, 0),
    )
    s2 = env.step(s, _action_place(0, 2, 4))
    # After graduation: three kittens removed, orange_pool[1] (cats) == 3.
    assert not s2.is_terminal
    assert s2.orange_pool[1] == 3
    # The cats are in POOL, not on the board.
    assert all(p != O_CAT for p in s2.board)


def test_win_all_eight_cats(env: Boop) -> None:
    """Having 8 cats on the board simultaneously = win (alternate win condition)."""
    # Put 7 orange cats on the board, nobody in adjacency that would boop them off.
    # Orange places an 8th cat somewhere not blocked.
    layout = {
        (0, 0): O_CAT,
        (0, 2): O_CAT,
        (0, 4): O_CAT,
        (2, 0): O_CAT,
        (2, 2): O_CAT,
        (2, 4): O_CAT,
        (4, 0): O_CAT,
    }
    s = _make_state(layout, to_play=0, orange_pool=(0, 1, 0))
    # Place cat at (4, 4), isolated, so no booping moves pieces.
    s2 = env.step(s, _action_place(1, 4, 4))
    # 8 cats on board, no row-of-3 (check cells): actually wait — (0,0),(0,2),(0,4)
    # are every-other cells so no 3-in-a-row. Good.
    on_board_cats = sum(1 for p in s2.board if p == O_CAT)
    assert on_board_cats == 8
    assert s2.is_terminal
    assert s2.winner == 0


# --- random-game smoke test -----------------------------------------


def test_random_games_terminate(env: Boop) -> None:
    """100 random self-play games must terminate within a generous move cap."""
    rng = random.Random(42)
    move_cap = 500
    wins = [0, 0, 0]  # orange, gray, none
    for _ in range(100):
        s = env.initial_state()
        moves = 0
        while not s.is_terminal and moves < move_cap:
            legal = env.legal_actions(s)
            assert legal, f"non-terminal state has no legal moves:\n{env.render(s)}"
            a = rng.choice(legal)
            s = env.step(s, a)
            moves += 1
        assert s.is_terminal, f"game did not terminate in {move_cap} moves"
        if s.winner == 0:
            wins[0] += 1
        elif s.winner == 1:
            wins[1] += 1
        else:
            wins[2] += 1
    # No draws observed with random play is fine; just sanity that we got games.
    assert wins[0] + wins[1] + wins[2] == 100


def test_immutability(env: Boop) -> None:
    """step() must not mutate its input state."""
    s = env.initial_state()
    board_before = s.board
    orange_before = s.orange_pool
    _ = env.step(s, _action_place(0, 2, 2))
    assert s.board == board_before
    assert s.orange_pool == orange_before
