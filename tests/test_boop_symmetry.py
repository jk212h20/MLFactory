"""Test the D4 symmetry group on Boop.

The critical property: symmetries commute with `step()`. That is, for any state
and any placement action,

    step(symmetry(s), symmetry(a)) == symmetry(step(s, a))

except for `pending_options` (which we don't symmetrize — see symmetry.py). If
this holds, we have a sound 8× data augmenter for training.
"""

from __future__ import annotations

import random

import pytest

from mlfactory.games.boop.rules import BoopState, Boop, N_PLACE_ACTIONS, BOARD_SIZE
from mlfactory.games.boop.symmetry import SYMMETRIES, Symmetry, was_stranded_fallback


@pytest.fixture
def env() -> Boop:
    return Boop()


def _state_without_pending(s: BoopState) -> tuple:
    """Canonical tuple for comparison that ignores pending_options."""
    return (
        s.board,
        s.orange_pool,
        s.gray_pool,
        s.to_play,
        s.phase,
        s.winner,
        s.move_number,
        s._is_terminal,
    )


def test_group_has_8_elements() -> None:
    assert len(SYMMETRIES) == 8
    names = {s.name for s in SYMMETRIES}
    assert len(names) == 8


@pytest.mark.parametrize("sym", SYMMETRIES, ids=lambda s: s.name)
def test_cell_transform_is_involution_or_cycle(sym: Symmetry) -> None:
    """Applying each D4 element a correct number of times returns to identity.

    Rotations of 90/270 are 4-cycles; 180, flips and identity are involutions.
    """
    cycle_len = {
        "identity": 1,
        "rot90": 4,
        "rot180": 2,
        "rot270": 4,
        "flip_h": 2,
        "flip_v": 2,
        "flip_diag": 2,
        "flip_anti": 2,
    }[sym.name]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rr, cc = r, c
            for _ in range(cycle_len):
                rr, cc = sym.apply_cell(rr, cc)
            assert (rr, cc) == (r, c), f"{sym.name} cycle broken at ({r},{c})"


def test_distinct_symmetries_are_distinct() -> None:
    """No two listed symmetries act identically on a probe pattern."""
    probe = [(0, 0), (0, 5), (1, 2), (2, 4)]
    sigs = []
    for sym in SYMMETRIES:
        sigs.append(tuple(sym.apply_cell(r, c) for r, c in probe))
    assert len(set(sigs)) == 8, "symmetries collapsed"


@pytest.mark.parametrize("sym", SYMMETRIES, ids=lambda s: s.name)
def test_commutativity_with_step_on_random_games(sym: Symmetry, env: Boop) -> None:
    """step(sym(s), sym(a)) == sym(step(s, a)) across many random rollouts.

    Exclusions (documented in `src/mlfactory/games/boop/symmetry.py`):
    - Graduation-choice phases are skipped (we do not symmetrize pending_options).
    - Transitions that fired the stranded-graduation fallback are skipped: the
      fallback picks "first kitten in row-major order", which is a tie-breaking
      rule that is NOT symmetry-invariant. Symmetry augmentation in training
      must detect these transitions and drop them (or pick one consistent
      representative).
    """
    rng = random.Random(12345 + hash(sym.name) % 1000)
    checked = 0
    skipped_stranded = 0
    # 10 games × ~60 moves ~= 600 checks per symmetry
    for _ in range(10):
        s = env.initial_state()
        while not s.is_terminal and checked < 300:
            if s.phase != "playing":
                legal = env.legal_actions(s)
                if not legal:
                    break
                s = env.step(s, rng.choice(legal))
                continue

            legal = env.legal_actions(s)
            if not legal:
                break
            a = rng.choice(legal)
            assert a < N_PLACE_ACTIONS

            s_then_step = env.step(s, a)

            if was_stranded_fallback(s, a):
                skipped_stranded += 1
                s = s_then_step
                continue

            transformed_1 = sym.apply_state(s_then_step)

            s_sym = sym.apply_state(s)
            a_sym = sym.apply_action(a)
            legal_sym = env.legal_actions(s_sym)
            assert a_sym in legal_sym, (
                f"{sym.name}: transformed action {a_sym} not legal in transformed state "
                f"(legal count={len(legal_sym)})"
            )
            stepped_sym = env.step(s_sym, a_sym)

            assert _state_without_pending(transformed_1) == _state_without_pending(stepped_sym), (
                f"{sym.name}: step-then-sym != sym-then-step\n"
                f"  before:\n{env.render(s)}\n"
                f"  action={a}, sym_action={a_sym}\n"
                f"  step-then-sym:\n{env.render(transformed_1)}\n"
                f"  sym-then-step:\n{env.render(stepped_sym)}"
            )

            s = s_then_step
            checked += 1
    assert checked > 50
    # Report (visible if pytest -s): sanity that the stranded skip rate is small.
    print(f"\n[{sym.name}] checked={checked}, skipped_stranded={skipped_stranded}")


def test_identity_is_identity(env: Boop) -> None:
    s = env.initial_state()
    rng = random.Random(0)
    # Make a few moves to have a non-trivial state.
    for _ in range(12):
        legal = env.legal_actions(s)
        if not legal or s.is_terminal:
            break
        s = env.step(s, rng.choice(legal))
    identity = SYMMETRIES[0]
    s2 = identity.apply_state(s)
    assert _state_without_pending(s) == _state_without_pending(s2)
    for a in range(N_PLACE_ACTIONS):
        assert identity.apply_action(a) == a
