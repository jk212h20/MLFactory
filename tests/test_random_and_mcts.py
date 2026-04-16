"""Phase 1 core claims:

1. Random vs Random is roughly 50/50 after colour balancing (no side bug).
2. MCTS with modest budget beats Random decisively on Connect 4.
3. Bigger MCTS beats smaller MCTS.
4. MCTS never makes an illegal move.
"""

from __future__ import annotations

import pytest

from mlfactory.agents.mcts import MCTSAgent
from mlfactory.agents.random_agent import RandomAgent
from mlfactory.games.connect4 import Connect4
from mlfactory.tools.arena import play_match


@pytest.fixture
def env() -> Connect4:
    return Connect4()


def test_mcts_plays_legal_moves(env: Connect4) -> None:
    agent = MCTSAgent(n_simulations=50, seed=0)
    s = env.initial_state()
    while not s.is_terminal:
        a = agent.act(env, s)
        assert a in env.legal_actions(s)
        s = env.step(s, a)
    assert s.is_terminal


def test_random_vs_random_is_balanced(env: Connect4) -> None:
    """With colour balancing, random vs random should be within a reasonable window of 50%.

    We play 400 games (200 each colour); with p=0.5 the standard deviation of
    the score is sqrt(0.25*400) = 10, so ±3 sigma is ~30 wins, i.e. a range
    of ~37%–63% is typical. We check a looser window (35%–65%) for robustness.
    """
    a = RandomAgent(name="random_a", seed=1)
    b = RandomAgent(name="random_b", seed=2)
    result = play_match(env, a, b, n_games=400)
    assert 0.35 <= result.a_win_rate <= 0.65, (
        f"random vs random wildly imbalanced: a_win_rate={result.a_win_rate:.3f}"
    )


def test_mcts_crushes_random(env: Connect4) -> None:
    """MCTS(200) should beat Random convincingly on Connect 4."""
    mcts = MCTSAgent(n_simulations=200, seed=0, name="mcts200")
    rnd = RandomAgent(name="rnd", seed=42)
    # 60 games, colour balanced
    result = play_match(env, mcts, rnd, n_games=60)
    assert result.a_win_rate > 0.85, (
        f"MCTS(200) should dominate random, got {result.a_win_rate:.3f} "
        f"({result.a_wins}W/{result.b_wins}L/{result.draws}D)"
    )


@pytest.mark.slow
def test_bigger_mcts_beats_smaller(env: Connect4) -> None:
    """MCTS(800) should beat MCTS(100) > 55%."""
    big = MCTSAgent(n_simulations=800, seed=0, name="mcts800")
    small = MCTSAgent(n_simulations=100, seed=1, name="mcts100")
    result = play_match(env, big, small, n_games=40)
    # Using > 0.55 is a weak claim but stronger would be flaky at 40 games.
    assert result.a_win_rate > 0.55, f"big MCTS should beat small MCTS, got {result.a_win_rate:.3f}"
