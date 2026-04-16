"""MCTS plays Boop correctly.

Phase 2 sanity: with the sign-correction in MCTS (see wiki/insights/2026-04-16-mcts-sign-bug),
MCTS must decisively beat Random on Boop.
"""

from __future__ import annotations

import pytest

from mlfactory.agents.mcts import MCTSAgent
from mlfactory.agents.random_agent import RandomAgent
from mlfactory.games.boop import Boop
from mlfactory.tools.arena import play_match


@pytest.fixture
def env() -> Boop:
    return Boop()


def test_mcts_plays_legal_moves_on_boop(env: Boop) -> None:
    agent = MCTSAgent(n_simulations=30, seed=0)
    s = env.initial_state()
    moves = 0
    while not s.is_terminal and moves < 300:
        a = agent.act(env, s)
        assert a in env.legal_actions(s)
        s = env.step(s, a)
        moves += 1
    assert s.is_terminal, "MCTS-vs-MCTS game did not terminate"


@pytest.mark.slow
def test_mcts_crushes_random_on_boop(env: Boop) -> None:
    """MCTS(100) should decisively beat Random on Boop."""
    mcts = MCTSAgent(n_simulations=100, seed=0, name="mcts100")
    rnd = RandomAgent(name="rnd", seed=42)
    result = play_match(env, mcts, rnd, n_games=30)
    assert result.a_win_rate > 0.80, (
        f"MCTS(100) should dominate random on Boop, got {result.a_win_rate:.3f} "
        f"({result.a_wins}W/{result.b_wins}L/{result.draws}D)"
    )
