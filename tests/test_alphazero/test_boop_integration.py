"""Integration test: AlphaZeroAgent plays a full game of Boop.

This verifies the full stack (net -> evaluator -> PUCT -> agent) works
on the real target game. We use a small untrained net; we don't care
about the quality of play, just that no stage crashes.
"""

from __future__ import annotations

import time

import pytest

from mlfactory.agents.alphazero.agent import AlphaZeroAgent
from mlfactory.agents.alphazero.evaluator import NetEvaluator, UniformEvaluator
from mlfactory.agents.alphazero.net import AlphaZeroNet, NetConfig
from mlfactory.agents.alphazero.puct import PUCTConfig
from mlfactory.games.boop import Boop
from mlfactory.games.boop.encode import encode_state, legal_mask


@pytest.mark.slow
def test_alphazero_uniform_plays_full_boop_game() -> None:
    """AZ with uniform evaluator completes a Boop game without errors."""
    env = Boop()
    ev = UniformEvaluator(env)
    agent = AlphaZeroAgent(ev, PUCTConfig(n_simulations=20), seed=0, name="az-uniform")

    s = env.initial_state()
    moves = 0
    max_moves = 200
    while not s.is_terminal and moves < max_moves:
        a = agent.act(env, s)
        assert a in env.legal_actions(s)
        s = env.step(s, a)
        moves += 1
    assert s.is_terminal or moves == max_moves


@pytest.mark.slow
def test_alphazero_with_tiny_net_plays_full_boop_game() -> None:
    """AZ with a tiny fresh (untrained) net completes a Boop game."""
    env = Boop()
    cfg = NetConfig(
        in_channels=11,
        board_h=6,
        board_w=6,
        n_actions=104,
        num_blocks=2,
        channels=16,
    )
    net = AlphaZeroNet(cfg).eval()

    def encode(state):
        return encode_state(state), legal_mask(state)

    ev = NetEvaluator(net, encoder=encode, device="cpu", name="tiny_net")
    agent = AlphaZeroAgent(ev, PUCTConfig(n_simulations=15), seed=1, name="az-tiny")

    s = env.initial_state()
    moves = 0
    max_moves = 200
    start = time.monotonic()
    while not s.is_terminal and moves < max_moves:
        a = agent.act(env, s)
        assert a in env.legal_actions(s)
        s = env.step(s, a)
        moves += 1
    elapsed = time.monotonic() - start
    assert s.is_terminal or moves == max_moves
    # Sanity: one game shouldn't take more than ~30s with 15 sims/move.
    assert elapsed < 30.0, f"game took {elapsed:.1f}s — too slow"
