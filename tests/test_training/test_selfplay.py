"""Tests for the self-play game generator."""

from __future__ import annotations

import numpy as np

from mlfactory.agents.alphazero.agent import AlphaZeroAgent
from mlfactory.agents.alphazero.evaluator import UniformEvaluator
from mlfactory.agents.alphazero.puct import PUCTConfig
from mlfactory.games.boop import Boop
from mlfactory.games.boop.encode import encode_state, legal_mask
from mlfactory.training.selfplay import play_selfplay_game


def _encoder(state):
    return encode_state(state), legal_mask(state)


def test_selfplay_produces_samples_and_record() -> None:
    env = Boop()
    ev = UniformEvaluator(env)
    agent = AlphaZeroAgent(ev, PUCTConfig(n_simulations=10), seed=0, name="az-test")

    result = play_selfplay_game(
        env,
        agent,
        encoder=_encoder,
        game_name="boop",
        iter_index=0,
        seed=0,
        max_moves=200,
    )

    # A game must produce some training samples.
    assert len(result.samples) > 0
    assert result.n_moves > 0
    assert len(result.record.moves) == result.n_moves
    # states list has n_moves + 1 entries (initial + after each move).
    assert len(result.record.states) == result.n_moves + 1

    # Value targets are in {-1, 0, +1} and consistent with winner.
    for sample in result.samples:
        assert sample.value_target in (-1.0, 0.0, 1.0)

    # The very first sample's mover was orange (0). If orange won, value=+1;
    # if orange lost, value=-1; draw=0.
    first_value = result.samples[0].value_target
    if result.winner is None:
        assert first_value == 0.0
    elif result.winner == 0:
        assert first_value == 1.0
    else:
        assert first_value == -1.0


def test_selfplay_policy_targets_sum_to_one() -> None:
    env = Boop()
    ev = UniformEvaluator(env)
    agent = AlphaZeroAgent(ev, PUCTConfig(n_simulations=10), seed=0)
    result = play_selfplay_game(env, agent, encoder=_encoder, game_name="boop")

    for sample in result.samples:
        s = sample.policy_target.sum()
        # Near 1.0 (rounding from PUCT visit ratios).
        assert abs(s - 1.0) < 1e-5, f"policy target sum = {s}"


def test_selfplay_record_roundtrip_via_states() -> None:
    """Replaying the recorded states should reconstruct the same sequence."""
    from mlfactory.training.sample_game import _state_from_dict

    env = Boop()
    ev = UniformEvaluator(env)
    agent = AlphaZeroAgent(ev, PUCTConfig(n_simulations=5), seed=0)
    result = play_selfplay_game(env, agent, encoder=_encoder, game_name="boop")

    # Reconstruct state 1 from state 0 + move 0.
    s0 = _state_from_dict("boop", result.record.states[0])
    s1_expected = env.step(s0, result.record.moves[0].action)
    s1_recorded = _state_from_dict("boop", result.record.states[1])
    assert s1_expected.board == s1_recorded.board
    assert s1_expected.to_play == s1_recorded.to_play


def test_selfplay_visits_are_recorded() -> None:
    env = Boop()
    ev = UniformEvaluator(env)
    agent = AlphaZeroAgent(ev, PUCTConfig(n_simulations=20), seed=0)
    result = play_selfplay_game(env, agent, encoder=_encoder, game_name="boop", record_visits=True)

    for move in result.record.moves:
        # Some visits recorded (unless the agent terminated very quickly).
        assert move.visits is not None
        assert sum(move.visits.values()) > 0
        assert move.q_values is not None
        assert move.root_value is not None
