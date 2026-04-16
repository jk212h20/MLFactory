"""Tests for UniformEvaluator and NetEvaluator."""

from __future__ import annotations

import numpy as np

from mlfactory.agents.alphazero.evaluator import NetEvaluator, UniformEvaluator, _masked_softmax
from mlfactory.agents.alphazero.net import AlphaZeroNet, NetConfig
from mlfactory.games.boop import Boop
from mlfactory.games.boop.encode import encode_state, legal_mask


def test_uniform_evaluator_legal_actions() -> None:
    env = Boop()
    s = env.initial_state()
    ev = UniformEvaluator(env)
    res = ev.evaluate(s)

    legal = env.legal_actions(s)
    # Priors sum to 1 over legal actions.
    assert abs(res.priors[legal].sum() - 1.0) < 1e-6
    # Priors are zero on illegal actions.
    illegal_mask = np.ones(env.num_actions, dtype=bool)
    illegal_mask[legal] = False
    assert res.priors[illegal_mask].sum() == 0.0
    # All legal priors are equal.
    assert np.allclose(res.priors[legal], 1.0 / len(legal))
    assert res.value == 0.0


def test_uniform_evaluator_on_limited_legal_set() -> None:
    """After a few moves, fewer cells are legal — uniform prior still sums to 1."""
    env = Boop()
    s = env.initial_state()
    # Play a few random moves.
    import random

    rng = random.Random(0)
    for _ in range(5):
        legal = env.legal_actions(s)
        s = env.step(s, legal[rng.randrange(len(legal))])

    ev = UniformEvaluator(env)
    res = ev.evaluate(s)
    legal = env.legal_actions(s)
    assert abs(res.priors.sum() - 1.0) < 1e-6
    assert np.allclose(res.priors[legal], 1.0 / len(legal))


def test_masked_softmax_all_legal() -> None:
    logits = np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float32)
    mask = np.array([True, True, True, True])
    p = _masked_softmax(logits, mask)
    assert abs(p.sum() - 1.0) < 1e-6
    # Largest logit -> largest probability
    assert p.argmax() == 2


def test_masked_softmax_zeroes_illegal() -> None:
    logits = np.array([1.0, 2.0, 3.0, 100.0], dtype=np.float32)
    mask = np.array([True, True, True, False])  # action 3 illegal despite huge logit
    p = _masked_softmax(logits, mask)
    assert p[3] == 0.0
    assert abs(p.sum() - 1.0) < 1e-6


def test_masked_softmax_no_legal() -> None:
    logits = np.zeros(4, dtype=np.float32)
    mask = np.array([False, False, False, False])
    p = _masked_softmax(logits, mask)
    assert np.allclose(p, 0.0)


def test_net_evaluator_shapes_and_legal_masking() -> None:
    env = Boop()
    cfg = NetConfig(in_channels=11, board_h=6, board_w=6, n_actions=104, num_blocks=1, channels=8)
    net = AlphaZeroNet(cfg).eval()

    def encode(state):
        return encode_state(state), legal_mask(state)

    ev = NetEvaluator(net, encoder=encode, device="cpu", name="tiny_net")
    s = env.initial_state()
    res = ev.evaluate(s)

    legal = env.legal_actions(s)
    # Priors shape correct
    assert res.priors.shape == (env.num_actions,)
    # Illegal actions get zero prior
    illegal_mask = np.ones(env.num_actions, dtype=bool)
    illegal_mask[legal] = False
    assert res.priors[illegal_mask].sum() == 0.0
    # Legal priors sum to 1
    assert abs(res.priors[legal].sum() - 1.0) < 1e-5
    # Value in [-1, 1]
    assert -1.0 <= res.value <= 1.0


def test_net_evaluator_batch_matches_singleton() -> None:
    env = Boop()
    cfg = NetConfig(in_channels=11, board_h=6, board_w=6, n_actions=104, num_blocks=1, channels=8)
    net = AlphaZeroNet(cfg).eval()

    def encode(state):
        return encode_state(state), legal_mask(state)

    ev = NetEvaluator(net, encoder=encode, device="cpu", name="tiny_net")
    s = env.initial_state()
    solo = ev.evaluate(s)
    batched = ev.evaluate_batch([s, s])
    # The two batched results are for the same state -> identical.
    assert np.allclose(batched[0].priors, solo.priors)
    assert np.allclose(batched[1].priors, solo.priors)
    assert abs(batched[0].value - solo.value) < 1e-5
