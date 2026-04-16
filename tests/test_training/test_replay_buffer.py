"""Tests for ReplayBuffer."""

from __future__ import annotations

import numpy as np
import pytest

from mlfactory.training.replay_buffer import ReplayBuffer, Sample


def _make_sample(i: int, n_actions: int = 10) -> Sample:
    planes = np.full((2, 3, 3), float(i), dtype=np.float32)
    policy = np.zeros(n_actions, dtype=np.float32)
    policy[i % n_actions] = 1.0
    return Sample(planes=planes, policy_target=policy, value_target=float(i % 3 - 1))


def test_buffer_empty() -> None:
    buf = ReplayBuffer(capacity=5)
    assert len(buf) == 0
    with pytest.raises(ValueError):
        buf.sample(1)


def test_buffer_add_and_len() -> None:
    buf = ReplayBuffer(capacity=5)
    for i in range(3):
        buf.add(_make_sample(i))
    assert len(buf) == 3


def test_buffer_evicts_when_full() -> None:
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.add(_make_sample(i))
    # Keeps the newest 3 entries (i=2, 3, 4).
    assert len(buf) == 3
    vals = [float(s.planes[0, 0, 0]) for s in buf._data]
    assert vals == [2.0, 3.0, 4.0]


def test_buffer_sample_without_replacement() -> None:
    buf = ReplayBuffer(capacity=10, rng=np.random.default_rng(0))
    for i in range(10):
        buf.add(_make_sample(i))
    batch = buf.sample(5)
    assert len(batch) == 5
    ids = {float(s.planes[0, 0, 0]) for s in batch}
    assert len(ids) == 5  # all distinct


def test_buffer_sample_exceeding_size_raises() -> None:
    buf = ReplayBuffer(capacity=10)
    for i in range(3):
        buf.add(_make_sample(i))
    with pytest.raises(ValueError):
        buf.sample(5)


def test_buffer_stack_shapes() -> None:
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.add(_make_sample(i))
    batch = buf._data
    planes, policies, values = buf.stack(batch)
    assert planes.shape == (5, 2, 3, 3)
    assert policies.shape == (5, 10)
    assert values.shape == (5,)


def test_buffer_extend() -> None:
    buf = ReplayBuffer(capacity=10)
    buf.extend([_make_sample(i) for i in range(4)])
    assert len(buf) == 4
