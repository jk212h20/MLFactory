"""Tests for D4 augmentation of Boop samples."""

from __future__ import annotations

import numpy as np

from mlfactory.games.boop import Boop
from mlfactory.games.boop.encode import encode_state, legal_mask
from mlfactory.games.boop.rules import N_ACTIONS, N_PLACE_ACTIONS
from mlfactory.training.augment import augment_boop, augment_many
from mlfactory.training.replay_buffer import Sample


def _initial_boop_sample() -> Sample:
    env = Boop()
    s = env.initial_state()
    planes = encode_state(s)
    # Uniform policy over 36 kitten placements (all empty cells in the initial state).
    policy = np.zeros(N_ACTIONS, dtype=np.float32)
    policy[:36] = 1.0 / 36
    return Sample(planes=planes, policy_target=policy, value_target=0.0)


def test_augment_boop_produces_8_variants() -> None:
    sample = _initial_boop_sample()
    variants = augment_boop(sample)
    assert len(variants) == 8


def test_augment_preserves_value_target() -> None:
    sample = _initial_boop_sample()
    sample = Sample(planes=sample.planes, policy_target=sample.policy_target, value_target=0.37)
    for v in augment_boop(sample):
        assert v.value_target == 0.37


def test_augment_preserves_policy_mass() -> None:
    sample = _initial_boop_sample()
    for v in augment_boop(sample):
        assert abs(v.policy_target.sum() - 1.0) < 1e-5


def test_augment_identity_is_noop() -> None:
    """Identity sym (first in the list) should be bit-identical."""
    sample = _initial_boop_sample()
    variants = augment_boop(sample)
    identity = variants[0]
    np.testing.assert_allclose(identity.planes, sample.planes)
    np.testing.assert_allclose(identity.policy_target, sample.policy_target)


def test_augment_rotation_is_not_identity() -> None:
    """A non-trivial Boop state: rotations should differ from the original."""
    env = Boop()
    s = env.initial_state()
    # Place an orange kitten at (0, 0) to break symmetry.
    s2 = env.step(s, 0)  # action 0 = kitten at cell (0, 0)
    planes = encode_state(s2)
    policy = np.zeros(N_ACTIONS, dtype=np.float32)
    # Concentrate policy mass on cell (0, 0) placement.
    policy[0] = 1.0
    sample = Sample(planes=planes, policy_target=policy, value_target=0.0)

    variants = augment_boop(sample)
    # The rot180 variant should have mass at cell (5,5) = index 35.
    rot180 = variants[2]
    assert rot180.policy_target[0] == 0.0
    assert rot180.policy_target[35] == 1.0


def test_augment_skips_graduation_choice_policy() -> None:
    """Samples with any mass on graduation-choice actions are not augmented."""
    sample = _initial_boop_sample()
    policy = np.zeros(N_ACTIONS, dtype=np.float32)
    policy[N_PLACE_ACTIONS] = 1.0  # single graduation-choice action
    sample = Sample(planes=sample.planes, policy_target=policy, value_target=0.0)
    variants = augment_boop(sample)
    assert len(variants) == 1
    np.testing.assert_allclose(variants[0].policy_target, policy)


def test_augment_many_dispatches_by_game() -> None:
    sample = _initial_boop_sample()
    # Boop: 8x.
    out = augment_many([sample], game="boop")
    assert len(out) == 8
    # Other game: no-op.
    out = augment_many([sample], game="other")
    assert len(out) == 1
