"""D4 symmetry augmentation for training samples.

For Boop, the 6x6 board has 8 symmetries (D4). Each spatial plane and the
placement portion of the policy target transforms equivariantly. Graduation-
choice actions (index >= 72) do NOT transform meaningfully under board
symmetry (their semantics depend on a computed option list in a
graduation-pending state), so samples from graduation-choice positions
are returned without augmentation.

Other games (like Connect4) don't have useful rotational symmetries for
training, so augmentation is a no-op there.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from mlfactory.games.boop.rules import N_CELLS, N_PLACE_ACTIONS
from mlfactory.games.boop.symmetry import SYMMETRIES, Symmetry
from mlfactory.training.replay_buffer import Sample


def augment_boop(sample: Sample) -> list[Sample]:
    """Return the 8 D4 transforms of a Boop sample (incl. identity).

    The input sample is expected to come from a `playing`-phase state;
    samples with graduation-choice-phase policies are skipped upstream.
    """
    planes = sample.planes  # (C, 6, 6)
    policy = sample.policy_target  # (104,)
    # Cheap sanity — only augment when the policy has its mass on placement
    # actions. If the policy has any mass on action >= 72, skip augmentation.
    if policy[N_PLACE_ACTIONS:].sum() > 1e-6:
        return [sample]

    return [_apply_boop_symmetry(sample, sym) for sym in SYMMETRIES]


def augment_many(samples: Iterable[Sample], game: str) -> list[Sample]:
    """Augment a batch of samples according to the game's symmetry group."""
    if game == "boop":
        out: list[Sample] = []
        for s in samples:
            out.extend(augment_boop(s))
        return out
    # No symmetry for other games in v1.
    return list(samples)


def _apply_boop_symmetry(sample: Sample, sym: Symmetry) -> Sample:
    # Spatial planes: permute via the sym's cell-index mapping.
    planes = sample.planes
    assert planes.shape[1] == 6 and planes.shape[2] == 6, planes.shape
    # Build a permutation for (36) cells: new_idx = sym(old_idx)
    perm = np.empty(N_CELLS, dtype=np.int64)
    for old in range(N_CELLS):
        perm[old] = sym.apply_cell_index(old)
    new_planes = np.empty_like(planes)
    flat_in = planes.reshape(planes.shape[0], N_CELLS)  # (C, 36)
    flat_out = new_planes.reshape(planes.shape[0], N_CELLS)
    flat_out[:, perm] = flat_in
    # Policy target: placement actions permute with the same mapping per piece type.
    policy = sample.policy_target
    new_policy = np.zeros_like(policy)
    # kind 0 (kitten): indices 0..35; kind 1 (cat): 36..71
    for kind in (0, 1):
        base = kind * N_CELLS
        block_in = policy[base : base + N_CELLS]
        new_policy[base + perm] = block_in
    # Graduation-choice portion (indices 72..103) shouldn't exist here (we
    # filtered), but copy through just in case.
    new_policy[N_PLACE_ACTIONS:] = policy[N_PLACE_ACTIONS:]
    return Sample(
        planes=new_planes,
        policy_target=new_policy.astype(np.float32, copy=False),
        value_target=sample.value_target,
    )
