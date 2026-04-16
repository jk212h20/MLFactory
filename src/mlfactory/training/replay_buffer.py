"""Bounded FIFO replay buffer for AlphaZero-style training.

Stores (planes, policy_target, value_target) tuples. Capacity is fixed;
oldest entries are evicted as new ones arrive. Samples are drawn uniformly
at random without replacement (per minibatch).

All storage is on CPU in NumPy (small memory footprint, no MPS surprises).
The training step copies samples to the net's device each batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    """One training example from a self-play game."""

    planes: np.ndarray  # (C, H, W) float32
    policy_target: np.ndarray  # (n_actions,) float32, sums to 1
    value_target: float  # scalar in [-1, 1], from the mover's perspective


class ReplayBuffer:
    """FIFO of Sample objects with fixed capacity.

    Parameters
    ----------
    capacity : maximum number of samples retained. Older ones are evicted.
    rng      : NumPy generator for sampling (seeded by caller).
    """

    def __init__(self, capacity: int, rng: np.random.Generator | None = None) -> None:
        self.capacity = int(capacity)
        self._data: list[Sample] = []
        self._rng = rng or np.random.default_rng()

    def __len__(self) -> int:
        return len(self._data)

    def add(self, sample: Sample) -> None:
        """Append one sample, evicting the oldest if over capacity."""
        if len(self._data) >= self.capacity:
            # Single eviction (amortised O(1) because list pops from front are O(n);
            # a deque would be faster but we sample by index a lot). For our target
            # capacity (50k) and iteration cadence (hundreds/iter), this is fine.
            self._data = self._data[-(self.capacity - 1) :]
        self._data.append(sample)

    def extend(self, samples: list[Sample]) -> None:
        for s in samples:
            self.add(s)

    def sample(self, batch_size: int) -> list[Sample]:
        """Uniform random sample without replacement. Raises if buffer has fewer."""
        if batch_size > len(self._data):
            raise ValueError(f"requested batch_size={batch_size} but buffer has {len(self._data)}")
        idx = self._rng.choice(len(self._data), size=batch_size, replace=False)
        return [self._data[i] for i in idx]

    def stack(self, samples: list[Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pack a list of samples into contiguous arrays ready for a forward pass."""
        planes = np.stack([s.planes for s in samples], axis=0)
        policies = np.stack([s.policy_target for s in samples], axis=0)
        values = np.array([s.value_target for s in samples], dtype=np.float32)
        return planes, policies, values
