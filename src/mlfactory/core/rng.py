"""Seeded RNG helpers."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager


def make_rng(seed: int | None = None) -> random.Random:
    """A deterministic `random.Random` instance. None = unseeded."""
    return random.Random(seed)


@contextmanager
def seeded(rng: random.Random, seed: int) -> Iterator[None]:
    """Temporarily seed an rng, restore its state after."""
    state = rng.getstate()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.setstate(state)
