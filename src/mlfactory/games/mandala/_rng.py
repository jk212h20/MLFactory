"""A deterministic PRNG matching a tiny JS reference implementation, used
ONLY for JS↔Python parity testing. Not used in training.

Algorithm: mulberry32 (https://gist.github.com/tommyettinger/46a3c...).
It's a standard 32-bit seeded PRNG, one of the simplest deterministic
generators that's still decent quality. Produces floats in [0, 1).

Matching JS reference:

    function mulberry32(seed) {
      return function() {
        seed = (seed + 0x6D2B79F5) | 0;
        var t = seed;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      }
    }

The Python version below uses identical 32-bit arithmetic, so both
emit exactly the same float stream from the same seed.
"""

from __future__ import annotations

_MASK = 0xFFFFFFFF
_INCR = 0x6D2B79F5


def _imul32(a: int, b: int) -> int:
    """32-bit signed-mod integer multiply, matching JS's Math.imul."""
    # Normalize to unsigned 32-bit, multiply, take low 32, interpret as signed.
    a &= _MASK
    b &= _MASK
    prod = (a * b) & _MASK
    if prod & 0x80000000:
        return prod - 0x100000000
    return prod


class Mulberry32:
    """Seeded PRNG identical to the JS mulberry32 reference above.

    Public API mirrors random.Random where we need it:
    - random() -> float in [0, 1)
    - randrange(n) -> int in [0, n) (internally computes floor(random() * n))

    The exhaustive per-operation bit-fidelity with the JS reference is what
    makes parity tests possible.
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK

    def random(self) -> float:
        # All bitwise ops must operate on the UNSIGNED 32-bit view of
        # intermediate values, because JS's `>>>` implicitly coerces to
        # uint32 before shifting. Python ints are unbounded and arithmetic,
        # so we explicitly `& _MASK` before any right-shift on a value
        # that could be negative (from _imul32). This was the bug in the
        # initial port — `_imul32` returns a *signed* 32-bit int, and
        # `t >> 7` on a negative int in Python is arithmetic-shift of an
        # infinite-sign-extended value, not the bit pattern we want.
        self._state = (self._state + _INCR) & _MASK
        t = self._state
        # t = Math.imul(t ^ (t >>> 15), t | 1); t is still unsigned here.
        t = _imul32(t ^ (t >> 15), t | 1)
        # t is now signed. Unsign before using `>>>`-equivalent shift.
        tu = t & _MASK
        rhs = (tu + _imul32(tu ^ (tu >> 7), tu | 61)) & _MASK
        t = (tu ^ rhs) & _MASK
        return ((t ^ (t >> 14)) & _MASK) / 4294967296.0

    def randrange(self, n: int) -> int:
        """Like random.Random.randrange(n). Uses floor(random() * n) to match
        what the JS shuffleDeck does."""
        import math

        return math.floor(self.random() * n)

    # Also expose a .choice used by the parity harness.
    def choice(self, seq: list):
        return seq[self.randrange(len(seq))]
