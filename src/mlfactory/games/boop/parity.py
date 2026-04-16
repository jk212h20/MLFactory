"""Python <-> TypeScript parity comparison for Boop states.

Given a Python `BoopState` and a TS-bridge state dict, convert both to the same
canonical comparable tuple and assert equality. Mismatches bubble up as a
human-readable diff so bugs are obvious.
"""

from __future__ import annotations

from mlfactory.games.boop.rules import BoopState


def canonical_py(state: BoopState) -> dict:
    """Convert a Python BoopState to a plain comparable dict."""
    # Canonicalise pending_options: sort inner trios + outer list of options.
    pending = [tuple(sorted(opt)) for opt in state.pending_options]
    pending.sort()
    return {
        "board": list(state.board),
        "orange_pool": list(state.orange_pool),
        "gray_pool": list(state.gray_pool),
        "to_play": state.to_play,
        "phase": state.phase,
        "winner": state.winner,
        "move_number": state.move_number,
        "pending_options": [list(opt) for opt in pending],
    }


def canonical_ts(state: dict) -> dict:
    """Convert a TS-bridge state snapshot to the same canonical form."""
    pending_raw = state.get("pending_options", [])
    pending = [tuple(sorted(opt)) for opt in pending_raw]
    pending.sort()
    return {
        "board": list(state["board"]),
        "orange_pool": list(state["orange_pool"]),
        "gray_pool": list(state["gray_pool"]),
        "to_play": state["to_play"],
        "phase": state["phase"],
        "winner": state["winner"],
        "move_number": state["move_number"],
        "pending_options": [list(opt) for opt in pending],
    }


def diff(py_canon: dict, ts_canon: dict) -> list[str]:
    """Return a list of human-readable field-level differences (empty if equal)."""
    out: list[str] = []
    keys = sorted(set(py_canon) | set(ts_canon))
    for k in keys:
        pv = py_canon.get(k)
        tv = ts_canon.get(k)
        if pv != tv:
            out.append(f"  {k}:\n    py = {pv!r}\n    ts = {tv!r}")
    return out


def assert_parity(py_state: BoopState, ts_state: dict, context: str = "") -> None:
    """Raise AssertionError with a full diff if Python and TS disagree."""
    py_canon = canonical_py(py_state)
    ts_canon = canonical_ts(ts_state)
    if py_canon != ts_canon:
        header = f"Python vs TS parity MISMATCH{(' at ' + context) if context else ''}:"
        lines = [header, *diff(py_canon, ts_canon)]
        raise AssertionError("\n".join(lines))
