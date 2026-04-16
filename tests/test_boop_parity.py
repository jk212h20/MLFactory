"""Python <-> TypeScript parity tests for Boop.

We spawn the TS bridge subprocess and replay many random games, asserting that
Python and TS produce byte-identical snapshots after every single step.

These tests are marked `slow` because the bridge startup + many games take
several seconds. Use `uv run pytest -m slow` to include them, or
`bash scripts/verify-boop-rules.sh` for the large-batch version outside pytest.

If the bridge is not installed (`node_modules` missing), tests are skipped so
the suite still runs in minimal environments.
"""

from __future__ import annotations

import random
import shutil

import pytest

from mlfactory.games.boop.bridge_client import BRIDGE_DIR, BoopBridge, BridgeError
from mlfactory.games.boop.parity import assert_parity
from mlfactory.games.boop.rules import Boop

_NODE_AVAILABLE = shutil.which("node") is not None
_BRIDGE_INSTALLED = (BRIDGE_DIR / "node_modules").exists()

skip_if_bridge_missing = pytest.mark.skipif(
    not (_NODE_AVAILABLE and _BRIDGE_INSTALLED),
    reason="boop bridge not set up (needs node + `npm install` in bridge/)",
)


@skip_if_bridge_missing
def test_bridge_starts_and_initial_state_matches() -> None:
    env = Boop()
    with BoopBridge() as br:
        gid, ts_state = br.new_game()
        py_state = env.initial_state()
        assert_parity(py_state, ts_state, "initial")
        # Legal actions match (both should give 36 kitten placements).
        assert sorted(br.legal_actions(gid)) == sorted(env.legal_actions(py_state))


@pytest.mark.slow
@skip_if_bridge_missing
def test_parity_over_many_random_games() -> None:
    """Replay 500 random games, assert every single state matches between Python and TS."""
    env = Boop()
    n_games = 500
    total_moves = 0
    with BoopBridge() as br:
        rng = random.Random(42)
        for g in range(n_games):
            gid, ts_state = br.new_game()
            py_state = env.initial_state()
            assert_parity(py_state, ts_state, f"game {g} start")
            moves = 0
            while not py_state.is_terminal and moves < 500:
                legal_py = sorted(env.legal_actions(py_state))
                legal_ts = sorted(br.legal_actions(gid))
                assert legal_py == legal_ts, (
                    f"game {g} move {moves}: legal actions mismatch\n"
                    f"  py={legal_py}\n  ts={legal_ts}"
                )
                a = rng.choice(legal_py)
                new_ts = br.step(gid, a)
                new_py = env.step(py_state, a)
                assert_parity(new_py, new_ts, f"game {g} move {moves}")
                py_state = new_py
                moves += 1
            total_moves += moves
            br.close_game(gid)
    assert total_moves > 10000, f"only played {total_moves} moves"
