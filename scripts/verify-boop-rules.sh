#!/usr/bin/env bash
#
# Large-batch Python<->TypeScript parity verification for Boop.
#
# Replays N random games via both the Python port and the TypeScript bridge,
# asserting byte-identical states after every step. If any divergence is found,
# prints a minimal reproducer and exits non-zero.
#
# Usage:   bash scripts/verify-boop-rules.sh [n_games]
# Default: 10000 games.

set -euo pipefail

cd "$(dirname "$0")/.."

N_GAMES="${1:-10000}"

# Install bridge deps once if missing.
BRIDGE_DIR="src/mlfactory/games/boop/bridge"
if [ ! -d "$BRIDGE_DIR/node_modules" ]; then
    echo "Installing Boop TS-bridge dependencies (one-time)..."
    (cd "$BRIDGE_DIR" && npm install --silent)
fi

uv run python -c "
import random, statistics, sys, time
from mlfactory.games.boop.bridge_client import BoopBridge
from mlfactory.games.boop.rules import Boop
from mlfactory.games.boop.parity import assert_parity

env = Boop()
n_games = int($N_GAMES)
start = time.monotonic()
total_moves = 0
parity_fails = 0
terminal_states = {0: 0, 1: 0, None: 0}
move_lengths = []

with BoopBridge() as br:
    rng = random.Random(0)
    for g in range(n_games):
        gid, ts_state = br.new_game()
        py_state = env.initial_state()
        try:
            assert_parity(py_state, ts_state, f'game {g} start')
        except AssertionError as e:
            print(e)
            parity_fails += 1
            break
        moves = 0
        while not py_state.is_terminal and moves < 500:
            legal_py = sorted(env.legal_actions(py_state))
            legal_ts = sorted(br.legal_actions(gid))
            if legal_py != legal_ts:
                print(f'game {g} move {moves} legal mismatch:\n py={legal_py}\n ts={legal_ts}')
                parity_fails += 1
                break
            a = rng.choice(legal_py)
            new_ts = br.step(gid, a)
            new_py = env.step(py_state, a)
            try:
                assert_parity(new_py, new_ts, f'game {g} move {moves}')
            except AssertionError as e:
                print(str(e))
                print(f'action was: {a}')
                print('state BEFORE action:')
                print(env.render(py_state))
                parity_fails += 1
                break
            py_state = new_py
            moves += 1
        if parity_fails:
            break
        total_moves += moves
        move_lengths.append(moves)
        terminal_states[py_state.winner] += 1
        br.close_game(gid)
elapsed = time.monotonic() - start

print()
print(f'=== Boop rules parity: {n_games} games ===')
print(f'Total state transitions: {total_moves}')
print(f'Parity failures:         {parity_fails}')
print(f'Wall time:               {elapsed:.1f}s ({total_moves/max(elapsed,1e-9):.0f} moves/s)')
if move_lengths:
    print(f'Moves/game: mean={statistics.mean(move_lengths):.1f}  median={int(statistics.median(move_lengths))}  max={max(move_lengths)}')
print(f'Wins: orange={terminal_states[0]}  gray={terminal_states[1]}  none={terminal_states[None]}')
if terminal_states[0] + terminal_states[1] > 0:
    p1_rate = terminal_states[0] / (terminal_states[0] + terminal_states[1])
    print(f'Orange (player 1) win rate under random play: {p1_rate:.3f}')

if parity_fails:
    sys.exit(1)
print()
print('PARITY OK')
"
