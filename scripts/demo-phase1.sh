#!/usr/bin/env bash
#
# Phase 1 proof: Env protocol works, Connect 4 is correct, MCTS beats Random,
# bigger MCTS beats smaller, arena produces a coherent ELO table.

set -euo pipefail

B="\033[1m"; G="\033[32m"; C="\033[36m"; N="\033[0m"
hr() { printf "\n${C}%s${N}\n" "────────────────────────────────────────────────────────────────────"; }
hd() { printf "${B}== %s ==${N}\n" "$1"; }

cd "$(dirname "$0")/.."

hr
hd "1. All tests (fast + slow)"
uv run pytest -v

hr
hd "2. Connect 4 tournament: random, mcts50, mcts200, mcts800"
hd "   (40 games per pair, colour-balanced)"
uv run mlfactory tournament \
    --game connect4 \
    --agents random,mcts50,mcts200,mcts800 \
    --games-per-match 40 \
    --seed 0

hr
hd "3. Head-to-head: mcts200 vs random (100 games)"
uv run mlfactory match \
    --game connect4 \
    --agent-a mcts200 \
    --agent-b random \
    --games 100 \
    --seed 0

hr
printf "${G}${B}Phase 1 demo complete.${N}\n"
printf "Read ${C}results/phase1-mcts-baseline.md${N} for the full writeup.\n"
