#!/usr/bin/env bash
#
# Phase 2 proof: Boop rules ported and verified against the TypeScript source,
# D4 symmetry group validated, tensor encoding tested, MCTS plays Boop correctly.

set -euo pipefail

B="\033[1m"; G="\033[32m"; C="\033[36m"; N="\033[0m"
hr() { printf "\n${C}%s${N}\n" "────────────────────────────────────────────────────────────────────"; }
hd() { printf "${B}== %s ==${N}\n" "$1"; }

cd "$(dirname "$0")/.."

hr
hd "1. Full test suite (rules, parity, symmetry, encoding, MCTS)"
uv run pytest -v

hr
hd "2. Boop rules parity with TypeScript (1000 random games)"
bash scripts/verify-boop-rules.sh 1000

hr
hd "3. MCTS tournament on Boop"
hd "   (20 games per pair, colour-balanced)"
uv run mlfactory tournament \
    --game boop \
    --agents random,mcts50,mcts200 \
    --games-per-match 20 \
    --seed 0

hr
hd "4. MCTS tournament on Connect 4 (regression check after sign fix)"
uv run mlfactory tournament \
    --game connect4 \
    --agents random,mcts50,mcts200,mcts800 \
    --games-per-match 40 \
    --seed 0

hr
printf "${G}${B}Phase 2 demo complete.${N}\n"
printf "Read ${C}results/phase2-boop-parity.md${N} for the full writeup.\n"
