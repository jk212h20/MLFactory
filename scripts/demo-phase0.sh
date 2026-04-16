#!/usr/bin/env bash
#
# Phase 0 proof: scaffold + environment verified.
#
# This script demonstrates end-to-end that the MLFactory repo is set up correctly:
#   1. Tooling versions
#   2. uv-managed venv works
#   3. Torch on MPS works (the core requirement for all future ML work)
#   4. Tests pass
#   5. The wiki exists and has the expected contents
#   6. Git is initialised with a clean tree
#
# Run from the MLFactory repo root.

set -euo pipefail

# Colours
B="\033[1m"; G="\033[32m"; C="\033[36m"; R="\033[31m"; N="\033[0m"

hr() { printf "\n${C}%s${N}\n" "────────────────────────────────────────────────────────────────────"; }
hd() { printf "${B}== %s ==${N}\n" "$1"; }
ok() { printf "${G}✓${N} %s\n" "$1"; }

cd "$(dirname "$0")/.."

hr
hd "1. Tooling versions"
printf "uv:            "; uv --version
printf "python (venv): "; uv run python --version
printf "git:           "; git --version | head -1

hr
hd "2. Hardware & OS"
sw_vers | sed 's/^/  /'
printf "  Chip:          "; sysctl -n machdep.cpu.brand_string
printf "  Cores:         "; sysctl -n hw.ncpu
printf "  Memory (GB):   "; python3 -c "print(round($(sysctl -n hw.memsize)/1024**3))"

hr
hd "3. Torch + MPS"
uv run mlfactory doctor

hr
hd "4. Tests"
uv run pytest -v

hr
hd "5. Repo structure (top-level)"
ls -la | sed 's/^/  /'
printf "\n  src tree:\n"
find src -type f -name '*.py' | sort | sed 's/^/    /'
printf "\n  tests:\n"
find tests -type f -name '*.py' | sort | sed 's/^/    /'

hr
hd "6. Wiki contents"
printf "  sources:    %s files\n" "$(find wiki/sources -type f -name '*.md' ! -name '_template.md' | wc -l | tr -d ' ')"
printf "  questions:  %s files\n" "$(find wiki/questions -type f -name '*.md' ! -name '_template.md' | wc -l | tr -d ' ')"
printf "  techniques: %s files\n" "$(find wiki/techniques -type f -name '*.md' ! -name '_template.md' | wc -l | tr -d ' ')"
printf "  advice:     %s files\n" "$(find wiki/advice -type f -name '*.md' | wc -l | tr -d ' ')"
printf "  trails:     %s files\n" "$(find wiki/trails -type f -name '*.md' ! -name '_template.md' | wc -l | tr -d ' ')"
printf "  games:      %s files\n" "$(find wiki/games -type f -name '*.md' | wc -l | tr -d ' ')"
printf "  insights:   %s files (INSIGHTS.md + entries)\n" "$(find wiki/insights -type f -name '*.md' ! -name '_template.md' | wc -l | tr -d ' ')"

hr
hd "7. Git status"
git log --oneline -10 || echo "  (no commits yet)"
printf "\n  status:\n"
git status --short | sed 's/^/  /' || true

hr
printf "${G}${B}Phase 0 demo complete.${N}\n"
printf "Read ${C}results/phase0-scaffold.md${N} for the full proof checklist.\n"
