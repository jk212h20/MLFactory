---
type: trail
status: draft
created: 2026-04-16
updated: 2026-04-16
tags: [phase-0, kickoff]
---

# 2026-04 — Getting MLFactory off the ground

The first month. The goal was to go from an empty directory to a repo that can credibly claim "we are building a self-improving automated games strategy research factory" without that being aspirational fluff.

## Setting out
A fresh project in `~/ActiveProjects/MLFactory`: Python-centric, plugging into the existing JS/TS game projects (Boop, Leverage, dotsAnd) via adapters. First target: **Boop**. Hardware: M4 Max, 128 GB. Wiki style: Karpathy-flavoured — sources drive questions drive techniques, insights get captured, trails narrate.

## What we read
- [[sources/silver2017-alphazero]] — the general recipe.
- [[sources/silver2017-alphago-zero]] — the canonical detailed spec for 2-player perfect-info games.
- [[sources/anthony2017-expert-iteration]] — independent derivation at a compute scale closer to ours. Main source of encouragement that we can do this locally.
- [[sources/browne2012-mcts-survey]] — baseline vocabulary for MCTS.
- [[sources/schrittwieser2020-muzero]] — confirmed that we do NOT need MuZero for Boop (rules are free).

## Questions opened
- [[questions/Q-001-where-to-start]] — what algorithm and starting architecture. **Answered**: AlphaZero-lite (4 blocks × 64 channels to start), built on top of vanilla UCT MCTS as a Phase-1 stepping stone. Continuous (gate-less) training. PUCT + Dirichlet noise. D4 augmentation. 100–200 MCTS sims/move.

## Techniques drafted (not yet promoted to stable)
- [[techniques/mcts-uct]] — will promote after Phase 1 implements it.
- [[techniques/self-play-pipeline]] — will promote after Phase 3 proves it works end-to-end.

## Advice written
- [[advice/getting-started]] — the ladder for approaching a new game.
- [[advice/debugging-rl]] — the checklist for when things go wrong.

## Insights logged
_None yet — Phase 0 ran no experiments. The first insight will come from Phase 1's UCT-vs-random results (expect nothing surprising) or Phase 2's Boop rules-parity investigation (likely to surface at least one rule we got wrong on first reading)._

## Loose ends heading into Phase 1
- Q-002 on gating vs continuous training not yet written up as a formal question.
- Q-003 on rules-parity methodology needs to exist before Phase 2 begins.
- No game code yet. `src/mlfactory/` is mostly empty directories.
- No arena, no agents, no game adapters.

## What we proved by end of Phase 0
- Python 3.13 + uv + torch 2.11 on MPS works on this M4 Max. Five green tests confirm.
- The package is importable, the CLI runs (`mlfactory doctor` prints Python/Torch/MPS status).
- The wiki workflow is documented, templated, and has five ingested sources, two technique drafts, two advice pages, one answered question, and this trail.
- The repo is its own git repo; plain-English commits; projects.json and PATTERNS.md updated in the parent ActiveProjects workspace.

On to Phase 1: vanilla MCTS on tic-tac-toe and Connect 4, with an arena that reports win rates honestly.

## Phase 1 — Connect 4, UCT, Arena (2026-04-16)

Skipped tic-tac-toe entirely at the user's call ("too boring"). Jumped straight to Connect 4.

What landed:
- **Env protocol** (`src/mlfactory/core/env.py`) — immutable states, pure `step()`, side-to-move in the state, legal-action mask is authoritative.
- **Connect 4** (`src/mlfactory/games/connect4.py`) — bitboard representation with the 49-bit sentinel-row trick so all four win checks are a few shifts + ANDs. 12 tests cover vert / horiz / both diagonals / illegals / immutability / legal-action consistency over 100 random games.
- **Agents**: `RandomAgent` and `MCTSAgent` (vanilla UCT with light random playouts). The sign convention for backprop (value is from perspective of mover-into-node, flip every ply) was the main thing I had to think through carefully; it's documented in the technique page.
- **Arena** (`src/mlfactory/tools/arena.py`) — `play_match`, `round_robin`, Wilson CIs, ELO via gradient descent on Bradley–Terry log-likelihood. Colour-balanced by default.
- **CLI**: `mlfactory tournament` and `mlfactory match` with `rich`-rendered tables.

Headline result from the Phase 1 tournament (40 games/pair, colour-balanced, ~26 s total):

| rank | agent   | ELO  |
|------|---------|------|
| 1    | mcts800 | 2161 |
| 2    | mcts200 | 1811 |
| 3    | mcts50  | 1510 |
| 4    | random  | 1500 |

**The one surprise**: mcts50 is statistically indistinguishable from random on Connect 4. 50 rollouts is below the signal threshold for vanilla UCT with light playouts. That became [[insights/2026-04-16-mcts-logarithmic-in-sims]] — our first real insight, and a direct input into Phase 3 planning (don't try to run AlphaZero-lite with fewer than ~100 sims unless the prior is already meaningful).

Techniques promoted:
- [[techniques/mcts-uct]] — from `seed` to `stable`, now linked to its implementation and its validation.

Questions opened but not yet answered:
- Q-006: Does the 300-ELO-per-4x law hold on Boop?
- Q-007: Analogous noise floor for AlphaZero-style MCTS?
- Q-008: Does heavier rollout policy change the threshold?

Insights logged:
- [[insights/2026-04-16-mcts-logarithmic-in-sims]]

On to Phase 2: port Boop rules and prove parity with the TypeScript source.
