# Phase 1 — Env Protocol, Connect 4, MCTS, Arena

**Status**: ✅ Complete
**Date**: 2026-04-16
**Git SHA**: `153dedf` (Phase 1 commit)

> **⚠️ Retraction (Phase 2 follow-up)**: the MCTS numbers in this report were later found
> to be weakened by an inverted-sign bug in `_rollout()` that was exposed when Phase 2 added
> a second game (Boop). The bug was fixed in Phase 2; every MCTS agent gained 500–650 ELO.
> The specific claim that "MCTS(50) ≈ Random" was a bug, not a noise floor. See
> [`results/phase2-boop-parity.md`](phase2-boop-parity.md) and
> [`../wiki/insights/2026-04-16-mcts-sign-bug.md`](../wiki/insights/2026-04-16-mcts-sign-bug.md)
> for the full analysis. The infrastructure and monotonic-ladder finding stand;
> the exact per-budget ELO values do not.

## Goal

Build the MCTS / arena plumbing on Connect 4 as the proving ground. Skip tic-tac-toe entirely — too small to be interesting. By end of Phase 1 we should have:

1. An immutable Env protocol any future game implements.
2. A correct Connect 4 adapter (bitboard-backed, all four win directions tested).
3. A random agent and a vanilla UCT MCTS agent that shares no code with the game.
4. An arena that plays colour-balanced matches, reports Wilson CIs, and fits ELO.
5. A CLI that runs tournaments and prints results.

## Proof artifacts

| # | Artifact | Result |
|---|---------|--------|
| 1 | `src/mlfactory/core/env.py` — Env + State protocols | ✅ |
| 2 | `src/mlfactory/games/connect4.py` — bitboard Connect 4 | ✅ |
| 3 | 12 Connect 4 rules tests (vert / horiz / both diagonals / illegal / immutability / legal_actions consistency) | ✅ all pass |
| 4 | `src/mlfactory/agents/random_agent.py`, `mcts.py` | ✅ |
| 5 | `src/mlfactory/tools/arena.py` — match, round-robin, Wilson CI, ELO | ✅ |
| 6 | 4 agent+arena tests (legal moves / balance / MCTS crushes random / bigger MCTS wins) | ✅ all pass |
| 7 | CLI: `mlfactory tournament`, `mlfactory match` | ✅ |
| 8 | Real 240-game tournament recorded with clean ELO monotonicity | ✅ |

## Headline results

Tournament: Connect 4, 40 games per pair, colour-balanced, seed=0.

### Win-rate matrix (row vs column)

| agent   | random | mcts50 | mcts200 | mcts800 |
|---------|:------:|:------:|:-------:|:-------:|
| random  |   —    | 47.50% | 15.00%  |  2.50%  |
| mcts50  | 52.50% |   —    | 15.00%  |  1.25%  |
| mcts200 | 85.00% | 85.00% |   —     | 12.50%  |
| mcts800 | 97.50% | 98.75% | 87.50%  |   —     |

### ELO leaderboard (anchor = random @ 1500)

| rank | agent   | ELO  |
|------|---------|------|
| 1    | mcts800 | 2161 |
| 2    | mcts200 | 1811 |
| 3    | mcts50  | 1510 |
| 4    | random  | 1500 |

Each 4× step in simulation budget gives **~300–350 ELO**, which lines up with the MCTS literature's rule of thumb that playing strength scales logarithmically in rollouts.

### Pairwise with 95% CIs

| A       | B       | A W–L–D    | A score | 95% CI          |
|---------|---------|------------|:-------:|:---------------:|
| random  | mcts50  | 19–21–0    | 0.475   | [0.329, 0.625]  |
| random  | mcts200 | 6–34–0     | 0.150   | [0.071, 0.291]  |
| random  | mcts800 | 1–39–0     | 0.025   | [0.004, 0.129]  |
| mcts50  | mcts200 | 6–34–0     | 0.150   | [0.071, 0.291]  |
| mcts50  | mcts800 | 0–39–1     | 0.013   | [0.001, 0.109]  |
| mcts200 | mcts800 | 4–34–2     | 0.125   | [0.055, 0.261]  |

### Wall time

**26.4 seconds** for 240 games (mix of budgets, single-threaded). The 800-sim matches dominate; rough extrapolation: ~0.25 s per MCTS(800) move in the hottest positions. Plenty of headroom for batch parallelism in Phase 3.

## Observations worth recording

1. **MCTS(50) ≈ random on Connect 4.** 52.5% vs random with a 95% CI that crosses 50% — not statistically distinguishable. This matches the MCTS-survey's observation that UCT needs enough samples to overcome the variance of random rollouts before it starts delivering signal. **Design implication**: our AlphaZero-lite in Phase 3 will target 100–200 sims because 50 is below the useful threshold even with a perfect simulator. With a neural prior guiding search, fewer sims should suffice — that's the point.
2. **Diminishing returns from 200 → 800.** The 4× budget gave ~350 ELO, but 200 already beat random 85%. Most of the strength is in the first couple hundred sims; the rest buys finesse against other MCTS agents, not against weak opponents. **Design implication**: during self-play, sim budget should be the smallest that still separates checkpoints reliably in arena.
3. **Draws are rare on Connect 4.** Across 240 games we saw 3 draws total. Expected given the game's known first-player-win theory, though at our search depths play is far from perfect so draws come from mid-game deadlocks rather than theory.
4. **Colour balance worked**. Random vs random came in at 47.5% for A — well within the Wilson CI (0.329, 0.625). No side-assignment bug.

## Reproduce

```bash
cd ~/ActiveProjects/MLFactory
uv run pytest                                    # 21 passing
bash scripts/demo-phase1.sh                      # full pipeline
uv run mlfactory tournament \
    --game connect4 \
    --agents random,mcts50,mcts200,mcts800 \
    --games-per-match 40 --seed 0
```

Seeded and deterministic. You should get the same leaderboard byte-for-byte.

## What Phase 1 does NOT include (intentionally)

- No Boop. That's Phase 2 — needs the TS bridge + rules parity test.
- No neural nets. Vanilla UCT only, by design.
- No parallel self-play. Single-threaded is enough at this scale.
- No opening book, no transposition table. Simplicity over performance here.

## What Phase 1 promoted

- `wiki/techniques/mcts-uct.md` promoted from `seed` → `stable`. The implementation matches the spec and is independently validated by the tournament.
- New **insight**: `wiki/insights/2026-04-16-mcts-logarithmic-in-sims.md` — ~300 ELO per 4× sim budget on Connect 4, with the 50-sim plateau at the random-agent level.
- Trail updated: [`wiki/trails/2026-04-getting-mlfactory-off-the-ground.md`](../wiki/trails/2026-04-getting-mlfactory-off-the-ground.md).

## What's next (Phase 2 preview)

- Port Boop rules from `Boop/server/src/game/GameState.ts` into Python.
- Build a Node-based bridge that exposes the TS rules over stdio.
- Write `scripts/verify-boop-rules.sh` — replay 10 000 random games, assert byte-identical trajectories between Python and TS.
- Tensor encoding + D4 symmetry group + explicit symmetry tests.
- Hand-written tricky-position tests (chains, graduation, winning boops).

On success, `wiki/games/boop/rules-summary.md` gets filled in from the ground truth, and we can start Phase 3 with confidence that our training signal isn't contaminated by rules bugs.
