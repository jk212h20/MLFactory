---
type: insight
status: stable
created: 2026-04-16
updated: 2026-04-16
tags: [mcts, connect4, scaling, phase-1]
links:
  experiment: results/phase1-mcts-baseline.md
  git_sha: phase-1-commit
  seed: 0
---

# ~300 ELO per 4× MCTS simulation budget on Connect 4 — with a plateau at 50 sims

## What we expected
More simulations → stronger play, monotonically. Some diminishing returns at high budgets. We did not have a sharp prior on how fast ELO grows with sim count, or on where the "useful" threshold sits for a game as simple as Connect 4.

## What we found
A clean logarithmic-ish ladder across {50, 200, 800} simulations:

| Agent    | ELO (anchor: random @ 1500) | vs random | vs mcts50 |
|----------|:---:|:---:|:---:|
| random   | 1500 | — | 47.5% |
| mcts50   | 1510 | 52.5% | — |
| mcts200  | 1811 | 85.0% | 85.0% |
| mcts800  | 2161 | 97.5% | 98.75% |

Each 4× budget step is worth ~300–350 ELO. Notable: **MCTS(50) is statistically indistinguishable from Random** (52.5% with a Wilson 95% CI of [0.37, 0.67]). The signal threshold on Connect 4 with uniform-random rollouts sits above 50 sims and below 200.

## Why this matters
1. **MCTS has a noise floor.** Below some sim count, the UCT estimator is too noisy for the selection rule to separate moves reliably; the agent plays as if random. The exact threshold depends on game variance and average rollout length. For Connect 4 + light playouts, that threshold is somewhere in (50, 200).
2. **Budget choice for Phase 3 AlphaZero-lite.** With a neural prior, we expect MCTS to be far more sample-efficient than vanilla UCT — priors focus search instead of spending sims on obviously bad moves. So targeting 100–200 sims for self-play on Boop is reasonable, *provided* the net gives meaningful priors. If it doesn't (early training), we'd be back in the noise-floor regime, and self-play games would be near-random. Mitigation: warm up with more sims while the net is weak, taper as it strengthens.
3. **Arena sanity**. Monotonic ELO + visible saturation is what "working MCTS" looks like. This result is the baseline we'll invoke later as "the infrastructure wasn't broken."

## Reproduction recipe
- Config: `uv run mlfactory tournament --game connect4 --agents random,mcts50,mcts200,mcts800 --games-per-match 40 --seed 0`
- Git SHA: to be recorded on commit.
- Seed: 0 (all RNGs derived from this).
- Command generates the full table end-to-end in ~26 s on M4 Max single-threaded.

## Implications
- Do not trust a new MCTS implementation that lacks a visible ELO ladder across sim budgets on a known game. If 50 vs 200 vs 800 look flat, the UCT/backprop is wrong.
- The "games needed for statistical significance" on Connect 4 with these agents is roughly 40 for strong gaps (mcts200+ vs random) and 100+ for narrow gaps (e.g., mcts50 vs random, mcts200 vs mcts800 head-to-head at fine resolution).
- ELO anchor choice matters only for display; rank order and gaps are stable regardless.

## Open questions this raises
- Q-006: Does the ~300-ELO-per-4×-sim law hold on Boop (smaller branching, shorter games)? Unknown.
- Q-007: What's the analogous threshold for an AlphaZero-style agent, where rollouts are replaced by a value head? Conjecture: much lower, because each visit is far more informative.
- Q-008: How does the noise-floor move when rollouts are heavier (e.g., depth-2 minimax at the leaves)? Worth a future ablation.
