---
type: insight
status: stable
created: 2026-04-16
updated: 2026-04-16
tags: [mcts, bug, sign-convention, connect4, boop, phase-2]
links:
  experiment: results/phase2-boop-parity.md
  git_sha: phase-2-commit
  seed: 0
---

# MCTS had an inverted-sign bug in the random-rollout branch that was half-hidden on Connect 4 and fully exposed by Boop

## What we expected
Phase 1 MCTS was validated on Connect 4 with a clean ELO ladder (random / mcts50 / mcts200 / mcts800 → 1500 / 1510 / 1811 / 2161). We expected the same monotonicity to hold on Boop.

## What we found
Initial Phase 2 Boop tournament: **MCTS lost badly to Random**.

```
            random  mcts50  mcts200
random      -       85%     100%
mcts50      15%     -       85%
mcts200     0%      15%     -
```

Not a small effect — completely inverted ladder. MCTS(200) went 0-20 vs Random.

## Root cause
A sign bug in `mcts.py::_rollout`. Two code paths returned a value for "reward for mover-into-node":

1. **Terminal node** (never actually in a rollout; only reached as the rollout entry point when expansion landed on a terminal leaf): used `return -env.terminal_value(state)`. **Correct.**
2. **After random playout**: used `return 1.0 if state.winner == node.to_play_at_entry else -1.0`. **Inverted.**

`node.to_play_at_entry == node.state.to_play` = the player who will move *from* this node. The **mover-into-node** is the OTHER player: `1 - node.to_play_at_entry`. The correct formula is `1.0 if state.winner == (1 - node.to_play_at_entry) else -1.0` — equivalently, `state.winner != node.to_play_at_entry`.

## Why Connect 4 "worked" anyway — the honest story

I initially hypothesised "short games let the correct terminal-branch fire often enough." That was **wrong**. Instrumentation showed 100% of MCTS iterations on both Connect 4 and Boop take the random-playout branch. The terminal-entry branch essentially never fires (it would require selection+expansion to land on a state that was already terminal, which our tree doesn't add as children).

So both games were subject to the inverted sign in 100% of their rollouts. Why did Connect 4 still reach 92.5% vs Random while Boop went 0-20?

I genuinely don't have a clean answer. Some factors that likely contribute:

1. **Final move selection is by visit count, not value** (`max(root.children, key=lambda n: (n.visits, n.mean_value))` — "robust child"). Visit counts reflect where UCT focused search, and UCB's exploration term can make UCT visit the "correct" node often even when exploitation points the wrong way. Visit counts are a partial fix.
2. **UCB exploration pushes visits onto under-visited children regardless of value sign**. The exploration term dominates when visit counts differ widely, so even inverted exploitation doesn't fully prevent balanced exploration early on.
3. **Opponent modelling via symmetry**: in zero-sum games, "pick the best move for me" and "pick the worst move for opponent" are equivalent. With inverted values, UCT at root picks what it thinks is opponent-best = me-worst; but deeper in the tree, the opponent's errors may partially cancel.
4. **Boop's long games + larger branching factor** mean UCT has many more levels of potential sign-inconsistency to corrupt, and the sparse-reward signal from 57-move rollouts compounds the error.

The honest summary: **an inverted exploitation term in UCT is a real bug that degrades strength substantially, but "robust child" final selection plus UCB exploration partially mask it. The degree of masking is game-dependent and hard to predict from first principles.**

Post-fix numbers (what good MCTS actually looks like):

| Agent | ELO (buggy MCTS) | ELO (fixed MCTS) |
|-------|:---:|:---:|
| random | 1500 | 1500 |
| mcts50 | 1510 | 2158 |
| mcts200 | 1811 | 2446 |
| mcts800 | 2161 | 2673 |

The fix added **~500–650 ELO per MCTS agent** across the board. MCTS(50) was "statistically indistinguishable from random" with the bug; with the fix, it beats Random 39-1 (97.5%). Phase 1's "MCTS(50) ≈ Random" insight has to be retracted — that was a bug, not a noise floor.

## Why Boop exposed the bug
Boop games average 57 moves and have branching factor ~36 early, ~30 mid-game (vs Connect 4's ~7). Long rollouts + wider trees = more compounding of the sign error. The "robust child" mitigation that partly saved Connect 4 couldn't save Boop. Exact mechanism is not fully understood; what matters is the empirical signature: inverted UCT breaks badly on long-game, high-branching-factor settings.

**One hypothesis worth noting (from the user)**: could the bug be game-specific via the "win on opponent's turn" rule? In Boop, gray's placement could theoretically boop orange's cats into a winning configuration, but the TS (and our port) only check win for the mover. So that rule doesn't actually trigger during normal play. I confirmed the bug is NOT about this — it's a pure sign inversion that affects all games. But the observation is worth recording: **games where wins can happen on the "wrong" player's turn would have additional perspective subtleties we'd need to handle**. That's a future consideration, not what happened here.

## Lesson for Phase 3
When a change "works" but "doesn't work as well as expected," suspect a silent sign / perspective bug. We will:

1. **Test MCTS on at least TWO games** before trusting it (one short/tactical, one long/positional). Connect 4 alone was not enough.
2. **Sanity-check obvious wins**: MCTS must find a 1-move win 100% of the time at modest sim budget. We now have this test for Boop and Connect 4.
3. **Record expected ranges**: MCTS(N) vs Random should scale roughly log(N) ELO. If the first few data points plateau, something is broken.
4. **Distrust large-margin "success" results on simple games**: 97% vs Random on Connect 4 seemed strong; post-fix it's 100% with MCTS(200). A 3% margin was the bug's thumbprint.
5. **Robust child is a real safety net**: final move selection by visit count (not value) partially masks value-sign bugs. Don't let that mask hide real bugs. Cross-check visit distributions against raw rollout statistics periodically.
6. **Games where wins can occur on either player's turn** need extra care. Boop could (in theory, via cat-boops-cat rearrangements) let gray's placement create a winning line for orange, though the existing TS rule only checks win for the mover so this is ignored. If we ever add a game where off-turn wins matter, the MCTS sign convention must be re-audited.

## Reproduction recipe
- Bad commit: `153dedf` (Phase 1). `uv run mlfactory tournament --game connect4 --agents random,mcts50,mcts200,mcts800 --games-per-match 40 --seed 0` shows mcts50 at 1510 ELO.
- Fixed commit: (Phase 2 commit). Same command now gives mcts50 at ~2158 ELO.
- Fix: `src/mlfactory/agents/mcts.py` `_rollout()` — change `node.to_play_at_entry` comparison to use `1 - node.to_play_at_entry` (mover-into-node), and make both terminal-branch and rollout-terminal paths consistent.

## Implications
- **Retroactively update Phase 1 results**: the "mcts50 ≈ random" finding in `insights/2026-04-16-mcts-logarithmic-in-sims.md` was entirely due to this bug. The real MCTS(50) is ~650 ELO above random.
- **Connect 4 ELO law stands, shifted**: the "~300 ELO per 4× sim budget" rule is still visible post-fix (mcts50 → mcts200 → mcts800 gives 2158 → 2446 → 2673, which is 288 and 227 ELO — tighter than before but still increasing).
- **Sign conventions must be documented in one place**. `mcts.py` now has a clear docstring.

## Open questions this raises
- Q-009: Are there similar perspective bugs lurking in arena / ELO calculation? We should audit.
- Q-010: What's the right "smoke test" for a new agent that catches this class of bug immediately? Obvious-win tests? Self-play symmetry tests?
