---
type: advice
status: draft
created: 2026-04-16
updated: 2026-04-16
tags: [process, onboarding]
links:
  sources: []
---

# Getting Started on a New Game

How to approach a new game in MLFactory without getting lost in the weeds.

## The ladder

Climb rung by rung. Do not skip.

1. **Rules in code + tests**. Port the rules to `games/<game>/rules.py`. Write explicit tests for: initial state, every terminal condition, every tricky rule (capture, promotion, repetition, …). If the game has a reference implementation (TS, pygame, etc.), set up a parity test that replays 10k random games and asserts identical trajectories.
2. **Random agent runs**. Random vs random should terminate every time. If it hangs, you have an infinite-loop bug in the rules — usually move repetition or missing terminal check.
3. **Arena with win-rate matrix**. Random vs random should be ~50% each (± noise) and 0% asymmetric. If it's 55/45, you have a side bug or a colour-swap bug.
4. **MCTS baseline**. Vanilla UCT should beat Random ≥ 95% at modest budget (100–800 sims). If it doesn't, either the reward signal is broken or the game has extreme variance.
5. **Symmetry map**. Work out the game's symmetry group (usually D4 for square boards, smaller for asymmetric). Test: for a random state, applying a symmetry to both state and action produces a state equal to applying the action then the symmetry (i.e., symmetries commute with the step function). This catches silent bugs.
6. **State/action encoding**. Design tensor encoding. Spatial planes for position-dependent info; global planes for turn, move count, etc. Action space: flat or factored. Legal-move mask is mandatory.
7. **Tiny AlphaZero**. 2–4 residual blocks, 32–64 channels. Train for an hour. Must beat MCTS at the same sim budget. If it doesn't, something is structurally wrong; don't scale up.
8. **Scale**. Only now. Bigger net, more sims, longer training.

## Red flags to watch for

- **Random vs random ≠ 50%**: colour bug, reward sign bug, or the game is genuinely asymmetric (in which case document the draw/win counts from the true equilibrium).
- **MCTS doesn't beat random**: something in the reward or game-end plumbing.
- **Value head loss plateaus at ~0.5 MSE and policy CE barely moves**: policy collapse. More exploration noise, check MCTS targets.
- **Training loss goes down but arena ELO doesn't improve**: distribution mismatch — net overfits to its own old moves. Shorten replay or add diversity.

## The "just play it yourself" test

Before you train anything, play the game yourself against a random agent. Notice:
- How long does a typical game last? (informs move-cap and replay sizing)
- What's the branching factor mid-game? (informs MCTS sims and progressive widening choice)
- Are there obvious heuristics? (informs what a decent baseline bot should do)
- Do you find sharp tactical moments? (if yes, expect UCT traps — argue for neural guidance sooner)

Write this down in `wiki/games/<game>/rules-summary.md`.

## Time-box each rung

If you spend more than one session stuck on a rung, escalate: write a question page, consult sources, consider that the game needs something non-standard (imperfect info? stochastic? needs CFR?).

## The best evidence you're making progress

- New green tests.
- Green arena match vs yesterday's agent.
- A new `wiki/insights/YYYY-MM-DD-*.md` entry.
- A question you answered that future-you would have asked.

If a day passed without any of those, you either needed a break or you're polishing instead of progressing.
