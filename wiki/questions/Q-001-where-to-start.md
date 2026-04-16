---
type: question
status: stable
created: 2026-04-16
updated: 2026-04-16
tags: [process, onboarding, alphazero, mcts]
links:
  sources: [silver2017-alphago-zero, silver2017-alphazero, anthony2017-expert-iteration, browne2012-mcts-survey, schrittwieser2020-muzero]
  uses: []
confidence: high
---

# Q-001 — Given that MLFactory's first target game is Boop on an M4 Max, what algorithm and starting architecture should we aim for?

## Context
Boop is a 2-player perfect-information deterministic game on a 6×6 board with ~72-way action space (piece-type × position), moderate branching factor, short games, and D4 symmetry. We have a perfect simulator (Boop's own TS implementation). We want to build an agent strong enough to beat the existing `DeepThinker` heuristic bot (75.6% in its tournament) and ultimately the human author. Hardware: Apple M4 Max, 128 GB unified memory, PyTorch on MPS.

The design space is wide: vanilla MCTS, AlphaZero, MuZero, ExIt, Player-of-Games, and various combinations. Which is the right starting point?

## Sources consulted
- [[sources/silver2017-alphago-zero]] — AlphaGo Zero: tabula-rasa self-play with MCTS + combined policy/value ResNet, on 19×19 Go. The canonical recipe for perfect-info board games.
- [[sources/silver2017-alphazero]] — AlphaZero: the same algorithm generalised to Go/Chess/Shogi with a single set of hyperparameters. Drops the best-player gate.
- [[sources/anthony2017-expert-iteration]] — ExIt: contemporaneous independent derivation, validated on Hex 9×9/13×13 — scale closer to Boop's than Go. Strong evidence the recipe works at hobby scales.
- [[sources/browne2012-mcts-survey]] — UCT and MCTS variants. Sets the vocabulary and baseline.
- [[sources/schrittwieser2020-muzero]] — MuZero: learned dynamics model in latent space. Relevant only when rules aren't available; not our situation.

## Answer

**Target: AlphaZero-lite, built on top of vanilla UCT MCTS.**

**Starting architecture (hypothesis, to be ablated):**
- Body: **4 residual blocks, 64 channels**, 3×3 convs with batchnorm + ReLU.
- Policy head: 1×1 conv (2 filters) → BN → ReLU → FC → softmax over `|A| = 72` (+ whatever Boop's graduation action needs); mask illegal actions.
- Value head: 1×1 conv (1 filter) → BN → ReLU → FC(64) → ReLU → FC(1) → tanh.
- Input: 6×6 spatial planes for {my kittens, my cats, opp kittens, opp cats, legal-action mask, colour-to-move}. Possibly add 2–4 steps of history.
- Loss: `(z - v)^2 + CE(pi_MCTS, pi_net) + 1e-4 * ||θ||^2`.

**Starting search:**
- PUCT (AlphaGo Zero formula), `c_puct ≈ 5`, Dirichlet noise `α ≈ 10/|legal|`, `ε = 0.25` at root during self-play only.
- **100–200 MCTS sims/move** during self-play (not 800 — Boop is small).
- Temperature `τ = 1` for the first ~8 moves, then `τ → 0`.

**Starting training loop:**
- Gate-less (AlphaZero style) — simpler, fewer knobs.
- Replay buffer: **last 50k–100k positions**, sampled uniformly.
- 8× D4 symmetry augmentation of spatial planes + spatial policy targets (piece-type dims left alone).
- Training cadence: ~1 gradient step per 2–4 self-play positions (AlphaGo Zero was closer to 1:2).

**Build order**:
1. Vanilla UCT MCTS that decisively beats Random on toy games (Phase 1).
2. Boop rules ported and parity-verified against the TS reference (Phase 2).
3. AlphaZero-lite training until it beats MCTS at same search budget (Phase 3 start).
4. Scale up: more sims, bigger replay, longer training. Beat DeepThinker, then the human (Phase 3 goal).
5. Consider solver / proof-number search / endgame tablebases (Phase 4).

## Confidence
**High** that this is the right recipe for Boop. The ExIt paper demonstrates that the AlphaZero template works on boards this scale without industrial compute; the AlphaGo Zero architecture is well-documented and easy to shrink. The only moderate-confidence call is the specific net size (4 blocks × 64) — this is a folklore guess from small-board reproductions, not a number pulled from a cited paper. We should ablate {2, 4, 8} blocks and {32, 64, 128} channels once training is stable.

**Low confidence** in: MCTS sim count (100–200 is a guess; measure self-play strength vs budget), temperature horizon (first 8 moves is a guess for Boop's game length), specific replay window size.

## What this changed
- Locks in Phase 3's target stack: AlphaZero-lite, not MuZero, not CFR-family.
- Phase 1 focuses on vanilla UCT MCTS as a genuine stepping stone, not throwaway plumbing.
- Confirms Boop is the right first target (small enough to converge, unstudied enough that results will be interesting).
- Motivates explicit symmetry tests in Phase 2 so Phase 3's augmentation is trustworthy.

## Follow-up questions
- Q-002: Should we use AlphaGo Zero's best-player gating or AlphaZero's gate-less continuous training at our scale? _(sketched, not yet written)_
- Q-003: How do we verify the Python rules port is identical to the TypeScript reference without playing through every edge case? _(will write during Phase 2)_
- Q-004: What's the right policy-head factoring for Boop's (piece-type, position) action space — flat 72-way softmax, or two separate heads?
- Q-005: At hobby-scale compute, what replay buffer size (in positions) is appropriate, and should we weight by recency?
