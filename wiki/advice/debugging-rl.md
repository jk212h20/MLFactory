---
type: advice
status: draft
created: 2026-04-16
updated: 2026-04-16
tags: [debugging, rl, self-play]
links:
  sources: []
---

# Debugging RL and Self-Play

RL debugging is uniquely painful because everything is probabilistic, asynchronous, and plausibly broken in silence. Here's a checklist before you blame the algorithm.

## Sanity checks that cost nothing

1. **Random agent self-play terminates.** Seed-determined trajectory length distribution is stable.
2. **Random vs Random ≈ 50%** over 1000 games. Any sharp deviation is a bug, not stats.
3. **MCTS(100) beats Random ≥95%.** If not, your MCTS or rewards are broken.
4. **MCTS(800) draws or beats MCTS(100)** in ≥60% of games. If not, your search has a bug.
5. **Symmetry test**: `env.step(symmetry(s), symmetry(a)) == symmetry(env.step(s, a))` for all games' native symmetries. Fails here mean encoding bugs that silently ruin augmentation.
6. **Reward sign**: after a game, check that the winning side's reward is +1 in every stored tuple where they were to move, -1 where they weren't.
7. **Legal-move mask**: sample 100 random states, verify the mask matches `env.legal_actions(s)`.
8. **Net forward on MPS works**: compare a forward pass on CPU vs MPS for identical weights and input. Max absolute diff should be < 1e-4.

## When training seems broken

### Loss going down, ELO not improving
- Distribution mismatch: net is fitting old self-play data that no longer reflects its own current policy. Shrink replay buffer.
- Arena opponents too weak or too strong: use a range of prior checkpoints as opponents and colour-balance.
- Policy collapse: visit-count distribution has collapsed onto a few moves; MCTS isn't finding new lines. Increase Dirichlet noise or temperature.
- Value head saturated: outputs always near +1 or -1, gradients vanish. Check `tanh` range and label balance (are most training games won by side-to-move?).

### Loss not going down
- Learning rate too high: NaN grads? Check gradient norms.
- Bad target: verify a single (state, pi, z) triple by eye — does the MCTS policy look reasonable? Does z match who won?
- Wrong input plane: off-by-one between "my pieces" and "opponent pieces" is brutal and silent.

### MCTS looks dumb
- Prior not being used: check PUCT formula — exploration term should use prior `P(s,a)`. If your `P` is uniform everywhere, your net isn't being queried.
- Reward not propagating: put print statements at backprop. Root visits should equal `n_sims`; its Q-values should move across iterations.
- Mutable state: each simulation must clone from the root, not play on the live game state.

### Self-play deadlocks or crashes
- Tree shared across workers: either use per-worker trees (root parallelism) or add virtual loss + locks.
- Memory grows unboundedly: the tree isn't being cleared between moves, or you're retaining computation graphs. Detach tensors.
- NN queries pile up: batch them across multiple simulations / workers.

## Evaluating "did it actually get better"

- Always play **both colours** equally. Report (W, L, D) from each side separately. If asymmetric, something is off.
- Use **many opponents**, not just "yesterday's net": a pool covering several training epochs + a MCTS baseline + a Random baseline. ELO over a pool is far more stable than pairwise matches.
- **Confidence intervals**: 100-game matches are noisy. A 55/45 split is not statistically significant. Report `p(A ≥ 55%)` from a Beta posterior.
- **Seed multiple runs**. A single 500-game training run shouldn't be trusted; three with different seeds is the minimum for any claim.

## Instrumentation that earns its keep

- Arena ELO over time, computed from a fixed opponent pool.
- Policy entropy per move. Crashes to ~0 → collapse.
- Value head calibration: bin predicted value, measure actual outcome frequency.
- Top-k policy overlap between net and MCTS at inference: if very different, the net isn't learning the search target.
- Self-play game-length distribution. Sudden drops or spikes indicate strategy changes worth investigating.

## When all else fails

Run the smallest possible version. Tic-tac-toe with a 2-block net should train to perfect play in minutes. If it doesn't, your infrastructure is broken; your Boop code won't work either.

## Psychological rules

- Fix one thing at a time.
- Write down your hypothesis **before** you check the evidence.
- Every training run is an experiment with a prediction. Record both in `experiments/<run>/report.md`.
- If you're about to say "it's probably just variance", run more seeds first.
