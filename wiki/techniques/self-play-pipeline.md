---
type: technique
status: seed
created: 2026-04-16
updated: 2026-04-16
tags: [self-play, alphazero, training-loop]
links:
  sources: [silver2017-alphago-zero, anthony2017-expert-iteration, silver2017-alphazero]
  answers: []
---

# The AlphaZero-style Self-Play Pipeline

## What it does
A training loop where the current neural network plays games against itself using MCTS guided by its own priors, stores `(state, improved_policy, game_outcome)` triples, and trains the network on those triples to match the improved policy and outcome. Repeats forever.

## When to use
- 2-player zero-sum perfect-information deterministic games (Go, Chess, Shogi, Hex, Boop, Connect 4…).
- When you have a perfect simulator.
- When you can batch many self-play games concurrently to amortize NN forward passes.

## When NOT to use
- Imperfect information (poker, Duke-style hidden info) → needs counterfactual regret minimization (CFR) family or Player-of-Games.
- Real-time / continuous-action games → AlphaZero's discrete-move assumption breaks.
- Single-agent RL problems (Atari, control) → use PPO/SAC/etc. or MuZero if you need a learned model.

## Pattern

```
loop forever:
    # --- self-play workers (many games in parallel) ---
    for game in parallel_games:
        s = env.reset()
        history = []
        while not s.terminal:
            pi = MCTS(s, net, n_sims, add_root_dirichlet_noise=True)
            a = sample_from(pi, temperature(move_number))
            history.append((s, pi))
            s = env.step(a)
        z = s.outcome                         # +1 / -1 / 0

        for (s_t, pi_t) in history:
            z_from_s_t = z * (+1 if to_play(s_t) == winner else -1)
            replay_buffer.push((s_t, pi_t, z_from_s_t))

    # --- trainer (continuous or periodic) ---
    for step in range(train_steps):
        batch = replay_buffer.sample(batch_size)      # may apply symmetry augmentation
        loss = CE(pi_target, net.policy) + MSE(z, net.value) + L2(net.params)
        loss.backward(); optimizer.step()

    # --- optional: arena gate (AlphaGo Zero) ---
    if gating:
        if arena(new_net, best_net, n_games=400).win_rate(new_net) > 0.55:
            best_net = new_net
        else:
            new_net = best_net    # roll back
    # AlphaZero skips the gate and uses new_net immediately.
```

## Key design decisions, with defaults
- **MCTS sims per self-play move**: 800 (AlphaGo Zero scale) → 100–200 is a reasonable start for small games. Ablate.
- **Temperature schedule**: τ = 1 for first N moves (exploration), then τ → 0 (greedy). N = 30 in AlphaGo Zero Go, but should scale to game length — e.g., ~15–25% of typical game length. For Boop (short games), N = 8–10 is a guess.
- **Root Dirichlet noise**: `α = 10 / |legal_moves|` is a common heuristic, `ε = 0.25`. Only added at the root, only during self-play.
- **Replay buffer size**: the last N self-play games (AGZ: 500k). At hobby scale, measure in positions not games; start at 50k–200k positions and adjust based on diversity.
- **Symmetry augmentation**: for square boards, apply the dihedral group D4 (8 elements) to position+policy at training time. Free 8× data multiplier.
- **Batch size / training pace**: train roughly 1 gradient step per N self-play positions; AGZ used ~1 step per 2 positions. At hobby scale, lean toward more training per position (data is expensive).
- **Gating vs continuous**: default continuous (AlphaZero). Re-add gating if training destabilises.

## Common pitfalls
- **Policy collapse**: net convergences on a narrow set of moves, MCTS visit counts stop being informative. Fix: more Dirichlet noise, higher temperature, larger replay buffer.
- **Stale replay**: replay buffer much larger than net's changing distribution → net trains on bad targets. Fix: smaller buffer or use priority by recency.
- **Value head learns before policy head**: loss ratio matters. Tune relative weights or use AlphaGo Zero's unweighted sum.
- **Symmetry bugs**: applying rotations to positions but forgetting to rotate the corresponding policy target → silent accuracy drop. Test explicitly.
- **MCTS using mutable state**: if your env mutates state rather than returning a new one, tree search corrupts shared state. Always clone or use immutable states.
- **Not balancing colour**: in arena evaluation, play both sides equally; otherwise a side-strong asymmetry poisons your ELO.

## Implementation in MLFactory
_(Phase 3 will implement this. File path: `src/mlfactory/agents/alphazero/trainer.py` orchestrates the loop, with `selfplay.py` workers and `replay_buffer.py`.)_

## Sources
- [[sources/silver2017-alphago-zero]] — the canonical recipe.
- [[sources/silver2017-alphazero]] — generalises it and drops the arena gate.
- [[sources/anthony2017-expert-iteration]] — independent derivation; shows it works at hobby-relevant scales (Hex 9×9/13×13).

## See also
- [[techniques/mcts-uct]] — what MCTS is without priors.
- [[techniques/puct-with-priors]] — the search side of this pipeline. _(to be written.)_
- [[techniques/symmetry-augmentation]] — the D4 trick. _(to be written.)_
