---
type: source
status: stable
created: 2026-04-16
updated: 2026-04-16
tags: [expert-iteration, exit, mcts, policy-distillation, hex]
provenance:
  url: https://arxiv.org/abs/1705.08439
  accessed: 2026-04-16
  read_level: abstract
speculation: false
---

# Anthony, Tian, Barber 2017 — Expert Iteration (ExIt)

## One-sentence claim
RL can be decomposed into a tree-search "expert" that plans and a neural-network "apprentice" that generalises those plans, with the two bootstrapping each other; trained tabula rasa on Hex, the agent defeats MoHex 1.0 (then the most recent publicly-released Hex Olympiad champion).

## Bibliographic
- Authors: Thomas Anthony, Zheng Tian, David Barber
- Venue/year: NeurIPS 2017; arXiv:1705.08439 (v1 May 2017, v4 Dec 2017)
- Link: https://arxiv.org/abs/1705.08439 ; https://proceedings.neurips.cc/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html

## Core idea
Two systems in the spirit of Kahneman's *Thinking, Fast and Slow*:

- **Apprentice (fast / System 1)**: a deep neural network π̂(a|s) (and, from v2, a value head V̂(s)). Cheap forward pass; intuitive pattern-matching over board states.
- **Expert (slow / System 2)**: a tree search (MCTS) that, given time, produces a stronger policy π* via explicit lookahead.

Teaching loop in both directions:

1. **Expert teaches apprentice (distillation)**: apprentice trained (supervised) to match expert's improved policy at visited states — distill MCTS visit distributions into the network.
2. **Apprentice teaches expert (search guidance)**: apprentice's policy (and value) biases the tree search — pruning, warm-starting priors, replacing rollouts with value estimates — so the next expert round is stronger.

Each cycle strengthens the expert, so the apprentice has a better target next iteration. Unlike plain deep RL, the network never has to *discover* good play on its own; it only has to *generalise* plans the search already found.

## Algorithm in pseudocode
```
initialize apprentice π̂_0  (random)
for i = 0, 1, 2, ...:
    D_i = {}
    for many self-play games:
        at each state s visited:
            π*(·|s) = MCTS(s, guided by π̂_i, V̂_i)   # expert move
            play move sampled from π*(·|s)
            store (s, π*(·|s), game_outcome z) in D_i
    π̂_{i+1}, V̂_{i+1} = train(D_i, loss = KL(π* ‖ π̂) + (z - V̂)^2)
```

## Game used for experiments
**Hex** on 9×9 and 13×13 boards. Perfect information, 2-player, deterministic, no draws, moderate state-space complexity — **closer in flavour to Boop** than Go (19×19, huge B) or Chess (asymmetric pieces, tactical complexity). Headline: trained agent defeated **MoHex 1.0**, then the most recent publicly released Hex Olympiad champion.

## Direct quotes (from abstract)
> "Sequential decision making problems, such as structured prediction, robotic control, and game playing, require a combination of planning policies and generalisation of those plans."

> "In this paper, we present Expert Iteration (ExIt), a novel reinforcement learning algorithm which decomposes the problem into separate planning and generalisation tasks."

> "Planning new policies is performed by tree search, while a deep neural network generalises those plans. Subsequently, tree search is improved by using the neural network policy to guide search, increasing the strength of new plans."

> "We show that ExIt outperforms REINFORCE for training a neural network to play the board game Hex, and our final tree search agent, trained tabula rasa, defeats MoHex 1.0, the most recent Olympiad Champion player to be publicly released."

## Relation to AlphaZero
ExIt and AlphaGo Zero / AlphaZero (Silver et al., 2017) are contemporaneous and arrive at the same algorithmic template independently; v3 of the ExIt paper explicitly clarifies independence from AGZ. Shared core:

- Self-play as the data source.
- MCTS as the policy-improvement operator.
- Supervised distillation of MCTS visit counts into a neural network policy.
- Bootstrapped value function from game outcomes.
- Network priors guiding subsequent MCTS.

AlphaZero added / specified:
- **One net, two heads**: a single residual conv net with policy + value heads.
- **PUCT** selection with explicit exploration constant.
- **Dirichlet noise** at the root each game for exploration.
- **Rollout-free** MCTS: leaves evaluated purely by value head.
- **Scale**: thousands of TPUs.
- **Generality**: Go / Chess / Shogi with no domain-specific features.

Short: **AlphaZero ≈ ExIt + residual 2-headed net + PUCT + Dirichlet noise + no rollouts + industrial compute.**

## Numbers worth remembering
- Board sizes: **9×9** and **13×13** Hex.
- Headline: beat **MoHex 1.0**.
- Benchmark: ExIt > **REINFORCE** on Hex policy training.
- Specific ELO, MCTS sim counts, net dims: **not extracted** from abstract.

## Our open questions
- MCTS simulation budget per move during training vs evaluation?
- Self-play / iteration counts to beat MoHex on 9×9 vs 13×13?
- Rollouts at leaves or value-net bootstrap? (v2 added a value function.)
- Distillation loss: cross-entropy against visit-count distribution, or against argmax?
- Sensitivity to apprentice capacity — does a small net still bootstrap, or cap expert quality?
- Any curriculum over board size (9×9 → 13×13 transfer)?

## How this maps to MLFactory
- **Why ExIt fits Boop especially well**: Boop is 6×6 with moderate branching — exactly the regime ExIt was validated in. We do NOT need AlphaZero-scale compute; modest MCTS (100s–1000s sims) + small conv net should suffice. **This is our main encouragement that a hobby-scale M4 Max run can reach strong play.**
- **Distillation target**: MCTS visit distributions over Boop's move set (placement + kitten/cat choice) map cleanly onto a categorical policy head. Graduation mechanic is extra action structure — encode as flat or factored? → future question.
- **Apprentice network size**: small-board ExIt-flavored reproductions typically use residual tower on the order of **4–10 blocks at width 64–128** (folklore, not directly from this paper — treat as hypothesis).
- **Practical loop**: start with very shallow MCTS (50–200 sims) while apprentice is weak so the expert still beats it meaningfully; scale sims up as the net improves. Keeps the teaching signal positive.
- **Open**: Boop's short games and knockback dynamics — enough signal per move, or add auxiliary targets (next-state prediction) to stabilize small-data regime?
