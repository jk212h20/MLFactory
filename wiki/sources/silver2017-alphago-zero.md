---
type: source
status: stable
created: 2026-04-16
updated: 2026-04-16
tags: [alphago-zero, self-play, mcts, residual-network, go]
provenance:
  url: https://www.nature.com/articles/nature24270
  accessed: 2026-04-16
  read_level: abstract
speculation: false
---

# Silver et al. 2017 — AlphaGo Zero

## One-sentence claim
Tabula-rasa self-play with a single combined policy+value residual network and PUCT search surpasses all prior Go programs — no human games, no handcrafted features beyond board + rules.

## Bibliographic
- Authors: David Silver, Julian Schrittwieser, Karen Simonyan (equal contribution), Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, Demis Hassabis.
- Venue/year: Nature 550 (7676), 354–359, published 19 October 2017.
- DOI: 10.1038/nature24270
- Links:
  - Nature: https://www.nature.com/articles/nature24270 (paywalled)
  - UCL green OA accepted manuscript: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf

## Why this matters for MLFactory
AlphaZero ([[sources/silver2017-alphazero]]) is AlphaGo Zero generalised across games. **AlphaGo Zero is the cleanest reference for 2-player perfect-information board games — exactly our setting for Boop.** The algorithmic recipe is the canonical blueprint we are re-implementing.

## Architecture (Methods-section details, widely documented; verify line-by-line before citing specific numbers)
- Input: stacked board planes encoding current-player stones, opponent stones, and last 8 half-moves (17 planes for 19×19 Go: 8 own + 8 opp + 1 colour-to-move).
- Residual tower:
  - "20-block" model (3-day training): 20 residual blocks.
  - "40-block" model (strongest, 40-day training): 40 residual blocks.
  - Each block: conv(256 filters, 3×3) → BN → ReLU → conv(256, 3×3) → BN → skip-add → ReLU.
  - Shared body feeds two heads.
- Policy head: 1×1 conv (2 filters) → BN → ReLU → FC → softmax over 19×19 + 1 (pass) = 362 logits.
- Value head: 1×1 conv (1 filter) → BN → ReLU → FC(256) → ReLU → FC(1) → tanh → scalar in [-1, 1].

## PUCT formula
At each in-tree node `s`, selection picks the action maximising:

    a* = argmax_a  [ Q(s, a) + U(s, a) ]
    U(s, a) = c_puct * P(s, a) * sqrt( Σ_b N(s, b) ) / ( 1 + N(s, a) )

Where:
- `Q(s, a)` = mean value of simulations taking action a from s (W/N).
- `N(s, a)` = visit count; `P(s, a)` = prior from the policy head.
- `c_puct ≈ 5` (constant; paper notes slight schedule tuning).
- At root: `P(s_0, a) = (1-ε) * p_a + ε * η_a`, with `η ~ Dir(0.03)` and `ε = 0.25` for Go (α scales inversely with legal-move count).

## Training details
- Self-play: 4.9M games for 20-block / 3-day run; 29M games for 40-block / 40-day run.
- MCTS: 1,600 simulations per move during self-play.
- Temperature: τ = 1 for the first 30 moves (sample ∝ N(s,a)^(1/τ)); τ → 0 (argmax) afterward.
- Mini-batches of 2,048 positions sampled uniformly from the last 500,000 self-play games (replay buffer).
- Optimiser: SGD, momentum 0.9, L2 c = 1e-4. LR stepped 1e-2 → 1e-3 → 1e-4.
- Loss: `L = (z − v)^2 − π^T log p + c ||θ||^2`.
- Data augmentation: 8× D4 dihedral symmetry applied per training step.
- Evaluation (best-player gate): candidate plays 400 games vs current best; must win ≥55% to be promoted as the self-play generator. **AlphaZero drops this gate** and uses a continuously updated net.

## Direct quotes (from Nature abstract)
> "A long-standing goal of artificial intelligence is an algorithm that learns, tabula rasa, superhuman proficiency in challenging domains."

> "Here we introduce an algorithm based solely on reinforcement learning, without human data, guidance or domain knowledge beyond game rules."

> "AlphaGo becomes its own teacher: a neural network is trained to predict AlphaGo's own move selections and also the winner of AlphaGo's games."

> "Starting tabula rasa, our new program AlphaGo Zero achieved superhuman performance, winning 100–0 against the previously published, champion-defeating AlphaGo."

## Numbers worth remembering
- 100–0 vs AlphaGo Lee.
- 1,600 MCTS sims / move during self-play.
- 4.9M self-play games (20-block) → surpassed AlphaGo Lee after ~36 h.
- 17 input planes.
- Loss: equal-weight value MSE + policy CE + L2 c = 1e-4.
- Dirichlet noise: α = 0.03, ε = 0.25.
- c_puct ≈ 5.
- Best-player gate: 55% over 400 eval games.
- 8× D4 symmetry augmentation.

## Our open questions
- Is the 55% gate worth the complexity at hobby-scale compute, or should we copy AlphaZero's gate-less single-net approach from the start? → **Q-002** on this.
- What batch size / replay window is appropriate when self-play throughput is 3–4 orders of magnitude lower than DeepMind's? Is "last 500k games" better expressed as "last N positions" at our scale?
- For Boop, how to represent turn history when piece placement + piece-type + 3-in-a-row capture mechanics interact? Is any history informative vs. just the current position?
- Does symmetry augmentation help or hurt when the policy head must predict piece *type* as well as position (symmetry acts on position but not on type)?

## How this maps to MLFactory
- **Action space**: Boop is (piece_type ∈ {kitten, cat}) × (position ∈ 6×6) = 72 actions, plus graduation choices. Policy head should be flat softmax over the full action set, masked by legality at both MCTS prior and loss time.
- **Architectural template**: copy AlphaGo Zero's body + two heads; just swap |A| and shrink the tower.
- **Symmetry augmentation**: D4 applies to spatial planes and to the *spatial* part of the policy target; piece-type channel is symmetry-invariant. Apply rotations/reflections to spatial planes + spatial policy; leave piece-type dims alone.
- **History planes**: probably reduce from 8 to 2–4 given Boop's shorter games; revisit empirically.
- **PUCT + Dirichlet noise**: copy verbatim. c_puct ≈ 5; α ≈ 10/|legal_moves|; ε = 0.25 at root.
- **Best-player gating**: start without it (AlphaZero style). Add back only if training destabilises.
- **MCTS sims/move**: 1,600 is out of reach at hobby scale; start at 50–200 and ablate.
