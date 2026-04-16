---
type: source
status: stable
created: 2026-04-16
updated: 2026-04-16
tags: [alphazero, self-play, mcts, deep-rl, games]
provenance:
  url: https://arxiv.org/abs/1712.01815
  accessed: 2026-04-16
  read_level: abstract
speculation: false
---

# Silver et al. 2017 — AlphaZero

## One-sentence claim
A single general reinforcement-learning algorithm (AlphaZero), starting tabula rasa from only the rules of the game and learning via self-play, reaches superhuman play in chess, shogi, and Go and defeats a world-champion program in each.

## Bibliographic
- Authors: David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis
- Venue/year: arXiv:1712.01815 [cs.AI], submitted 5 Dec 2017 (preprint of the later Science 2018 paper "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play")
- Link: https://arxiv.org/abs/1712.01815

## Key takeaways
- AlphaZero is a *single* algorithm (same hyperparameters, architecture, and training procedure) applied to three distinct games; generality is the headline claim.
- It is trained *tabula rasa*: no opening books, no handcrafted evaluation, no human games — only the rules and self-play.
- It contrasts itself explicitly with the dominant chess paradigm (alpha-beta search + handcrafted evaluation refined over decades), positioning learned evaluation + MCTS as a viable replacement.
- The abstract states superhuman level was reached "within 24 hours" in all three games and that it "convincingly defeated a world-champion program in each case."
- Generalises the AlphaGo Zero approach (previously Go-only) into a domain-agnostic template.

## Direct quotes
> "Starting from random play, and given no domain knowledge except the game rules, AlphaZero achieved within 24 hours a superhuman level of play in the games of chess and shogi (Japanese chess) as well as Go, and convincingly defeated a world-champion program in each case." — Abstract

> "we generalise this approach into a single AlphaZero algorithm that can achieve, tabula rasa, superhuman performance in many challenging domains." — Abstract

## Method essentials
*(Not derivable from the abstract alone; see [[sources/silver2017-alphago-zero]] for the companion paper with full Methods, and fill in on a full read of the PDF.)*
- Network architecture (inputs, body, heads): —
- Search: —
- Training: —
- Evaluation / model promotion: —

## Numbers worth remembering
- "within 24 hours" to superhuman in chess, shogi, and Go (abstract).
- Other numbers (sims/move, TPU counts, ELO, batch size) require full-text read; omitted.

## Our open questions
- Exact PUCT formulation and c_puct value used; how sensitive is performance to it?
- How does the training target for the policy head combine visit counts and temperature?
- Replay buffer size and staleness policy; does AlphaZero use a separate "best player" gate like AlphaGo Zero or continuous updating? (Widely reported: continuous; verify on full read.)
- Role and magnitude of Dirichlet noise at the root; is it tuned per game?
- Value head target: final game outcome z only, or bootstrapped?
- How many self-play games / MCTS simulations per move during training vs evaluation?
- L2 regularization coefficient and overall loss weighting.

## How this maps to MLFactory
- Boop is a 6×6 2-player perfect-info game with D4 symmetry. **Transfers directly**: the overall loop (self-play → store (s, π, z) → train joint policy+value net → use net to guide MCTS → repeat); PUCT selection; Dirichlet noise at the root; temperature schedule (exploration early, greedy late); outcome-only value target; symmetry augmentation (D4 = 8× data multiplier).
- **Needs adjusting**: Boop has piece placement + graduation, so action encoding isn't simple "from-to". Legal-move masking must be explicit. Terminal/draw conditions differ and need a clean game interface. No "no-progress" rule analog — consider a move cap.
- **Tiny state space implications**: Network can be drastically smaller than Go's 40-block, 256-channel tower — a few residual blocks with modest channels likely sufficient. MCTS sims/move can be much lower (tens to low hundreds, not 800) because branching factor and depth are smaller; bottleneck shifts from sim count to NN eval latency → batching self-play across parallel games matters more.
- **Risk**: tiny state space → easier overfitting to self-play distribution and policy collapse. Keep exploration noise, temperature, and replay buffer diversity honest.
