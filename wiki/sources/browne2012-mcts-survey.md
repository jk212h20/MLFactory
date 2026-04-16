---
type: source
status: stable
created: 2026-04-16
updated: 2026-04-16
tags: [mcts, uct, search, survey]
provenance:
  url: https://ieeexplore.ieee.org/document/6145622
  accessed: 2026-04-16
  read_level: skim
speculation: false
notes: |
  Survey PDF not fetched cleanly (paywalled IEEE HTML + dead mirror). Content reconstructed
  from well-known field material cross-referenced against Wikipedia's direct citations to
  Browne et al. 2012. No verbatim quotes captured. Do a full read before citing specific
  empirical numbers from the survey.
---

# Browne et al. 2012 — A Survey of MCTS Methods

## One-sentence claim
MCTS is a family of best-first search algorithms that iteratively build an asymmetric game tree via (selection → expansion → simulation → backpropagation), using bandit-style confidence bounds (UCT) to balance exploration and exploitation, and works without a domain-specific evaluation function.

## Bibliographic
- Authors: Cameron Browne, Edward Powley, Daniel Whitehouse, Simon Lucas, Peter Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, Simon Colton
- Venue/year: IEEE Transactions on Computational Intelligence and AI in Games, vol. 4, no. 1, March 2012, pp. 1–43
- DOI: 10.1109/TCIAIG.2012.2186810
- Link: https://ieeexplore.ieee.org/document/6145622

## The four MCTS phases
1. **Selection** — Starting from the root, recursively pick a child using a tree policy (typically UCT) until reaching a node that is not fully expanded or terminal. Tree policy biases toward high-value children while still exploring under-visited ones.
2. **Expansion** — Unless the selected leaf is terminal, add one (or more) child nodes corresponding to untried legal moves.
3. **Simulation (rollout / playout)** — From the newly expanded child, play to a terminal state using a default policy (commonly uniform-random = "light"; "heavy" = hand-crafted/learned heuristics).
4. **Backpropagation** — Propagate terminal reward back up the path, incrementing visit counts n_i and accumulating wins w_i with perspective flipped for the player to move.

## UCT formula, written out
At each node, select child i maximising:

    UCT(i) = (w_i / n_i) + C * sqrt( ln(N) / n_i )

- `w_i` = total reward at child i (child's-player perspective).
- `n_i` = visits to child i.
- `N` = Σ n_i over siblings = parent visits.
- `C` = exploration constant; theoretical value `√2` for rewards in [0, 1]; tuned empirically in practice.

First term = exploitation (empirical mean). Second term = exploration bonus that grows for less-visited children.

## Variants relevant to 2-player perfect-info games
- **RAVE / AMAF (All-Moves-As-First)**: Share stats across nodes — every move in a rollout updates the AMAF counter at the current node. Speeds up early learning when move order matters less (Go, Hex). Combined with UCT via a β(n, ñ) weighting that decays AMAF as real visits grow (Gelly–Silver).
- **Progressive bias / progressive widening**: Add a heuristic `H(i)/n_i` bias in early selection; progressive widening restricts branching factor as a function of n. Essential for very high B.
- **Parallelisation** — three flavours:
  - *Leaf*: many rollouts from one leaf in parallel.
  - *Root*: independent trees per thread, aggregate root stats at the end.
  - *Tree*: shared tree with mutex/lock-free; **virtual loss** keeps threads off the same branch.
- **Transposition tables in MCTS**: Back tree by a DAG keyed on state hash so distinct sequences to the same position share stats. Speeds convergence when transpositions are common.
- **MC-RAVE / Prior knowledge (Gelly–Silver)**: Initialize new children with synthetic visit counts from a heuristic, biasing early selection.
- **Move selection at root** after budget: highest visit count (robust child) or highest value (max child); survey discusses robust-max, secure-child.
- **Heavy vs light playouts**: Hand-crafted 3×3 patterns (MoGo's) can help — counterintuitively, slightly suboptimal but correlated playouts sometimes beat uniform-random.

## Common pitfalls the survey calls out
- **Trap states / shallow traps**: UCT can overlook narrow tactical lines because selective expansion prunes the branch before rollouts reveal the refutation. Relevant to any sharp-tactics game.
- **Tuning C**: Optimal exploration constant depends on reward scale and game; `√2` is only correct for [0, 1] rewards + i.i.d. assumptions the tree doesn't satisfy.
- **Playout bias ≠ strength**: Stronger playouts can *hurt* MCTS if they reduce rollout diversity (less variance but biased estimate).
- **Convergence is asymptotic**: UCT provably converges to minimax in the limit but says nothing about finite-budget behavior.
- **Correlated evaluations**: Standard UCT assumes independent rewards per pull; in trees they're deeply correlated, so the UCB1 regret bound is heuristic, not tight.

## Direct quotes
*(Not extracted — paywalled IEEE HTML and dead mirror. Filled in on a full read.)*

## Numbers worth remembering
- Survey length: ~43 pages, IEEE TCIAIG vol. 4 no. 1, 2012.
- UCT exploration constant `C = √2 ≈ 1.414` for rewards in [0, 1] (theoretical).
- UCT introduced by Kocsis & Szepesvári 2006; MCTS name coined by Coulom 2006.

## Our open questions
- What's the right C for Boop given our reward scale? If we use [-1, +1] from side-to-move, nominal C is `√2` after rescaling; empirical tuning likely needed.
- How big must the search budget be before UCT meaningfully beats flat MC on Boop?
- Does RAVE make sense on Boop? RAVE assumes "move at X is valuable regardless of when played" — true for Go stones, plausibly for Boop placements, but piece-pushing dynamics may break the AMAF assumption.
- Is progressive widening needed at Boop's branching (~20–40)? Probably not.

## How this maps to MLFactory
- **First variants worth trying on Boop** (B≈20–40, small board):
  1. Plain UCT with tuned C as baseline — must beat flat MC first.
  2. Virtual loss + tree parallelisation if we run self-play on multi-core CPU; root parallelisation is simplest and almost free.
  3. AlphaZero-style ([[sources/silver2017-alphago-zero]]): replace simulation with a neural-net value head; replace UCT exploration term with PUCT (uses a learned prior P(a|s)). Survey doesn't cover it but sets up the vocabulary.
  4. Dirichlet noise at root during self-play for exploration diversity.
  5. RAVE: probably low priority for Boop because piece-pushing makes values order-dependent.
- **Transposition tables**: Boop positions *can* recur (symmetries, commutative early placements), but in self-play with Dirichlet noise and modest budget, most search positions are unique per-game. Low priority until baseline works.
- **Progressive widening**: Skip. B≈20–40 is in UCT's comfort zone.
- **Pitfall to watch**: Boop has tactical traps (3-in-a-row setups that can be pushed apart). Plain UCT at low budgets will probably miss them — argument for neural value guidance sooner rather than later.
