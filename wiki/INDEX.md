# Wiki Index

A cross-referenced table of contents for the MLFactory wiki. See [`README.md`](README.md) for the workflow.

## Sources (5)
- [[sources/silver2017-alphazero]] — Silver et al. 2017, AlphaZero. Tags: `alphazero self-play mcts deep-rl games`. read_level: abstract.
- [[sources/silver2017-alphago-zero]] — Silver et al. 2017, AlphaGo Zero. The canonical detailed reference. Tags: `alphago-zero self-play mcts residual-network go`. read_level: abstract.
- [[sources/browne2012-mcts-survey]] — Browne et al. 2012, MCTS survey. Tags: `mcts uct search survey`. read_level: skim.
- [[sources/anthony2017-expert-iteration]] — Anthony, Tian, Barber 2017, ExIt. Tags: `expert-iteration exit mcts policy-distillation hex`. read_level: abstract.
- [[sources/schrittwieser2020-muzero]] — Schrittwieser et al. 2020, MuZero. Tags: `muzero model-based-rl planning mcts`. read_level: skim.

## Questions (1)
- [[questions/Q-001-where-to-start]] — What algorithm and starting architecture for Boop on M4 Max? **Answered** (high confidence): AlphaZero-lite, 4 blocks × 64 channels, PUCT + Dirichlet, 100–200 MCTS sims/move, D4 augmentation, gate-less training.

## Techniques (2, both `seed` — promote after first use in code)
- [[techniques/mcts-uct]] — Vanilla UCT MCTS. Pattern + pitfalls. To be implemented in Phase 1.
- [[techniques/self-play-pipeline]] — The AlphaZero-style training loop. Pattern + pitfalls + defaults. To be implemented in Phase 3.

## Advice (2)
- [[advice/getting-started]] — The ladder for approaching a new game: rules → random → MCTS → symmetry → encoding → net → scale.
- [[advice/debugging-rl]] — Checklist and instrumentation for when training goes sideways.

## Games
- Boop
  - [[games/boop/rules-summary]] — placeholder; filled in during Phase 2.

## Trails
- [[trails/2026-04-getting-mlfactory-off-the-ground]] — the Phase 0 narrative.

## Insights
- [[insights/INSIGHTS]] — chronological feed (currently empty; first entries arrive in Phase 2 or 3).

## Navigating

Every page that makes a non-trivial claim should link to the source that supports it (in its `links.sources` frontmatter and inline). Every question should link to the sources it consulted. Every technique should link to the sources that introduced or validate it. This is what lets us regenerate the causal graph.

## Maintenance

This file is currently hand-maintained. A future `scripts/build-wiki-index.py` will regenerate it by walking frontmatter across all pages.
