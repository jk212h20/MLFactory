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

## Techniques (2: 1 stable, 1 seed)
- [[techniques/mcts-uct]] — **stable**. Vanilla UCT MCTS. Implemented in `src/mlfactory/agents/mcts.py`; validated on Connect 4 tournament.
- [[techniques/self-play-pipeline]] — **seed**. The AlphaZero-style training loop. Pattern + pitfalls + defaults. To be implemented in Phase 3.

## Advice (2)
- [[advice/getting-started]] — The ladder for approaching a new game: rules → random → MCTS → symmetry → encoding → net → scale.
- [[advice/debugging-rl]] — Checklist and instrumentation for when training goes sideways.

## Games
- Boop
  - [[games/boop/rules-summary]] — **stable**. Complete rules with empirical numbers from the 10k-game parity verification.
  - [[games/boop/status]] — **current**. Training status: champion is `boop-run52-iter15.pt` (promoted 2026-04-17, beat baseline 62.5% / p=0.040). No active training.

## Trails
- [[trails/2026-04-getting-mlfactory-off-the-ground]] — the Phase 0 narrative.

## Insights (2)
- [[insights/INSIGHTS]] — chronological feed.
- [[insights/2026-04-16-mcts-sign-bug]] — **2026-04-16**. MCTS had an inverted-sign bug in the random-rollout branch; fix added 500–650 ELO per MCTS agent on Connect 4 and fully inverted the Boop result (MCTS was *losing* to Random pre-fix).
- [[insights/2026-04-16-mcts-logarithmic-in-sims]] — **2026-04-16**. ~300 ELO per 4× sim budget on Connect 4. Partially retracted: the 50-sim "noise floor" was the sign bug, not a real floor.

## Navigating

Every page that makes a non-trivial claim should link to the source that supports it (in its `links.sources` frontmatter and inline). Every question should link to the sources it consulted. Every technique should link to the sources that introduced or validate it. This is what lets us regenerate the causal graph.

## Maintenance

This file is currently hand-maintained. A future `scripts/build-wiki-index.py` will regenerate it by walking frontmatter across all pages.
