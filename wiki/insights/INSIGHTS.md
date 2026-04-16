# Insights — Chronological Index

A dated feed of surprising findings from MLFactory experiments. Each entry links to a full writeup in `wiki/insights/` and to the experiment run that produced it.

> An insight is a finding that **changed our understanding**. Not everything that was merely confirmed belongs here.

## Rules
- One line per insight. Terse. Dramatic if warranted.
- Link to the detail page and to the experiment run.
- Most recent at the top.

## Log

- **2026-04-16** — [[insights/2026-04-16-mps-cpu-crossover-small-nets]]: **MPS is slower than CPU for batch-1 inference on tiny nets.** 313k-param AZ net with 100 sims: 108 ms/move on CPU vs 123 ms/move on MPS. Implication: self-play runs on CPU; training (batched) runs on MPS.
- **2026-04-16** — [[insights/2026-04-16-mcts-sign-bug]]: **MCTS had an inverted-sign bug in the random-rollout branch that was half-hidden on Connect 4 and fully exposed by Boop.** The Phase-1 "MCTS(50) ≈ Random" finding was not a noise floor — it was a bug. Post-fix MCTS(50) is ~650 ELO above random on Connect 4. Retroactively invalidates part of the 2026-04-16 logarithmic-ELO insight below.
- **2026-04-16** — [[insights/2026-04-16-mcts-logarithmic-in-sims]]: on Connect 4, vanilla UCT gains ~300 ELO per 4× simulation budget. (Retracted in part: the 50-sim plateau was a bug, not a real noise floor. The logarithmic scaling holds at higher budgets post-fix.)

---

*To add an insight: write `insights/YYYY-MM-DD-slug.md` (copy `_template.md`), then prepend a one-liner here linking to it.*
