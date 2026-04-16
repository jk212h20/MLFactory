# MLFactory

> **North Star**: MLFactory is a self-improving automated games strategy research factory. We build ML tools and agents that discover strategies for games, measure them honestly, and distill what we learn into a living wiki. **Results and interesting insights are the product.**

## Status

**Phase 0: Scaffold & environment.** Repo structure, tooling, environment verification, and a seeded research wiki. No game code or agents yet — those land in Phase 1+.

## What's here

- `src/mlfactory/` — Python package (core protocols, games, agents, tools, analysis, CLI)
- `tests/` — pytest suite
- `wiki/` — the research knowledge base (see [`wiki/README.md`](wiki/README.md))
- `results/` — committed reports from each phase
- `experiments/` — per-run artifacts (gitignored; reproducible from config)
- `scripts/` — operational helpers, including `demo-phase*.sh` proofs

## Quickstart

```bash
# From this directory
uv sync --extra dev                 # install python 3.13 + torch + dev deps
uv run mlfactory doctor             # verify torch + MPS are working
uv run pytest                       # all tests should pass
bash scripts/demo-phase0.sh         # the end-to-end phase-0 proof
```

## Design principles

1. **Results and insights are the product.** Every training run produces a report. Surprises get captured in `wiki/insights/`.
2. **Reproducibility is non-negotiable.** Every run under `experiments/<game>/<run-id>/` stores config, git SHA, seed, hardware, logs, checkpoints, and a `report.md`.
3. **The wiki is living, not static.** Karpathy-style: we ingest sources, answer questions, promote techniques after second use, and narrate the journey in trails. See [`wiki/README.md`](wiki/README.md).
4. **Agent code must not import opponent code.** When we measure our agents against existing heuristic bots (e.g., Boop's `DeepThinker`), the external bot is treated as a black-box opponent via the arena. We never port or inspect its logic into our training loop.
5. **Rules reuse is fine; agent logic reuse is not.** Game rules are a spec. We port them cleanly and cross-verify for parity.
6. **Honest measurement.** Arena uses colour-balanced matches (each agent plays both sides), reports confidence intervals, and computes ELO with seeded opponents.

## Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Scaffold + env verified + wiki seeded | ✅ Done |
| 1 | Env protocol, Connect 4 (bitboard), vanilla UCT MCTS, colour-balanced arena with ELO | ✅ Done |
| 2 | Boop rules ported from `Boop/server/src/game/GameState.ts`; 10k-game Python↔TS parity test; symmetry group | ⏳ Next |
| 3 | AlphaZero-lite trained on Boop; beats `DeepThinker` (currently 75.6% tournament champion) ≥60% in 100-game match; beats the human (you) in 10-game match | ⏳ |
| 4 | Solve / strong evidence of first-player advantage in Boop | ⏳ stretch |

See [`results/`](results/) for phase reports.

## How to prove each phase worked

Every phase ships **four proof artifacts** so progress is legible:

1. **Green test suite**: `uv run pytest -v`
2. **Reproducible demo**: one command (`scripts/demo-phaseN.sh`) a stranger could run
3. **Results report**: `results/phaseN-*.md` with numbers, hardware captured, git SHA
4. **Wiki deltas**: new `sources/`, `questions/`, `techniques/`, or `insights/` entries

## Why this exists alongside the other ActiveProjects

Boop, Leverage, dotsAnd, ChessCards, RotatingConnect4, etc. are game **products** — they run matches between humans and between simple heuristic bots. MLFactory is the **research workshop** where we try to build agents that are strong enough to surprise us. Adapters in `src/mlfactory/games/` let us plug in existing game servers so the two sides stay in sync over time.

## Hardware

Primary development target: Apple M4 Max, 128 GB unified memory, PyTorch on MPS. Everything should also run CPU-only, just slower.

## License

MIT.
