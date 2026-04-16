# Contributing to MLFactory

## Coding conventions
- Python 3.13, `uv` for env management.
- `ruff` for lint/format: `uv run ruff check .` and `uv run ruff format .`.
- Type hints everywhere public. `from __future__ import annotations` at file top.
- Docstrings: terse and purposeful; no boilerplate.
- Prefer pure functions. Mutating game state is a bug.

## Testing
- `uv run pytest` must be green before any commit to `main`.
- New game adapters need a parity test against a reference implementation where one exists.
- New agents need a smoke test that confirms they produce legal moves.

## Wiki workflow
See [`wiki/README.md`](wiki/README.md). Summary:
1. New research question? Open a `wiki/questions/Q-NNN-slug.md`.
2. Read a new source? Write `wiki/sources/author-year-slug.md` with honest provenance and `read_level`.
3. Used a pattern twice? Promote to `wiki/techniques/slug.md`.
4. Discovered something surprising from an experiment? Write `wiki/insights/YYYY-MM-DD-slug.md` and prepend it to `wiki/insights/INSIGHTS.md`.
5. Every month or so, narrate in `wiki/trails/YYYY-MM-slug.md`.

## Experiment discipline
Every training run lives under `experiments/<game>/<run-id>/` and stores:
- `config.yaml` — full hyperparameters.
- `git_sha.txt` — commit hash.
- `seed.txt` — RNG seed.
- `hardware.txt` — output of `uname -a`, chip info, torch device.
- `log.txt` — full stdout.
- `checkpoints/` — model snapshots (gitignored).
- `report.md` — human-readable summary generated at run end.

A run is not "done" until `report.md` exists.

## Commit style
Plain English. Write what changed and, if non-obvious, why. Example: `scaffold repo: python 3.13, uv, torch MPS verified; seed wiki with 5 source notes`.

## Proof artifacts
Every phase ships:
1. Green tests (`uv run pytest`)
2. A `scripts/demo-phaseN.sh` that reproduces the proof
3. A `results/phaseN-*.md` report with numbers + hardware + git SHA
4. Wiki deltas (new `sources/`, `questions/`, `techniques/`, or `insights/`)

## Branching / PR flow
Solo repo. Work on `main`. If something becomes speculative, work on a branch and document in a trail.

## Anti-contamination rule
When measuring our agents against external bots (e.g., Boop's `DeepThinker`), those bots are opponents only. We do not read or port their logic into our training pipeline. Game *rules* are shared (as specs); agent *strategies* are not.
