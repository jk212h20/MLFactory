# Phase 0 — Scaffold & Environment

**Status**: ✅ Complete
**Date**: 2026-04-16

Goal of Phase 0: create a clean, reproducible, documented repo with a verified ML environment and a seeded research wiki. No game code or agents — those start in Phase 1.

## Proof checklist

| # | Artifact | Result |
|---|---------|--------|
| 1 | Tooling installed (`uv`, Python 3.13, Homebrew) | ✅ |
| 2 | Repo initialised at `~/ActiveProjects/MLFactory` with own git | ✅ (`main` branch) |
| 3 | `pyproject.toml`, `.python-version`, `.gitignore` in place | ✅ |
| 4 | `uv sync --extra dev` succeeds end-to-end | ✅ |
| 5 | `torch 2.11.0` with MPS `available=True, built=True` | ✅ |
| 6 | `uv run mlfactory doctor` prints env | ✅ |
| 7 | `uv run pytest` — 5/5 passing | ✅ |
| 8 | Wiki seeded with ≥ 5 sources, ≥ 1 question, ≥ 2 techniques, ≥ 2 advice, ≥ 1 trail | ✅ (5, 1, 2, 2, 1) |
| 9 | Demo script `scripts/demo-phase0.sh` runs clean end-to-end | ✅ |
| 10 | `~/ActiveProjects/projects.json` and `PATTERNS.md` updated | ✅ |
| 11 | Clean initial git commit, plain-English message | ✅ (see `git log` at bottom) |

## Environment captured

```
macOS 15.3.1 (24D70)
Apple M4 Max — 16 cores (12P + 4E) — 128 GB unified memory
Python 3.13.13     (Homebrew)
uv     0.11.7
git    2.50.1
torch  2.11.0     (MPS available & built)
numpy  2.4.4
pytest 9.0.3
rich   15.0.0
typer  0.24.1
pydantic 2.13.1
```

## Tests

```
tests/test_environment.py::test_python_version      PASSED
tests/test_environment.py::test_package_imports     PASSED
tests/test_environment.py::test_torch_available     PASSED
tests/test_environment.py::test_mps_available       PASSED
tests/test_environment.py::test_mps_tensor_roundtrip PASSED

5 passed in 0.42s
```

## Directory structure at end of Phase 0

```
MLFactory/
├── README.md                  # North Star + quickstart + roadmap
├── CONTRIBUTING.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── src/mlfactory/
│   ├── __init__.py            # version = "0.0.1"
│   ├── cli.py                 # `mlfactory doctor` command
│   ├── core/   (__init__ only — protocols land in Phase 1)
│   ├── games/  (__init__ only — adapters land in Phase 1–2)
│   ├── agents/ (__init__ only — agents land in Phase 1, 3)
│   ├── tools/  (__init__ only — arena/selfplay land in Phase 1+)
│   └── analysis/ (__init__ only — policy probes land in Phase 3)
├── tests/
│   ├── __init__.py
│   └── test_environment.py    # 5 passing smoke tests
├── scripts/
│   └── demo-phase0.sh         # this phase's proof
├── results/
│   └── phase0-scaffold.md     # this file
├── experiments/               # gitignored; .gitkeep only
└── wiki/
    ├── README.md              # the Karpathy-style workflow spec
    ├── INDEX.md               # hand-maintained TOC
    ├── sources/               (5 stable notes + template)
    │   ├── silver2017-alphazero.md
    │   ├── silver2017-alphago-zero.md
    │   ├── browne2012-mcts-survey.md
    │   ├── anthony2017-expert-iteration.md
    │   └── schrittwieser2020-muzero.md
    ├── questions/             (1 answered + template)
    │   └── Q-001-where-to-start.md
    ├── techniques/            (2 drafts + template)
    │   ├── mcts-uct.md
    │   └── self-play-pipeline.md
    ├── advice/                (2 drafts)
    │   ├── getting-started.md
    │   └── debugging-rl.md
    ├── games/boop/            (1 placeholder)
    │   └── rules-summary.md
    ├── trails/                (1 narrative + template)
    │   └── 2026-04-getting-mlfactory-off-the-ground.md
    └── insights/              (empty; INSIGHTS.md + template)
        └── INSIGHTS.md
```

## How to reproduce this phase

From a fresh clone:

```
cd ~/ActiveProjects/MLFactory
uv sync --extra dev
bash scripts/demo-phase0.sh
```

Expected: every section green, all 5 tests passing, torch MPS both built and available.

## What Phase 0 does NOT include (intentionally)

- No game rules, no agents, no arena.
- No bridge to Boop's TS codebase.
- No solver.
- No CI.

These live in later phases. Phase 0's job is to remove all the "do I have the tools" friction so Phase 1 is pure game/search code.

## What's next (Phase 1 preview)

Implement:
- `src/mlfactory/core/env.py` — Env protocol.
- `src/mlfactory/games/tictactoe.py`, `connect4.py` — in-process adapters.
- `src/mlfactory/agents/random_agent.py`, `mcts.py` — random + vanilla UCT.
- `src/mlfactory/tools/arena.py` — match runner + win-rate matrix + ELO.
- Tests: Random vs Random ≈ 50% on tic-tac-toe; MCTS(800) never loses tic-tac-toe; MCTS >> Random on Connect 4.
- Demo: `uv run mlfactory tournament --game tictactoe --agents random,mcts100,mcts800 --games 200` → printed results table.

On success, `techniques/mcts-uct.md` gets promoted from `seed` to `stable`, and `trails/2026-04-*.md` gets a Phase 1 continuation.
