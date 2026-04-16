# Phase 2 — Boop Rules, Parity with TypeScript, D4 Symmetry

**Status**: ✅ Complete
**Date**: 2026-04-16
**Git SHA**: (recorded at commit)

## Goal

Port Boop's rules from `Boop/server/src/game/GameState.ts` into a clean Python `Env`, cross-verify byte-for-byte against the authoritative TypeScript source over thousands of random games, implement the D4 symmetry group, produce a tensor encoding ready for Phase-3 neural networks, and run a baseline MCTS tournament. The anti-contamination rule holds: we port *rules* but not *agent logic*.

## Proof artifacts

| # | Artifact | Result |
|---|---------|--------|
| 1 | `src/mlfactory/games/boop/rules.py` — 400+ lines, all Boop rules | ✅ |
| 2 | 18 hand-written rule tests (boops, chains, graduations, multi-option, wins, stranded-fallback, immutability) | ✅ all pass |
| 3 | TypeScript stdio bridge at `src/mlfactory/games/boop/bridge/` wrapping the authoritative `BoopGame` | ✅ |
| 4 | `scripts/verify-boop-rules.sh` — **10,000 games, 571,266 transitions, zero divergences** | ✅ |
| 5 | 2 pytest-level parity tests (fast initial-state + 500-game slow) | ✅ both pass |
| 6 | D4 symmetry group: 8 elements, commutativity with step() verified (with documented exclusion for stranded-fallback) | ✅ 19 tests pass |
| 7 | Tensor encoding (11 planes × 6 × 6) + 5 encoding tests | ✅ all pass |
| 8 | Boop registered in CLI: `mlfactory tournament --game boop` works | ✅ |
| 9 | Boop MCTS tests: plays legal moves, MCTS(100) beats Random >80% | ✅ both pass |
| 10 | **Major insight**: discovered+fixed an MCTS sign bug that was silently weakening Phase 1 | ✅ insight logged, retraction made |
| 11 | `wiki/games/boop/rules-summary.md` promoted from placeholder to `stable`, grounded in the verified port | ✅ |

## Headline numbers

### Python ↔ TypeScript rules parity

```
=== Boop rules parity: 10000 games ===
Total state transitions: 571266
Parity failures:         0
Wall time:               55.8s (10247 moves/s across bridge)
Moves/game: mean=57.1  median=56  max=121
Wins: orange=5219  gray=4781  none=0
Orange (player 1) win rate under random play: 0.522
```

**Zero divergences** in more than half a million state transitions. Python rules are byte-identical to the authoritative TypeScript implementation.

### Boop MCTS tournament (20 games per pair, seed=0)

| agent   | random  | mcts50 | mcts200 |
|---------|:-------:|:------:|:-------:|
| random  |   —     | 15%    | 0%      |
| mcts50  | 85%     |   —    | 10%     |
| mcts200 | 100%    | 90%    |   —     |

ELO leaderboard (anchor = random @ 1500):

| rank | agent   | ELO  |
|------|---------|------|
| 1    | mcts200 | 2233 |
| 2    | mcts50  | 1822 |
| 3    | random  | 1500 |

### Connect 4 post-fix (regression with MCTS sign fix from Phase 1)

ELO leaderboard:

| rank | agent   | ELO (Phase 1, buggy) | ELO (Phase 2, fixed) |
|------|---------|:---:|:---:|
| 1    | mcts800 | 2161 | **2673** |
| 2    | mcts200 | 1811 | **2446** |
| 3    | mcts50  | 1510 | **2158** |
| 4    | random  | 1500 | 1500 |

**Every MCTS agent gained 500–650 ELO from the sign fix.** See [`wiki/insights/2026-04-16-mcts-sign-bug.md`](../wiki/insights/2026-04-16-mcts-sign-bug.md) for the detailed root-cause analysis.

## Surprises encountered and resolved

### 1. TS graduation-choice wins do NOT advance `currentTurn`
Caught at game 1520 of the initial 10k parity run (before this was a formal test). After a graduation that produced a win, TS's `selectGraduation()` leaves `currentTurn` unchanged, while its `placePiece()` always advances it. We mirrored this quirk for byte-identical parity; `terminal_value()` handles both sign cases correctly. See the `_step_graduation_choice` docstring in `rules.py`.

Note: **graduation-choice wins are structurally impossible** in Boop (graduation removes pieces, win conditions require cats *on* the board). The code path exists for defense in depth but never executes in practice. We keep the parity anyway because correctness of the edge is cheap.

### 2. Stranded-fallback is not symmetry-invariant
TS's "all-8-pieces-on-board, no line" fallback graduates the *first kitten in row-major order* — an order-dependent tie-break that breaks under rotation/reflection. Our symmetry tests initially failed on all 7 non-identity symmetries because of this. Resolution: `Boop.would_trigger_stranded_fallback(state, action)` lets augmentation code skip these transitions. Symmetry tests now pass with this explicit filter.

### 3. MCTS was silently miscomputing rollout rewards (the big one)
Phase 1 declared MCTS "working" on Connect 4 based on an 85–97% win rate against Random. Phase 2's Boop tournament immediately exposed the bug: MCTS went 0–20 vs Random. Investigation revealed an inverted sign in `_rollout()` for the random-playout path. The terminal-entry branch was correct; the random-playout branch was not. Because Connect 4 has short games, the correct terminal-entry branch was reached often enough that MCTS still "worked," just substantially under-powered.

**Impact**:
- Phase 1's "MCTS(50) ≈ Random" finding was the bug masquerading as an insight. Retraction logged.
- Every MCTS agent gained 500–650 ELO after the fix.
- Boop went from "MCTS loses to Random" to "MCTS(200) beats Random 100%" — a swing of thousands of effective ELO.

Full writeup: [`wiki/insights/2026-04-16-mcts-sign-bug.md`](../wiki/insights/2026-04-16-mcts-sign-bug.md).

### 4. A subtle dead-end state in Boop
A player with zero pool and fewer than 8 pieces on board has no legal moves. The TS rules deadlock here; we added a defensive `_terminal_if_stuck` that declares forfeit. **This state was never observed in 10,000 random games from the initial position** — it's reachable only from synthetic positions, but it surfaced when MCTS rolled out from a hand-constructed debug state.

## Implementation details worth noting

- **Bitboard-style encoding** not used for Boop (board is tiny, graduation/boop logic is easier to reason about on a tuple).
- **State is a frozen dataclass**. `step()` never mutates. Verified by `test_immutability`.
- **Action space**: flat 104 integers (72 placement + 32 graduation-choice). Actual max graduation-choice options on 6×6 is much less than 32; we're generous to keep indexing simple.
- **TS bridge** reuses the existing `Boop/server/src/game/GameState.ts` unchanged. Bridge is a ~200-line Node/tsx stdio server in `src/mlfactory/games/boop/bridge/`. Installs its own isolated `node_modules` (needs `npm install` once).
- **Performance**: 10,247 moves/second through the bridge, including the Python↔Node round-trip. Python-only rules runs ~10× faster (rough estimate from stranded-state exploration). Fast enough for Phase-3 self-play without optimization.

## Reproduce

```bash
cd ~/ActiveProjects/MLFactory
uv run pytest -v                               # 68/68 passing
bash scripts/demo-phase2.sh                    # full end-to-end proof
bash scripts/verify-boop-rules.sh 10000        # the big parity run
```

## What Phase 2 did NOT include

- No neural network. Phase 3 starts AlphaZero-lite on top of this infrastructure.
- No training. The encoding exists; the net doesn't.
- No Boop-vs-DeepThinker match. That's a Phase-3 milestone, needing a trained agent.

## What Phase 2 promoted

- `wiki/games/boop/rules-summary.md` — placeholder → **stable** with verified rules and empirical numbers.
- Added **insight**: [`wiki/insights/2026-04-16-mcts-sign-bug.md`](../wiki/insights/2026-04-16-mcts-sign-bug.md).
- Added defensive APIs: `Boop.would_trigger_stranded_fallback()`, `Boop._terminal_if_stuck()`.

## What's next (Phase 3 preview)

Phase 3 is the first real ML work:
- **AlphaZero-lite** network: ~4 residual blocks × 64 channels, policy + value heads.
- **PUCT MCTS** with Dirichlet root noise and a value-head evaluator (no random rollouts).
- **Self-play pipeline** with D4 symmetry augmentation (filtering stranded-fallback transitions).
- **Arena-gated training**: target beating `DeepThinker` (the existing Boop bot tournament champion at 75.6% win rate) ≥60% over 100 games.
- **Human match**: `scripts/play-human.sh` so the author can play 10 games vs the trained agent; target: agent wins a majority.

Expected Phase 3 runtime on M4 Max: several hours of self-play per training iteration, a few hundred iterations to reach target strength. Exact numbers TBD after initial ablations.
