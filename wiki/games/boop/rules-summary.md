---
type: game-note
status: stable
created: 2026-04-16
updated: 2026-04-16
tags: [boop, rules]
links:
  sources: []
  validated_by: [tests/test_boop_rules.py, tests/test_boop_parity.py, scripts/verify-boop-rules.sh]
  ts_source: Boop/server/src/game/GameState.ts
---

# Boop — rules summary

Authoritatively derived from `Boop/server/src/game/GameState.ts` and cross-verified by byte-identical Python↔TS parity over **10,000 random games, 571,266 state transitions, zero divergences** (see `scripts/verify-boop-rules.sh`).

## Board and pieces

- **Board**: 6×6. Cells indexed `(row, col)` with row 0 at the top.
- **Players**: orange (player 0, first to move) and gray (player 1).
- **Pieces**: kittens and cats.
- **Starting pool per player**: 8 kittens, 0 cats. Max cats per player: 8.

## Turn: place a piece

On your turn you choose a piece from your pool (kitten or cat) and place it on any empty cell. You cannot pass.

## The boop mechanic

After placement, check **all 8 neighboring cells** (orthogonal + diagonal). For each neighbor that holds a piece:

- **Rule**: a piece pushes the neighbor one square away, in the same direction as (center → neighbor).
- **Exception**: kittens cannot boop cats. Cats can boop anything (kittens and cats).
- **Blocked boop**: if the cell the neighbor would move into is occupied, the boop fails silently (the neighbor stays put).
- **Push off board**: if the neighbor would be pushed off the edge, it is returned to its owner's pool (as its existing piece type).

All boops for one placement are independent — no cascading chains.

## Graduation (3-in-a-row) mechanic

After boops, find all **3-in-a-row** formations of the **current player's** pieces (orthogonal OR diagonal).

- Qualifying line: 3 consecutive same-color pieces with **at least one kitten**.
- In a line of 4+ consecutive same-color pieces, every length-3 sliding window is an option.
- Duplicate sets (same cells, different order) are deduplicated.

Resolution:
- **Zero options**: continue to the next rule (stranded-fallback, below).
- **One option**: auto-execute. Remove all 3 pieces from the board. Each kitten → `kittens_retired++, cats_in_pool++` (kitten becomes a cat in pool). Each cat just returns to pool (`cats_in_pool++`).
- **Two or more options**: game enters `selecting_graduation` phase. **The same player chooses** one option (this is not an opponent turn). After choice, resolve as the one-option case.

## Stranded-fallback graduation

Edge case from the TS rules: if after placement+boops the current player has **all 8 of their pieces on the board** (`on_board >= 8`) AND **no 3-in-a-row options** AND `kittens_retired + cats_in_pool < 8`, then **one kitten auto-graduates**. The TS chooses the **first kitten in row-major order**.

This rule is **not symmetry-invariant** (row-major tie-break breaks under rotation/reflection). MLFactory detects these transitions (`Boop.would_trigger_stranded_fallback()`) so training-time data augmentation can skip them.

## Win conditions

A player wins if, at any point during their turn's resolution (placement, boop, or graduation):

1. **Three of their cats in a row** (orthogonal OR diagonal), OR
2. **All 8 of their cats simultaneously on the board**.

Win check happens AFTER boop AND AFTER graduation resolves. If either win condition holds for the just-moved player, the game ends.

## Turn advancement quirk

Sharp observation from the TS source: on a **graduation-choice win** (hypothetical — never actually occurs, see below), the TS does NOT advance `currentTurn`, while on a **placement win** it does. Our port mirrors this for byte-identical parity, and `terminal_value()` handles both sign conventions correctly.

Whether a graduation-choice can actually produce a win is a subtle question: graduation *removes* pieces from the board, and all win conditions require *cats on the board*. So graduation-choice wins are **structurally impossible** — the code path exists in both TS and our port, but never executes in practice.

## Dead-end states

Theoretically, a player could have zero pieces in pool and no 3-in-a-row, while also lacking 8 pieces on the board to trigger the fallback. In this state they have no legal move. The TS returns `valid:false` for all moves, deadlocking the game. We extend our port to treat this as a forfeit (opponent wins) via `Boop._terminal_if_stuck()`.

**Empirical note**: this state was **never observed in 10,000 random games from the initial position**. It's a pathological state reachable only from contrived synthetic positions.

## Observations from 10,000 random games

| Metric | Value |
|---|---|
| Total state transitions | 571,266 |
| Orange (first player) wins | 5219 |
| Gray wins | 4781 |
| Draws | 0 |
| Orange win rate under random play | **52.2%** |
| Mean game length | 57.1 moves |
| Median game length | 56 moves |
| Max game length | 121 moves |
| Bridge throughput (Python + TS) | ~10,000 moves/sec |

Orange's 52% edge under random play is small (z ≈ 4.4 given n=10,000, but effect size is modest). It's too little to call Boop a "solved first-player win," but large enough to be a real asymmetry worth investigating with stronger agents in Phase 3.

## Action space used by MLFactory

Flat action space of size 104:
- Actions 0..71: place a piece. `kind = action // 36` (0 kitten, 1 cat). `cell = action % 36`, then `row = cell // 6, col = cell % 6`.
- Actions 72..103: in `selecting_graduation` phase, choose the graduation option at index `action - 72`. The bound of 32 options is generous — actual max is far smaller.

Legal-action mask is always explicit; see `mlfactory.games.boop.encode.legal_mask()`.

## Symmetries

The 6×6 board has **D4 dihedral symmetry** (8 elements: 4 rotations + 4 reflections). We verified commutativity of the symmetry group with `step()` for every element on randomly-generated positions, with one documented exclusion: stranded-fallback transitions are not symmetry-invariant and must be skipped during augmentation. See `src/mlfactory/games/boop/symmetry.py`.

This gives training up to **8× data augmentation** per position.
