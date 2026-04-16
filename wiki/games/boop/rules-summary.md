---
type: game-note
status: seed
created: 2026-04-16
updated: 2026-04-16
tags: [boop, rules]
links:
  sources: []
---

# Boop — rules summary (placeholder)

This page will be filled in during Phase 2 when we port Boop's rules from `Boop/server/src/game/GameState.ts` into Python.

## What we know at Phase 0
- 6×6 board.
- Two players, each has 8 kittens and 8 cats (16 total pieces per player).
- On your turn you place a piece from your supply. Placed pieces push adjacent pieces (kittens pushed by any; cats push kittens but cats are not pushed by kittens — to be verified against the rules file).
- Three-in-a-row of your **cats** = win. Three-in-a-row of **kittens** promotes (graduates) them to cats.
- Pushed off the board → returned to your supply.

## Open questions to answer during Phase 2
- Exact push rules: does a piece push into an occupied square (cascade) or stop?
- What happens when you have no kittens left — can you place cats? Can you place anywhere?
- Are there any stalemate/draw conditions?
- What exactly constitutes "three in a row" — orthogonal only, or diagonal too?
- When multiple three-in-a-rows happen in a single move, what gets graduated?
- Is the initial player chosen randomly or fixed?

These will be answered from the TypeScript source and locked in by the parity test.
