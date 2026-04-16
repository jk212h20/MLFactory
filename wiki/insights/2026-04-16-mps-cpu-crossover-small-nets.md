---
title: MPS is slower than CPU for batch-1 inference on tiny nets
date: 2026-04-16
status: stable
tags: [performance, mps, hardware, alphazero, boop]
provenance: measured
experiment: phase 3b benchmark (one-off, not in experiments/)
---

# MPS is slower than CPU for batch-1 inference on tiny nets

## Finding

Running `AlphaZeroNet(4 blocks × 64 channels, 313k params)` on Boop with 100
PUCT simulations per move:

| device | ms/move |
|---|---|
| CPU | 108 |
| MPS | 123 |

CPU is **~14% faster** for this workload. Measured on M4 Max, torch 2.11,
batch size 1, 10 × 5 move samples after warmup.

## Why

PUCT's hot loop is single-state evaluations (expand one leaf per simulation).
Each net forward is a batch of size 1. At that batch size, MPS kernel dispatch
overhead dominates the actual matmul cost on a 6×6 board with a small net.

## Implication for Phase 3

- **Single-process self-play: run on CPU.** Simpler and faster for our size.
- **MPS wins when we batch.** Training (B=256) should live on MPS.
  Parallel self-play that batches leaf evaluations across workers could too.
- **Don't optimize prematurely.** If 108 ms/move scales a 50-game self-play
  round to ~6 min, that's fine for a first pass.

## Reproducing

One-off benchmark; not promoted to a test since it's hardware-specific
and timing-noisy. Run manually with:

```bash
uv run python -c "
import time
from mlfactory.agents.alphazero.agent import AlphaZeroAgent
from mlfactory.agents.alphazero.evaluator import NetEvaluator
from mlfactory.agents.alphazero.net import AlphaZeroNet, NetConfig
from mlfactory.agents.alphazero.puct import PUCTConfig
from mlfactory.games.boop import Boop
from mlfactory.games.boop.encode import encode_state, legal_mask
# ... (see conversation or git log for full snippet) ...
"
```

## Caveats

- A single run, small sample; don't read deeply into the 14% delta.
- On a larger net (say 200k → 1M params) or larger board, MPS likely
  starts winning even at batch 1. Re-measure before assuming otherwise.
- Does NOT apply to training — training batches (B=256) will be on MPS.

## Followups

- Once parallel self-play lands, measure batched evaluator throughput on
  MPS vs CPU across batch sizes 1, 4, 16, 64, 256. Pick the crossover point.
- If we ever grow to a wider net, re-audit.
