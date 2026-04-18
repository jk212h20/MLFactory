---
type: game-status
status: current
created: 2026-04-18
updated: 2026-04-18
tags: [boop, training, status]
links:
  validated_by: [experiments/boop/2026-04-17-125259-run52-aggressive/events.jsonl]
---

# Boop training status

Current snapshot of where Boop AlphaZero training stands. Update this file
when a new run finishes or a new champion is promoted.

## Champion (current best)

**`deploy/checkpoints/boop-run52-iter15.pt`** — copied from
`experiments/boop/2026-04-17-125259-run52-aggressive/checkpoints/iter-0015.pt`.

This is the strongest committed Boop AlphaZero checkpoint. It is the
policy/value net used for site deployment.

## How it got there

| Run | Source | Iters reached | Outcome |
|---|---|---|---|
| `2026-04-16-202702-smoke-pre3` | scratch | smoke | sanity-check that the pipeline trains end-to-end |
| `2026-04-16-222906-run50` | scratch | 50/50 (full) | first full training run; produced `boop-run50-iter50.pt` |
| `2026-04-17-115858-run51-warmstart` | warm-start from run50 | finished | warm-start sanity, did not promote |
| `2026-04-17-125259-run52-aggressive` | warm-start from run50-iter50 | **15/100 (early-stop)** | beat run50 baseline at 62.5% over 40 games (one-sided binomial p=0.040). Promoted to current champion. |

Run52 hit the early-stop gate (`stop_on_baseline_pvalue=0.05`,
`stop_requires_consecutive=1`) at iter 15. That's the validation
criterion that proved it actually improved over run50 — not just looked
better in noisy rolling stats.

## Run52 config (the one that worked)

The parameters that produced the current champion:

```
trainer:        alphazero
device:         mps
selfplay_games: 80
selfplay_sims:  200
train_batches:  300
batch_size:     128
lr:             1e-3
net:            4 blocks x 64 channels
augment:        true
warm-start:     experiments/boop/2026-04-16-222906-run50/checkpoints/iter-0050.pt
baseline:      same as warm-start
eval gate:      40 games every 5 iters, stop on one-sided p<=0.05
```

## Active work

**None.** No Boop training is currently running.

## Next move (if pushing strength further)

Resume from `boop-run52-iter15.pt` with `--resume-from` and
`--baseline-ckpt` both pointing to it. Run another 50-100 iters under
the same config (or aggressive-er learning rate). The early-stop gate
will tell us when iter N is significantly stronger than iter 15.

## Production deployment

Boop bot is live in production, served from `deploy/checkpoints/boop-run52-iter15.pt`
via the FastAPI service template at `src/mlfactory/service/app.py`,
deployed from `deploy/Dockerfile` to Railway. The Dockerfile pins
`AZ_CHECKPOINT=/app/checkpoints/boop-run52-iter15.pt`; `railway.toml`
does not override it.

(Caveat: a Railway dashboard env var could in principle override the
Dockerfile default. Not verified here. If in doubt, hit the service's
`/health` or `/info` endpoint and check the reported checkpoint name.)

**The deployed bot is the strongest checkpoint we have.** There is no
stronger Boop net sitting on disk waiting to ship — `run52-iter15` is
both the current champion and what's live.

## User feedback

- 2026-04-18: User reports they can no longer beat the deployed bot.
  This is the first signal of bot-vs-human strength against the
  current champion. Anecdotal (single human, unrecorded games), but
  suggestive that run52-iter15 is at or above the user's playing
  strength.
