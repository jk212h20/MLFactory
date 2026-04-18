"""Smoothed value targets for Mandala training.

Background: The naive value target z(s) ∈ {-1, 0, +1} based on actual
game outcome from position s is a single noisy sample. Most of the
variance comes from things the net cannot see: opponent's hidden hand,
opponent's hidden starting cup cards, and future deck draws.

Training the value head against z(s) forces it to fit irreducible
noise. By analogy: if you can see 3 dice and 2 are hidden, the optimal
predictor of the sum is `visible + 2 × 3.5 = visible + 7`. The hidden
dice contribute irreducible variance — fitting the actual sum makes
training slower than fitting `visible + 7`.

What this module does: replace z(s) with a SMOOTHED value v̂(s) =
average outcome over K consistent completions × M random rollouts each.
The smoothed target has variance ≈ var(z) / (K·M), so the net sees a
much cleaner gradient signal per sample.

Implementation:
- Sample a consistent hidden state (opp hand + opp starting cup + deck)
  from the residual color pool, given what's visible at s.
- Run a random or heuristic rollout to terminal.
- Average outcomes from K such samples.

This is intentionally NOT cheating-value — we never expose the actual
hidden state to the net. We just average over the distribution of
"plausibly possible" hidden states from the public-info perspective.
"""

from __future__ import annotations

import multiprocessing as mp
import random
from dataclasses import dataclass

import numpy as np

from mlfactory.games.mandala.env import MandalaEnv, MandalaState
from mlfactory.games.mandala.heuristic_agent import HeuristicMandalaAgent
from mlfactory.games.mandala.rules import (
    CARDS_PER_COLOR,
    COLORS,
    INITIAL_CUP_SIZE,
)


def _resample_hidden_state(
    public_state: dict,
    mover: int,
    rng: random.Random,
) -> dict:
    """Given the engine's full state, resample the hidden parts (opp's
    hand + opp's starting cup cards + deck), keeping all visible parts
    fixed. Returns a new state dict with the same shape.

    'Visible parts': my hand, my cup, both rivers, both mandalas
    (mountains + fields), discard pile, current scores/phase. The
    residual color pool (cards-per-color minus visible counts) is
    what we redistribute among the hidden zones.
    """
    new_state = {k: v for k, v in public_state.items()}
    # Deep-copy mutable structures we'll change.
    new_state["players"] = [dict(p) for p in public_state["players"]]
    new_state["mandalas"] = [dict(m) for m in public_state["mandalas"]]

    opp_idx = 1 - mover
    opp = new_state["players"][opp_idx]
    me = new_state["players"][mover]

    # Determine how many cards in each hidden zone.
    opp_hand_size = len(opp["hand"])
    opp_starting_cup_size = opp.get("startingCupCount", INITIAL_CUP_SIZE)
    opp_starting_cup_size = min(opp_starting_cup_size, len(opp["cup"]))
    deck_size = len(public_state["deck"])

    # Compute residual color pool: per-color count of cards NOT visible
    # to `mover`. These are the cards that MIGHT be in opp's hand, opp's
    # starting cup, or the deck (in some unknown distribution).
    visible_counts = [0] * len(COLORS)
    color_idx = {c: i for i, c in enumerate(COLORS)}

    # My hand, my cup (fully visible to me).
    for c in me["hand"]:
        if c["color"] in color_idx:
            visible_counts[color_idx[c["color"]]] += 1
    for c in me["cup"]:
        if c["color"] in color_idx:
            visible_counts[color_idx[c["color"]]] += 1
    # Opp's cup beyond starting (claimed cards are visible).
    for c in opp["cup"][opp_starting_cup_size:]:
        if c["color"] in color_idx:
            visible_counts[color_idx[c["color"]]] += 1
    # Discard pile, mountains, fields — all public.
    for c in public_state["discardPile"]:
        if c["color"] in color_idx:
            visible_counts[color_idx[c["color"]]] += 1
    for m in public_state["mandalas"]:
        for c in m["mountain"]:
            if c["color"] in color_idx:
                visible_counts[color_idx[c["color"]]] += 1
        for f in m["fields"]:
            for c in f:
                if c["color"] in color_idx:
                    visible_counts[color_idx[c["color"]]] += 1

    residual_pool: list[dict] = []
    counter = 1_000_000  # synthetic ids to avoid colliding with real ids
    for ci, color in enumerate(COLORS):
        n = CARDS_PER_COLOR - visible_counts[ci]
        for _ in range(n):
            residual_pool.append({"id": f"hidden-{counter}", "color": color})
            counter += 1

    rng.shuffle(residual_pool)

    needed = opp_hand_size + opp_starting_cup_size + deck_size
    if needed > len(residual_pool):
        # Shouldn't happen given correct accounting, but be defensive.
        needed = len(residual_pool)

    # Distribute: first opp_hand_size to opp hand, next opp_starting_cup_size
    # cards REPLACE opp's first starting_cup positions (preserve any
    # later visible cup cards), rest to deck.
    cursor = 0
    new_opp_hand = residual_pool[cursor : cursor + opp_hand_size]
    cursor += opp_hand_size
    new_opp_starting_cup = residual_pool[cursor : cursor + opp_starting_cup_size]
    cursor += opp_starting_cup_size
    new_deck = residual_pool[cursor : cursor + deck_size]

    opp["hand"] = new_opp_hand
    # Replace only the starting-cup portion; later cards (publicly
    # claimed) stay as-is.
    opp["cup"] = new_opp_starting_cup + opp["cup"][opp_starting_cup_size:]
    new_state["deck"] = new_deck

    return new_state


def _rollout_to_terminal(
    state: MandalaState,
    mover: int,
    rng: random.Random,
    rollout_policy: str = "random",
    max_moves: int = 200,
) -> float:
    """Random or heuristic rollout to terminal. Returns ±1/0 from
    `mover`'s perspective."""
    env = MandalaEnv(rng=rng)
    cur = state
    if rollout_policy == "heuristic":
        h0 = HeuristicMandalaAgent(seed=rng.randint(0, 2**31 - 1))
        h1 = HeuristicMandalaAgent(seed=rng.randint(0, 2**31 - 1))
    n = 0
    while not cur.is_terminal and n < max_moves:
        legal = env.legal_actions(cur)
        if not legal:
            break
        if rollout_policy == "heuristic":
            a = (h0 if cur.to_play == 0 else h1).act(env, cur)
        else:
            a = rng.choice(legal)
        cur = env.step(cur, a)
        n += 1
    if not cur.is_terminal or cur.winner is None:
        return 0.0
    return 1.0 if cur.winner == mover else -1.0


def smooth_value_target(
    public_state: dict,
    mover: int,
    n_completions: int = 8,
    rollouts_per_completion: int = 1,
    rollout_policy: str = "random",
    seed: int | None = None,
) -> float:
    """Compute v̂(s) = average outcome over n_completions×rollouts_per_completion
    samples of consistent hidden state + random rollout.

    Returns scalar in [-1, 1].
    """
    rng = random.Random(seed)
    outcomes: list[float] = []
    for _ in range(n_completions):
        sampled = _resample_hidden_state(public_state, mover, rng)
        # Wrap into MandalaState (history can be empty for rollout purposes).
        sampled_state = MandalaState(core=sampled, history=[])
        for _ in range(rollouts_per_completion):
            v = _rollout_to_terminal(sampled_state, mover, rng, rollout_policy)
            outcomes.append(v)
    if not outcomes:
        return 0.0
    return float(np.mean(outcomes))


# --- Parallel batch generator -----------------------------------------------


def _worker_init() -> None:
    import torch

    try:
        torch.set_num_threads(1)
    except:
        pass
    try:
        torch.set_num_interop_threads(1)
    except:
        pass


@dataclass
class _SmoothJob:
    state_core: dict  # the public state (already a player-view if desired)
    mover: int
    n_completions: int
    rollouts_per_completion: int
    rollout_policy: str
    seed: int


def _do_smooth(job: _SmoothJob) -> float:
    return smooth_value_target(
        public_state=job.state_core,
        mover=job.mover,
        n_completions=job.n_completions,
        rollouts_per_completion=job.rollouts_per_completion,
        rollout_policy=job.rollout_policy,
        seed=job.seed,
    )


def smooth_values_parallel(
    states_and_movers: list[tuple[dict, int]],
    *,
    n_workers: int = 10,
    n_completions: int = 8,
    rollouts_per_completion: int = 1,
    rollout_policy: str = "random",
    base_seed: int = 0,
    progress_every: int = 0,
) -> np.ndarray:
    """Compute smoothed value targets for many positions in parallel.

    Returns np.ndarray (len(states_and_movers),) of float values."""
    import time as _time

    jobs = [
        _SmoothJob(
            state_core=core,
            mover=mover,
            n_completions=n_completions,
            rollouts_per_completion=rollouts_per_completion,
            rollout_policy=rollout_policy,
            seed=base_seed + i,
        )
        for i, (core, mover) in enumerate(states_and_movers)
    ]
    out = np.zeros(len(jobs), dtype=np.float32)
    ctx = mp.get_context("spawn")
    t0 = _time.monotonic()
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        for i, v in enumerate(pool.imap(_do_smooth, jobs)):
            out[i] = v
            if progress_every and (i + 1) % progress_every == 0:
                dt = _time.monotonic() - t0
                print(
                    f"  smoothed {i + 1}/{len(jobs)} positions in {dt:.0f}s",
                    flush=True,
                )
    return out
