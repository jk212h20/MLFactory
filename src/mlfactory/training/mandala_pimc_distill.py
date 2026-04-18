"""PIMC self-play distillation with smoothed value targets.

Phase 2 of the Mandala plan: use PIMCMandalaAgent (Phase 1 winner at 73.3%
vs HP-PUCT at matched budget) as the teacher in self-play, then distill its
visit-count policy into a network.

Why PIMC-as-teacher:
- PIMC sees through the imperfect-info noise that defeated HP-PUCT-trained
  nets. Per-determinization searches give the policy a clean signal across
  many possible opponent hands.
- The student net learns "what action does PIMC choose averaged over hidden
  states", which is exactly the kind of marginalized policy a fast neural
  prior should output.

Why smoothed values:
- Single-game outcome (±1/0) is a noisy target in a high-variance,
  imperfect-info card game. Smoothed values (sample K consistent
  completions × M random rollouts each, average) trade some bias for a
  large variance reduction. This was the only value formulation that
  trained well in Phase 0.

Phase 2 gate: distilled net beats heuristic agent >= 60% over 40 games.

Usage:
    uv run python -m mlfactory.training.mandala_pimc_distill \\
        --output deploy/checkpoints/mandala-pimc-distill.pt \\
        --games 500 --pimc-dets 20 --pimc-sims 10 --n-workers 10 \\
        --smooth-completions 8 --smooth-rollouts 1 \\
        --epochs 8

Always detach for any non-smoke run (R1).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mlfactory.agents.alphazero_mlp import AlphaZeroMLP, MLPConfig
from mlfactory.games.mandala.actions import N_TEMPLATES
from mlfactory.games.mandala.encode import FEATURE_DIM
from mlfactory.training.mandala_hp_distill_smooth import pretrain
from mlfactory.training.mandala_value_smooth import smooth_values_parallel


# --- Worker init + entry point --------------------------------------------


def _worker_init() -> None:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _play_one_pimc_game(args: tuple[int, int, int, str, int]) -> dict:
    """Self-play one game where both sides use PIMCMandalaAgent.

    Records, per ply: feature view, PIMC visit-distribution policy target,
    mover, and a deepcopy of the full state core (for value smoothing).
    """
    seed, pimc_dets, pimc_sims, rollout_policy, temperature_moves = args

    import copy as _copy

    from mlfactory.games.mandala.encode import encode_view
    from mlfactory.games.mandala.env import MandalaEnv
    from mlfactory.games.mandala.pimc_agent import PIMCMandalaAgent
    from mlfactory.games.mandala.rules import get_player_view

    env = MandalaEnv(rng=random.Random(seed))

    def make_pimc(sd: int) -> PIMCMandalaAgent:
        return PIMCMandalaAgent(
            n_determinizations=pimc_dets,
            sims_per_det=pimc_sims,
            prior_temperature=1.0,
            rollout_policy=rollout_policy,
            mode="sample",
            sample_temperature=1.0,
            temperature_moves=temperature_moves,
            seed=sd,
        )

    a = make_pimc(seed)
    b = make_pimc(seed + 10_000)
    agents = (a, b)

    state = env.initial_state()
    per_move_features: list[np.ndarray] = []
    per_move_policy: list[np.ndarray] = []
    per_move_mover: list[int] = []
    per_move_state_core: list[dict] = []

    n = 0
    while not state.is_terminal and n < 300:
        legal = env.legal_actions(state)
        if not legal:
            break
        mover = state.to_play
        view = get_player_view(state.core, mover)
        features = encode_view(view, mover, state.history)

        action = agents[mover].act(env, state)
        search = agents[mover].last_search
        if search is not None:
            policy_target = search.policy_target.copy()
        else:
            policy_target = np.zeros(N_TEMPLATES, dtype=np.float32)
            policy_target[action] = 1.0

        per_move_features.append(features)
        per_move_policy.append(policy_target)
        per_move_mover.append(mover)
        per_move_state_core.append(_copy.deepcopy(state.core))

        state = env.step(state, action)
        n += 1

    if state.is_terminal and state.winner is not None:
        winner = state.winner
        values = [(1.0 if mover == winner else -1.0) for mover in per_move_mover]
        outcome_str = "p0_win" if winner == 0 else "p1_win"
    else:
        values = [0.0] * len(per_move_mover)
        outcome_str = "draw"

    n_total = len(per_move_mover)
    distances = [n_total - i for i in range(n_total)]

    return {
        "features": per_move_features,
        "policies": per_move_policy,
        "values": values,
        "distances": distances,
        "movers": per_move_mover,
        "state_cores": per_move_state_core,
        "n_moves": n,
        "outcome": outcome_str,
    }


def generate_pimc_data(
    n_games: int,
    pimc_dets: int,
    pimc_sims: int,
    n_workers: int,
    rollout_policy: str,
    temperature_moves: int,
    base_seed: int,
    progress_every: int = 0,
):
    jobs = [
        (base_seed + g, pimc_dets, pimc_sims, rollout_policy, temperature_moves)
        for g in range(n_games)
    ]
    ctx = mp.get_context("spawn")

    all_features: list[np.ndarray] = []
    all_policies: list[np.ndarray] = []
    all_values: list[float] = []
    all_distances: list[int] = []
    all_movers: list[int] = []
    all_state_cores: list[dict] = []
    n_p0_wins = n_p1_wins = n_draws = 0
    total_moves = 0

    t0 = time.monotonic()
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        for i, result in enumerate(pool.imap_unordered(_play_one_pimc_game, jobs), 1):
            all_features.extend(result["features"])
            all_policies.extend(result["policies"])
            all_values.extend(result["values"])
            all_distances.extend(result["distances"])
            all_movers.extend(result["movers"])
            all_state_cores.extend(result["state_cores"])
            total_moves += result["n_moves"]
            if result["outcome"] == "p0_win":
                n_p0_wins += 1
            elif result["outcome"] == "p1_win":
                n_p1_wins += 1
            else:
                n_draws += 1
            if progress_every and i % progress_every == 0:
                elapsed = time.monotonic() - t0
                avg_mv = total_moves / i
                rate = i / elapsed
                eta = (n_games - i) / rate if rate > 0 else 0
                print(
                    f"  [{i}/{n_games}] p0={n_p0_wins} p1={n_p1_wins} "
                    f"draws={n_draws} avg_moves={avg_mv:.0f} "
                    f"samples={len(all_features)} elapsed={elapsed:.0f}s "
                    f"eta={eta:.0f}s",
                    flush=True,
                )

    wall = time.monotonic() - t0
    stats = {
        "n_games": n_games,
        "p0_wins": n_p0_wins,
        "p1_wins": n_p1_wins,
        "draws": n_draws,
        "n_samples": len(all_features),
        "avg_moves": total_moves / max(n_games, 1),
        "wall_seconds": wall,
        "pimc_dets": pimc_dets,
        "pimc_sims": pimc_sims,
        "rollout_policy": rollout_policy,
    }
    return (
        np.stack(all_features, axis=0).astype(np.float32),
        np.stack(all_policies, axis=0).astype(np.float32),
        np.array(all_values, dtype=np.float32),
        np.array(all_distances, dtype=np.int32),
        np.array(all_movers, dtype=np.int8),
        all_state_cores,
        stats,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PIMC self-play distill with smoothed values")
    p.add_argument("--output", required=True)
    p.add_argument("--initial-checkpoint", default=None)
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--pimc-dets", type=int, default=20, help="PIMC determinizations per move")
    p.add_argument("--pimc-sims", type=int, default=10, help="inner PUCT sims per determinization")
    p.add_argument("--n-workers", type=int, default=10)
    p.add_argument("--rollout-policy", choices=["random", "heuristic"], default="random")
    p.add_argument(
        "--temperature-moves",
        type=int,
        default=4,
        help="ply count to use sample mode before switching to greedy",
    )
    p.add_argument("--smooth-completions", type=int, default=8)
    p.add_argument("--smooth-rollouts", type=int, default=1)
    p.add_argument("--smooth-rollout-policy", choices=["random", "heuristic"], default="random")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--value-hidden", type=int, default=128)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--value-decay", type=float, default=0.0)
    args = p.parse_args(argv)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("mps unavailable, falling back to cpu")
        device = "cpu"

    print(
        f"generating {args.games} PIMC self-play games "
        f"({args.pimc_dets}x{args.pimc_sims}={args.pimc_dets * args.pimc_sims} sims/move, "
        f"workers={args.n_workers})..."
    )
    features, policies, raw_values, distances, movers, state_cores, stats = generate_pimc_data(
        n_games=args.games,
        pimc_dets=args.pimc_dets,
        pimc_sims=args.pimc_sims,
        n_workers=args.n_workers,
        rollout_policy=args.rollout_policy,
        temperature_moves=args.temperature_moves,
        base_seed=args.seed,
        progress_every=args.progress_every,
    )
    print(
        f"  generation: {stats['wall_seconds']:.0f}s, "
        f"p0={stats['p0_wins']} p1={stats['p1_wins']} draws={stats['draws']} "
        f"samples={stats['n_samples']} avg_moves={stats['avg_moves']:.1f}"
    )
    if features.shape[0] == 0:
        print("no samples; aborting")
        return 1

    print(
        f"smoothing value targets ({args.smooth_completions} completions x "
        f"{args.smooth_rollouts} rollouts x {args.smooth_rollout_policy} per sample)..."
    )
    states_and_movers = list(zip(state_cores, [int(m) for m in movers]))
    smoothed_values = smooth_values_parallel(
        states_and_movers,
        n_workers=args.n_workers,
        n_completions=args.smooth_completions,
        rollouts_per_completion=args.smooth_rollouts,
        rollout_policy=args.smooth_rollout_policy,
        base_seed=args.seed * 31 + 7,
        progress_every=max(1, len(states_and_movers) // 20),
    )

    diffs = np.abs(raw_values - smoothed_values)
    print(
        f"  smoothing done: raw vs smoothed mean|diff|={diffs.mean():.3f}, "
        f"smoothed range [{smoothed_values.min():.2f}, {smoothed_values.max():.2f}], "
        f"smoothed std={smoothed_values.std():.3f}"
    )

    print("training (smoothed targets)...")
    net, train_stats = pretrain(
        features,
        policies,
        smoothed_values,
        distances,
        initial_checkpoint=args.initial_checkpoint,
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        value_hidden=args.value_hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        seed=args.seed,
        value_decay=args.value_decay,
    )

    net.save(
        out,
        extra={
            "source": "pimc_self_play_distill_smoothed",
            "data_stats": stats,
            "train_stats": train_stats,
            "pimc_dets": args.pimc_dets,
            "pimc_sims": args.pimc_sims,
            "smooth_completions": args.smooth_completions,
            "smooth_rollouts": args.smooth_rollouts,
        },
    )
    print(f"saved: {out} ({net.param_count():,} params)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
