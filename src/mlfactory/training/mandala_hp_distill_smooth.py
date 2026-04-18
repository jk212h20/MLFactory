"""HP-PUCT distillation with SMOOTHED value targets.

Same as mandala_hp_distill but for each training sample, the value
target is the AVERAGE outcome over K consistent completions × M random
rollouts each, instead of the single noisy ±1/0 from the actual game.

See mandala_value_smooth.py for the rationale (the "hidden dice" point):
training the value head on a noisy single-sample target forces it to fit
irreducible variance; smoothed targets converge faster and to lower loss.

Usage:
    uv run python -m mlfactory.training.mandala_hp_distill_smooth \\
        --output deploy/checkpoints/mandala-hp-distill-smooth.pt \\
        --games 500 --hp-sims 50 --n-workers 10 \\
        --smooth-completions 8 --smooth-rollouts 1 \\
        --epochs 8

Detach via nohup for any nontrivial run.
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
import torch.nn.functional as F

from mlfactory.agents.alphazero_mlp import AlphaZeroMLP, MLPConfig
from mlfactory.games.mandala.actions import N_TEMPLATES
from mlfactory.games.mandala.encode import FEATURE_DIM
from mlfactory.training.mandala_value_smooth import smooth_values_parallel


# --- Worker init + entry point (mirrors mandala_hp_distill) ---------------


def _worker_init() -> None:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _play_one_hp_game_with_states(args: tuple[int, int, str]) -> dict:
    """Same as the worker in mandala_hp_distill, but also records the raw
    state core + mover at each ply, so we can later smooth the value
    targets per position."""
    seed, hp_sims, rollout_policy = args

    import copy as _copy
    from mlfactory.agents.alphazero.agent import AlphaZeroAgent
    from mlfactory.agents.alphazero.puct import PUCTConfig
    from mlfactory.games.mandala.encode import encode_view
    from mlfactory.games.mandala.env import MandalaEnv
    from mlfactory.games.mandala.heuristic_evaluator import HeuristicPriorEvaluator
    from mlfactory.games.mandala.rules import get_player_view

    env = MandalaEnv(rng=random.Random(seed))

    def make_hp(sd):
        ev = HeuristicPriorEvaluator(
            env,
            prior_temperature=1.0,
            rollout_policy=rollout_policy,
            rng_seed=sd,
        )
        return AlphaZeroAgent(
            ev,
            PUCTConfig(n_simulations=hp_sims),
            mode="sample",
            temperature=1.0,
            temperature_moves=4,
            add_root_noise=False,
            seed=sd,
        )

    a = make_hp(seed)
    b = make_hp(seed + 10_000)
    agents = (a, b)

    state = env.initial_state()
    per_move_features: list[np.ndarray] = []
    per_move_policy: list[np.ndarray] = []
    per_move_mover: list[int] = []
    per_move_state_core: list[dict] = []  # NEW

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
        # Record a deepcopy of the FULL state so the smoother can resample
        # consistent hidden state per position. We don't need the player
        # view here — the smoother itself will mask + resample.
        per_move_state_core.append(_copy.deepcopy(state.core))

        state = env.step(state, action)
        n += 1

    if state.is_terminal and state.winner is not None:
        winner = state.winner
        values = [(1.0 if mover == winner else -1.0) for mover in per_move_mover]
        outcome_str = "p0_win" if winner == 0 else "p1_win"
    else:
        winner = None
        values = [0.0] * len(per_move_mover)
        outcome_str = "draw"

    n_total = len(per_move_mover)
    distances = [n_total - i for i in range(n_total)]

    return {
        "features": per_move_features,
        "policies": per_move_policy,
        "values": values,  # raw, unsmoothed
        "distances": distances,
        "movers": per_move_mover,
        "state_cores": per_move_state_core,  # NEW
        "n_moves": n,
        "outcome": outcome_str,
    }


def generate_hp_data_with_states(
    n_games: int,
    hp_sims: int,
    n_workers: int,
    rollout_policy: str,
    base_seed: int,
    progress_every: int = 0,
):
    """Like generate_hp_data but also returns per-sample (state_core, mover)
    so we can compute smoothed value targets later."""
    jobs = [(base_seed + g, hp_sims, rollout_policy) for g in range(n_games)]
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
        for i, result in enumerate(pool.imap_unordered(_play_one_hp_game_with_states, jobs), 1):
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
                print(
                    f"  [{i}/{n_games}] p0={n_p0_wins} p1={n_p1_wins} "
                    f"draws={n_draws} avg_moves={avg_mv:.0f} "
                    f"samples={len(all_features)} elapsed={elapsed:.0f}s",
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
        "hp_sims": hp_sims,
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


# --- Pretrain (essentially same as mandala_hp_distill.pretrain) -----------


def pretrain(
    features: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    distances: np.ndarray | None = None,
    *,
    initial_checkpoint: str | None,
    hidden: int,
    n_blocks: int,
    value_hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
    value_decay: float = 0.0,
) -> tuple[AlphaZeroMLP, dict]:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    cfg = MLPConfig(
        feature_dim=FEATURE_DIM,
        n_actions=N_TEMPLATES,
        hidden=hidden,
        n_blocks=n_blocks,
        value_hidden=value_hidden,
    )
    net = AlphaZeroMLP(cfg).to(device)
    if initial_checkpoint:
        prior, _ = AlphaZeroMLP.load(initial_checkpoint, map_location="cpu")
        if prior.config != cfg:
            raise ValueError(f"checkpoint config {prior.config} doesn't match {cfg}")
        net.load_state_dict(prior.state_dict())
        net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = features.shape[0]

    if distances is None or value_decay <= 0:
        weights = np.ones(n, dtype=np.float32)
    else:
        weights = np.exp(-distances.astype(np.float32) / value_decay).astype(np.float32)

    print(
        f"pretraining: {n} samples, {epochs} epochs, batch {batch_size}, value_decay={value_decay}"
    )
    history = []
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        perm = np_rng.permutation(n)
        ep_p = ep_v = 0.0
        ep_total = 0
        net.train()
        for i0 in range(0, n, batch_size):
            idx = perm[i0 : i0 + batch_size]
            x = torch.from_numpy(features[idx]).to(device)
            pt = torch.from_numpy(policies[idx]).to(device)
            vt = torch.from_numpy(values[idx]).to(device)
            wt = torch.from_numpy(weights[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            logits, value_pred = net(x)
            log_probs = F.log_softmax(logits, dim=1)
            p_loss = -(pt * log_probs).sum(dim=1).mean()
            v_diff = value_pred.squeeze(-1) - vt
            v_sq = v_diff * v_diff
            wt_sum = wt.sum()
            v_loss = (v_sq * wt).sum() / wt_sum if wt_sum.item() > 0 else v_sq.mean()
            loss = p_loss + v_loss
            loss.backward()
            opt.step()
            ep_p += p_loss.item() * idx.size
            ep_v += v_loss.item() * idx.size
            ep_total += idx.size

        avg_p = ep_p / ep_total
        avg_v = ep_v / ep_total
        with torch.no_grad():
            x_all = torch.from_numpy(features[:1024]).to(device)
            pt_all = torch.from_numpy(policies[:1024]).to(device)
            logits, _ = net(x_all)
            argmax_match = (logits.argmax(1) == pt_all.argmax(1)).float().mean().item()
        print(
            f"  epoch {epoch}/{epochs}: policy_xent={avg_p:.4f} "
            f"value_mse={avg_v:.4f} argmax_match={argmax_match:.3f}"
        )
        history.append({"epoch": epoch, "policy": avg_p, "value": avg_v, "match": argmax_match})

    wall = time.monotonic() - start
    return net.cpu().eval(), {
        "wall_seconds": wall,
        "history": history,
        "final_policy_xent": history[-1]["policy"],
        "final_value_mse": history[-1]["value"],
        "final_argmax_match": history[-1]["match"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HP-PUCT distill with smoothed value targets")
    p.add_argument("--output", required=True)
    p.add_argument("--initial-checkpoint", default=None)
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--hp-sims", type=int, default=50)
    p.add_argument("--n-workers", type=int, default=10)
    p.add_argument("--rollout-policy", choices=["random", "heuristic"], default="random")
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
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--value-decay", type=float, default=0.0)
    args = p.parse_args(argv)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("mps unavailable, falling back to cpu")
        device = "cpu"

    print(
        f"generating {args.games} HP-PUCT games (sims={args.hp_sims}, workers={args.n_workers})..."
    )
    features, policies, raw_values, distances, movers, state_cores, stats = (
        generate_hp_data_with_states(
            n_games=args.games,
            hp_sims=args.hp_sims,
            n_workers=args.n_workers,
            rollout_policy=args.rollout_policy,
            base_seed=args.seed,
            progress_every=args.progress_every,
        )
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
        f"smoothing value targets ({args.smooth_completions} completions × "
        f"{args.smooth_rollouts} rollouts × {args.rollout_policy} per sample)..."
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

    # Stat: mean abs diff between raw and smoothed (sanity).
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
        smoothed_values,  # KEY DIFFERENCE: smoothed instead of raw
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
            "source": "hp_puct_distill_smoothed",
            "data_stats": stats,
            "train_stats": train_stats,
            "smooth_completions": args.smooth_completions,
            "smooth_rollouts": args.smooth_rollouts,
        },
    )
    print(f"saved: {out} ({net.param_count():,} params)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
