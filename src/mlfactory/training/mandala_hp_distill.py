"""Distill HP-PUCT (heuristic-prior PUCT) into the MLP net.

Pipeline:
  1. Run N self-play games where BOTH sides are HP-PUCT (heuristic-prior
     evaluator + PUCT search). For each game-position record:
       features (player-view)
       PUCT visit distribution (the policy target)
       final game outcome from this position's mover-perspective (value
         target, in {-1, 0, +1})
  2. Supervised-train the AlphaZeroMLP on those samples (cross-entropy
     on policy + MSE on value), optionally warm-starting from a prior
     checkpoint.

This is the "teacher-student distillation" path:
- Teacher = HP-PUCT (slow, ~200ms/move at 50 sims, but stronger than
  the heuristic alone — verified 70% vs heuristic at 200 sims).
- Student = the MLP, which after distillation should have priors at
  least as good as the heuristic (since it's imitating a strictly
  stronger source).

After distillation, the student net can be used as the prior in normal
AZ self-play, where it'll improve further.

Usage:
    uv run python -m mlfactory.training.mandala_hp_distill \
        --output deploy/checkpoints/mandala-hp-distill.pt \
        --games 500 --hp-sims 50 --n-workers 10 --epochs 8

Wraps cleanly with scripts/timed for budget safety:
    scripts/timed 600 30 -- uv run python -m mlfactory.training.mandala_hp_distill ...
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


# --- Worker init + entry point --------------------------------------------


def _worker_init() -> None:
    """Single-thread torch (matches our other parallel workers)."""
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _play_one_hp_game(args: tuple[int, int, str]) -> dict:
    """Worker entry. Plays one HP-PUCT vs HP-PUCT game and returns
    per-position samples + outcome.

    Imports happen inside the worker (spawn context); each spawned
    process pays its own torch import cost but parent stays light."""
    seed, hp_sims, rollout_policy = args

    from mlfactory.agents.alphazero.agent import AlphaZeroAgent
    from mlfactory.agents.alphazero.puct import PUCTConfig
    from mlfactory.games.mandala.encode import encode_view, make_history
    from mlfactory.games.mandala.env import MandalaEnv
    from mlfactory.games.mandala.heuristic_evaluator import HeuristicPriorEvaluator
    from mlfactory.games.mandala.rules import get_player_view

    env = MandalaEnv(rng=random.Random(seed))

    # Two distinct HP-PUCT agents (different seeds for tie-break diversity).
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
            mode="sample",  # sample from visit distribution for diversity
            temperature=1.0,
            temperature_moves=4,
            add_root_noise=False,  # heuristic prior already has spread
            seed=sd,
        )

    a = make_hp(seed)
    b = make_hp(seed + 10_000)
    agents = (a, b)

    state = env.initial_state()
    per_move_features: list[np.ndarray] = []
    per_move_policy: list[np.ndarray] = []
    per_move_mover: list[int] = []

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

        state = env.step(state, action)
        n += 1

    # Build value targets from outcome + per-position distance-from-terminal.
    # The distance is `n_moves - ply`, where ply is the position's index in the
    # game (0-based). So the very last move played has distance=1, the one
    # before distance=2, etc. The opening position has distance=n_moves.
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
        "values": values,
        "distances": distances,
        "n_moves": n,
        "outcome": outcome_str,
    }


# --- Generation orchestrator ----------------------------------------------


def generate_hp_data(
    n_games: int,
    hp_sims: int,
    n_workers: int,
    rollout_policy: str,
    base_seed: int,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Spawn n_workers processes; each plays HP-PUCT-vs-HP-PUCT games.

    Returns (features, policies, values, stats).
    Prints progress every `progress_every` games (0 = quiet)."""
    jobs = [(base_seed + g, hp_sims, rollout_policy) for g in range(n_games)]
    ctx = mp.get_context("spawn")

    all_features: list[np.ndarray] = []
    all_policies: list[np.ndarray] = []
    all_values: list[float] = []
    all_distances: list[int] = []
    n_p0_wins = n_p1_wins = n_draws = 0
    total_moves = 0

    t0 = time.monotonic()
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        for i, result in enumerate(pool.imap_unordered(_play_one_hp_game, jobs), 1):
            all_features.extend(result["features"])
            all_policies.extend(result["policies"])
            all_values.extend(result["values"])
            all_distances.extend(result["distances"])
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
        stats,
    )


# --- Pretrain (mirrors the existing bootstrap modules) --------------------


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
    endgame_only_epochs: int = 0,
    endgame_max_distance: int = 8,
) -> tuple[AlphaZeroMLP, dict]:
    """Cross-entropy on soft policy targets (PUCT visit dists) + MSE on
    value targets. Optionally warm-start from a prior checkpoint.

    Endgame backchaining (when value_decay > 0 OR endgame_only_epochs > 0):
    - distances[i] = how many plies until the game ended for sample i.
      The very last move played has distance=1, opening positions have
      distance ≈ avg_game_length.
    - per-sample value loss weight = exp(-distance / value_decay) if
      value_decay > 0, else 1.0.
      With value_decay=10, a position 5 plies from terminal has weight
      0.61; a position 50 plies out has weight 0.007. Net learns endgame
      values cleanly first because they dominate the loss.
    - endgame_only_epochs > 0: train ONLY on samples within
      endgame_max_distance of terminal for the first N epochs, then
      switch to weighted training on all samples. Stricter form of
      backchaining — net masters endgames before mid-game even matters.

    Policy loss is unweighted regardless of distance — the policy targets
    from PUCT visit distributions are equally valid at all distances."""
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

    # Per-sample weighting for endgame focus.
    if distances is None:
        weights = np.ones(n, dtype=np.float32)
    elif value_decay > 0:
        weights = np.exp(-distances.astype(np.float32) / value_decay).astype(np.float32)
    else:
        weights = np.ones(n, dtype=np.float32)

    print(
        f"pretraining: {n} samples, {epochs} epochs, batch {batch_size}, "
        f"value_decay={value_decay}, endgame_only_epochs={endgame_only_epochs}"
    )
    if value_decay > 0:
        # Stat: how concentrated is the weight on endgames?
        sorted_w = np.sort(weights)[::-1]
        cum = sorted_w.cumsum() / sorted_w.sum()
        n_for_50 = int((cum < 0.5).sum()) + 1
        n_for_90 = int((cum < 0.9).sum()) + 1
        print(
            f"  value-weight concentration: 50% in top {n_for_50} samples, "
            f"90% in top {n_for_90} (out of {n})"
        )

    history = []
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        # Endgame-only restriction for early epochs?
        if epoch <= endgame_only_epochs and distances is not None:
            mask = distances <= endgame_max_distance
            indices_pool = np.flatnonzero(mask)
            if indices_pool.size == 0:
                indices_pool = np.arange(n)
            mode = f"endgame≤{endgame_max_distance}"
        else:
            indices_pool = np.arange(n)
            mode = "all"

        perm = np_rng.permutation(indices_pool)
        ep_p = ep_v = 0.0
        ep_total = 0
        net.train()
        for i0 in range(0, perm.size, batch_size):
            idx = perm[i0 : i0 + batch_size]
            x = torch.from_numpy(features[idx]).to(device)
            pt = torch.from_numpy(policies[idx]).to(device)
            vt = torch.from_numpy(values[idx]).to(device)
            wt = torch.from_numpy(weights[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            logits, value_pred = net(x)
            # Policy loss: unweighted soft cross-entropy.
            log_probs = F.log_softmax(logits, dim=1)
            p_loss = -(pt * log_probs).sum(dim=1).mean()
            # Value loss: weighted MSE, mean over batch with weights normalized
            # so the loss magnitude stays comparable to unweighted MSE.
            v_diff = value_pred.squeeze(-1) - vt
            v_sq = v_diff * v_diff
            wt_sum = wt.sum()
            if wt_sum.item() > 0:
                v_loss = (v_sq * wt).sum() / wt_sum
            else:
                v_loss = (v_sq).mean()
            loss = p_loss + v_loss
            loss.backward()
            opt.step()
            ep_p += p_loss.item() * idx.size
            ep_v += v_loss.item() * idx.size
            ep_total += idx.size

        avg_p = ep_p / ep_total
        avg_v = ep_v / ep_total
        # Also report top-1 agreement with the argmax target (sanity).
        with torch.no_grad():
            x_all = torch.from_numpy(features[:1024]).to(device)
            pt_all = torch.from_numpy(policies[:1024]).to(device)
            logits, _ = net(x_all)
            argmax_match = (logits.argmax(1) == pt_all.argmax(1)).float().mean().item()
        print(
            f"  epoch {epoch}/{epochs} ({mode}, n={perm.size}): "
            f"policy_xent={avg_p:.4f} value_mse={avg_v:.4f} "
            f"argmax_match={argmax_match:.3f}"
        )
        history.append(
            {
                "epoch": epoch,
                "policy": avg_p,
                "value": avg_v,
                "match": argmax_match,
                "mode": mode,
                "n_used": int(perm.size),
            }
        )

    wall = time.monotonic() - start
    return net.cpu().eval(), {
        "wall_seconds": wall,
        "history": history,
        "final_policy_xent": history[-1]["policy"],
        "final_value_mse": history[-1]["value"],
        "final_argmax_match": history[-1]["match"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Distill HP-PUCT into MLP")
    p.add_argument("--output", required=True)
    p.add_argument("--initial-checkpoint", default=None)
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--hp-sims", type=int, default=50)
    p.add_argument("--n-workers", type=int, default=10)
    p.add_argument("--rollout-policy", choices=["random", "heuristic"], default="random")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--value-hidden", type=int, default=128)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument(
        "--value-decay",
        type=float,
        default=0.0,
        help=(
            "Endgame backchaining: weight per-sample value loss by "
            "exp(-distance/value_decay) where distance is plies-from-terminal. "
            "0 = unweighted (default). 8-15 is a reasonable range; smaller = "
            "more endgame-focused."
        ),
    )
    p.add_argument(
        "--endgame-only-epochs",
        type=int,
        default=0,
        help=(
            "First N epochs train ONLY on samples within --endgame-max-distance "
            "of terminal. Strict form of backchaining."
        ),
    )
    p.add_argument(
        "--endgame-max-distance",
        type=int,
        default=8,
        help="For --endgame-only-epochs: max plies from terminal to include.",
    )
    args = p.parse_args(argv)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("mps unavailable, falling back to cpu")
        device = "cpu"

    print(
        f"generating {args.games} HP-PUCT games "
        f"(sims={args.hp_sims}, workers={args.n_workers}, "
        f"rollout={args.rollout_policy})..."
    )
    features, policies, values, distances, stats = generate_hp_data(
        n_games=args.games,
        hp_sims=args.hp_sims,
        n_workers=args.n_workers,
        rollout_policy=args.rollout_policy,
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

    print("training...")
    net, train_stats = pretrain(
        features,
        policies,
        values,
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
        endgame_only_epochs=args.endgame_only_epochs,
        endgame_max_distance=args.endgame_max_distance,
    )

    net.save(
        out,
        extra={
            "source": "hp_puct_distill",
            "initial_checkpoint": args.initial_checkpoint,
            "data_stats": stats,
            "train_stats": train_stats,
        },
    )
    print(f"saved: {out} ({net.param_count():,} params)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
