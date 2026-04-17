"""Heuristic-bootstrap pretraining for Mandala.

Cold-start AZ self-play from a random-init net is expensive on Mandala:
games are long (~100 moves), action space is 150, and random play
produces many draws from soft-locks. Iter 1 value signal is noisy and
policy targets from PUCT with a random-init prior are near-uniform.

Instead: pretrain the MLP to imitate a rule-based heuristic bot, then
start AZ self-play from that pretrained net. This skips ~5 iters of
cold-start thrashing.

Pipeline:
  1. Play N games where BOTH sides use the HeuristicMandalaAgent with
     small random perturbation (epsilon-greedy) for data diversity.
  2. For each game, collect (features_at_ply_t, action_at_ply_t,
     final_outcome_from_mover_perspective) tuples from BOTH sides.
     Optionally filter to only winning-side moves.
  3. Supervised training: cross-entropy on the heuristic's action
     (one-hot over 150 templates) + MSE on the final outcome. N epochs,
     MLP config matching the eventual self-play net.
  4. Save the pretrained checkpoint; AZ self-play resume-from points
     here.

Run as:
    uv run python -m mlfactory.training.mandala_bootstrap \\
        --output deploy/checkpoints/mandala-bootstrap.pt \\
        --games 500 --epochs 6 --epsilon 0.10
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mlfactory.agents.alphazero_mlp import AlphaZeroMLP, MLPConfig
from mlfactory.games.mandala.actions import (
    N_TEMPLATES,
    index_to_template,
    legal_template_indices,
    template_to_index,
)
from mlfactory.games.mandala.encode import (
    FEATURE_DIM,
    encode_view,
    make_history,
    record_action,
)
from mlfactory.games.mandala.env import MandalaEnv, MandalaState
from mlfactory.games.mandala.heuristic_agent import HeuristicMandalaAgent
from mlfactory.games.mandala.rules import get_player_view


def generate_bootstrap_data(
    n_games: int,
    epsilon: float,
    base_seed: int,
    winners_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Play n_games heuristic-vs-heuristic games. For each ply t, record:
      - features_t: encoded player-view at ply t
      - action_t:   the heuristic's chosen action (template index)
      - value_t:    final outcome from the mover-at-ply-t's perspective

    With probability `epsilon`, override the heuristic's pick with a
    uniform-random legal action (adds diversity to the training set).
    If `winners_only=True`, only ply t's from the eventual winner's
    moves are kept — much cleaner supervised signal at the cost of half
    the data.

    Returns (features, actions, values, stats).
    """
    all_features: list[np.ndarray] = []
    all_actions: list[int] = []
    all_values: list[float] = []

    n_finished = 0
    n_timed_out = 0
    n_wins_p0 = 0

    t0 = time.monotonic()
    for g in range(n_games):
        env = MandalaEnv(rng=random.Random(base_seed + g))
        agents = [
            HeuristicMandalaAgent(seed=base_seed + g * 2),
            HeuristicMandalaAgent(seed=base_seed + g * 2 + 1),
        ]
        state = env.initial_state()
        game_rng = random.Random(base_seed + g + 999999)

        ply_features: list[np.ndarray] = []
        ply_actions: list[int] = []
        ply_movers: list[int] = []
        history: list[dict] = make_history()

        turn = 0
        while not state.is_terminal and turn < 300:
            legal = env.legal_actions(state)
            if not legal:
                break
            view = get_player_view(state.core, state.to_play)
            features = encode_view(view, state.to_play, history)

            # Pick action: usually the heuristic, sometimes random (epsilon).
            if game_rng.random() < epsilon:
                action = game_rng.choice(legal)
            else:
                action = agents[state.to_play].act(env, state)

            ply_features.append(features)
            ply_actions.append(action)
            ply_movers.append(state.to_play)

            # Advance history BEFORE stepping env so next iteration's
            # encoded view reflects this action in history. We append
            # with (template_index, actor_index) matching encode_view's
            # contract.
            record_action(history, action, state.to_play)

            state = env.step(state, action)
            turn += 1

        # Outcome
        if state.is_terminal and state.winner is not None:
            n_finished += 1
            winner = state.winner
            if winner == 0:
                n_wins_p0 += 1
            for feats, act, mover in zip(ply_features, ply_actions, ply_movers):
                if winners_only and mover != winner:
                    continue
                value = 1.0 if mover == winner else -1.0
                all_features.append(feats)
                all_actions.append(act)
                all_values.append(value)
        else:
            n_timed_out += 1
            # Treat as draw: value 0 for both sides. Include either all
            # or none. For winners_only, skip entirely (no clear signal).
            if not winners_only:
                for feats, act, mover in zip(ply_features, ply_actions, ply_movers):
                    all_features.append(feats)
                    all_actions.append(act)
                    all_values.append(0.0)

    wall = time.monotonic() - t0
    stats = {
        "n_games": n_games,
        "n_finished": n_finished,
        "n_timed_out": n_timed_out,
        "n_wins_p0": n_wins_p0,
        "n_samples": len(all_features),
        "wall_seconds": wall,
    }
    return (
        np.stack(all_features, axis=0).astype(np.float32),
        np.array(all_actions, dtype=np.int64),
        np.array(all_values, dtype=np.float32),
        stats,
    )


def pretrain(
    features: np.ndarray,
    actions: np.ndarray,
    values: np.ndarray,
    *,
    hidden: int,
    n_blocks: int,
    value_hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> tuple[AlphaZeroMLP, dict]:
    """Supervised pretraining.
    Returns (net, training_stats)."""
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
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    n = features.shape[0]
    print(f"pretraining: {n} samples, {epochs} epochs, batch {batch_size}")
    losses_hist = []
    acc_hist = []

    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        perm = np_rng.permutation(n)
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        net.train()
        for i0 in range(0, n, batch_size):
            idx = perm[i0 : i0 + batch_size]
            x = torch.from_numpy(features[idx]).to(device)
            y_a = torch.from_numpy(actions[idx]).to(device)
            y_v = torch.from_numpy(values[idx]).to(device)

            opt.zero_grad(set_to_none=True)
            policy_logits, value_pred = net(x)
            # Policy: standard cross-entropy (hard target = action index).
            p_loss = F.cross_entropy(policy_logits, y_a)
            v_loss = F.mse_loss(value_pred.squeeze(-1), y_v)
            loss = p_loss + v_loss
            loss.backward()
            opt.step()

            epoch_policy_loss += p_loss.item() * idx.size
            epoch_value_loss += v_loss.item() * idx.size
            pred = policy_logits.argmax(dim=1)
            epoch_correct += (pred == y_a).sum().item()
            epoch_total += idx.size

        avg_p = epoch_policy_loss / epoch_total
        avg_v = epoch_value_loss / epoch_total
        acc = epoch_correct / epoch_total
        print(
            f"  epoch {epoch}/{epochs}: policy_loss={avg_p:.4f} "
            f"value_loss={avg_v:.4f} top1_acc={acc:.3f}"
        )
        losses_hist.append({"epoch": epoch, "policy": avg_p, "value": avg_v, "acc": acc})

    wall = time.monotonic() - start
    return net.cpu().eval(), {
        "wall_seconds": wall,
        "final_policy_loss": losses_hist[-1]["policy"],
        "final_value_loss": losses_hist[-1]["value"],
        "final_top1_acc": losses_hist[-1]["acc"],
        "history": losses_hist,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Heuristic-bootstrap pretrain for Mandala")
    p.add_argument(
        "--output", type=str, required=True, help="Where to save the pretrained checkpoint (.pt)."
    )
    p.add_argument("--games", type=int, default=500, help="Number of heuristic-vs-heuristic games.")
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.10,
        help="Probability of a random (vs heuristic) action for diversity.",
    )
    p.add_argument(
        "--epochs", type=int, default=6, help="Supervised training epochs over the collected data."
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--value-hidden", type=int, default=128)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--include-losers",
        action="store_true",
        help="Include loser-side moves too (2x data, noisier signal).",
    )
    args = p.parse_args(argv)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve device (mps -> cpu fallback if unavailable).
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("mps unavailable, falling back to cpu")
        device = "cpu"

    print(f"generating {args.games} heuristic games (epsilon={args.epsilon})...")
    features, actions, values, stats = generate_bootstrap_data(
        n_games=args.games,
        epsilon=args.epsilon,
        base_seed=args.seed,
        winners_only=not args.include_losers,
    )
    print(
        f"  done in {stats['wall_seconds']:.1f}s: "
        f"{stats['n_finished']}/{stats['n_games']} finished, "
        f"{stats['n_timed_out']} timed out, p0 wins={stats['n_wins_p0']}"
    )
    print(f"  collected {stats['n_samples']} (feature, action, value) tuples")

    print("pretraining MLP...")
    net, train_stats = pretrain(
        features,
        actions,
        values,
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        value_hidden=args.value_hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        seed=args.seed,
    )

    # Save
    net.save(
        output_path,
        extra={
            "source": "heuristic_bootstrap",
            "data_stats": stats,
            "train_stats": train_stats,
            "generation_seed": args.seed,
        },
    )
    print(f"saved pretrained net: {output_path} ({net.param_count():,} params)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
