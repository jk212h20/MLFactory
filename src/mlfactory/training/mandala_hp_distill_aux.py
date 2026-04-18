"""HP-PUCT distillation with smoothed value targets AND opponent-hand
auxiliary loss.

Builds on mandala_hp_distill_smooth by additionally training an
opp-hand prediction head. Rationale (per user observation): late-game
value is computable arithmetically from public state, but mid-game
outcomes depend critically on what's in opponent's hand. Even if the
aux head's predictions are noisy, training it forces the shared trunk
to learn representations that capture opp-hand-relevant patterns. The
value head benefits implicitly.

Usage:
    nohup uv run python -m mlfactory.training.mandala_hp_distill_aux \\
        --output deploy/checkpoints/mandala-hp-distill-smooth-aux.pt \\
        --games 500 --hp-sims 50 --n-workers 10 \\
        --smooth-completions 8 --smooth-rollouts 1 \\
        --epochs 12 --aux-weight 0.5 \\
        > /tmp/hp-distill-aux.log 2>&1 &

The aux-weight knob controls the strength of the auxiliary loss in
the total loss. 0.0 disables (degrades to mandala_hp_distill_smooth).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mlfactory.agents.alphazero_mlp import AlphaZeroMLP, MLPConfig
from mlfactory.games.mandala.actions import N_TEMPLATES
from mlfactory.games.mandala.encode import FEATURE_DIM
from mlfactory.games.mandala.rules import COLORS, MAX_HAND_SIZE
from mlfactory.training.mandala_hp_distill_smooth import (
    generate_hp_data_with_states,
)
from mlfactory.training.mandala_value_smooth import smooth_values_parallel


_AUX_BINS = MAX_HAND_SIZE + 1  # 0..8 inclusive = 9 bins
_COLOR_IDX = {c: i for i, c in enumerate(COLORS)}


def opp_hand_targets_from_states(
    state_cores: list[dict],
    movers: np.ndarray,
) -> np.ndarray:
    """For each (state, mover), compute opp's actual hand color counts.

    Returns an int64 array of shape (N, 6) where entry [i, c] is the
    number of cards of color c in opponent's hand at sample i. Clipped
    to [0, MAX_HAND_SIZE]. Used as cross-entropy target with 9 bins.
    """
    n = len(state_cores)
    targets = np.zeros((n, len(COLORS)), dtype=np.int64)
    for i, (core, mover) in enumerate(zip(state_cores, movers.tolist())):
        opp = core["players"][1 - int(mover)]
        for card in opp["hand"]:
            color = card.get("color")
            if color in _COLOR_IDX:
                targets[i, _COLOR_IDX[color]] += 1
    return np.clip(targets, 0, MAX_HAND_SIZE)


def pretrain_aux(
    features: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    aux_targets: np.ndarray,  # (N, 6) int64
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
    aux_weight: float,
) -> tuple[AlphaZeroMLP, dict]:
    """Train policy + value + opp-hand-aux jointly.

    Total loss = policy_xent + value_mse + aux_weight × aux_xent.
    """
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    cfg = MLPConfig(
        feature_dim=FEATURE_DIM,
        n_actions=N_TEMPLATES,
        hidden=hidden,
        n_blocks=n_blocks,
        value_hidden=value_hidden,
        aux_opp_hand=True,
        aux_opp_hand_bins=_AUX_BINS,
    )
    net = AlphaZeroMLP(cfg).to(device)
    if initial_checkpoint:
        prior, _ = AlphaZeroMLP.load(initial_checkpoint, map_location="cpu")
        # Allow loading from a non-aux checkpoint into an aux net (the aux
        # head's weights stay random).
        prior_sd = prior.state_dict()
        net_sd = net.state_dict()
        loaded = 0
        for k, v in prior_sd.items():
            if k in net_sd and net_sd[k].shape == v.shape:
                net_sd[k] = v
                loaded += 1
        net.load_state_dict(net_sd)
        print(f"warm-started {loaded}/{len(prior_sd)} parameters from {initial_checkpoint}")
        net = net.to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = features.shape[0]
    print(f"pretraining: {n} samples, {epochs} epochs, batch {batch_size}, aux_weight={aux_weight}")

    history = []
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        perm = np_rng.permutation(n)
        ep_p = ep_v = ep_a = 0.0
        ep_total = 0
        ep_aux_correct = 0
        net.train()
        for i0 in range(0, n, batch_size):
            idx = perm[i0 : i0 + batch_size]
            x = torch.from_numpy(features[idx]).to(device)
            pt = torch.from_numpy(policies[idx]).to(device)
            vt = torch.from_numpy(values[idx]).to(device)
            at = torch.from_numpy(aux_targets[idx]).to(device)  # (B, 6) int64

            opt.zero_grad(set_to_none=True)
            logits, value_pred, aux_logits = net.forward_with_aux(x)
            # Policy: soft cross-entropy over PUCT visits.
            log_probs = F.log_softmax(logits, dim=1)
            p_loss = -(pt * log_probs).sum(dim=1).mean()
            # Value: MSE.
            v_loss = F.mse_loss(value_pred.squeeze(-1), vt)
            # Aux: per-color cross-entropy. aux_logits: (B, 6, 9). Reshape
            # so cross_entropy treats each color as an independent
            # 9-class classification problem.
            B = aux_logits.shape[0]
            aux_flat = aux_logits.reshape(B * 6, _AUX_BINS)
            at_flat = at.reshape(B * 6)
            a_loss = F.cross_entropy(aux_flat, at_flat)
            loss = p_loss + v_loss + aux_weight * a_loss
            loss.backward()
            opt.step()

            ep_p += p_loss.item() * idx.size
            ep_v += v_loss.item() * idx.size
            ep_a += a_loss.item() * idx.size
            ep_total += idx.size
            with torch.no_grad():
                ep_aux_correct += (aux_flat.argmax(1) == at_flat).sum().item()

        avg_p = ep_p / ep_total
        avg_v = ep_v / ep_total
        avg_a = ep_a / ep_total
        aux_acc = ep_aux_correct / (ep_total * 6)
        with torch.no_grad():
            x_all = torch.from_numpy(features[:1024]).to(device)
            pt_all = torch.from_numpy(policies[:1024]).to(device)
            logits, _ = net(x_all)
            argmax_match = (logits.argmax(1) == pt_all.argmax(1)).float().mean().item()
        print(
            f"  epoch {epoch}/{epochs}: policy_xent={avg_p:.4f} "
            f"value_mse={avg_v:.4f} aux_xent={avg_a:.4f} "
            f"aux_acc={aux_acc:.3f} argmax_match={argmax_match:.3f}",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "policy": avg_p,
                "value": avg_v,
                "aux": avg_a,
                "aux_acc": aux_acc,
                "match": argmax_match,
            }
        )

    wall = time.monotonic() - start
    return net.cpu().eval(), {
        "wall_seconds": wall,
        "history": history,
        "final_policy_xent": history[-1]["policy"],
        "final_value_mse": history[-1]["value"],
        "final_aux_xent": history[-1]["aux"],
        "final_aux_acc": history[-1]["aux_acc"],
        "final_argmax_match": history[-1]["match"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="HP-PUCT distill with smoothed value + opp-hand aux head"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--initial-checkpoint", default=None)
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--hp-sims", type=int, default=50)
    p.add_argument("--n-workers", type=int, default=10)
    p.add_argument("--rollout-policy", choices=["random", "heuristic"], default="random")
    p.add_argument("--smooth-completions", type=int, default=8)
    p.add_argument("--smooth-rollouts", type=int, default=1)
    p.add_argument("--smooth-rollout-policy", choices=["random", "heuristic"], default="random")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--value-hidden", type=int, default=128)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument(
        "--aux-weight",
        type=float,
        default=0.5,
        help="Multiplier on the aux opp-hand loss in the total. 0 disables.",
    )
    args = p.parse_args(argv)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("mps unavailable, falling back to cpu")
        device = "cpu"

    print(f"generating {args.games} HP-PUCT games...")
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
    print(f"  generation: {stats['wall_seconds']:.0f}s, samples={stats['n_samples']}")

    print(f"smoothing value targets...")
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
    print(
        f"  smoothing done: range [{smoothed_values.min():.2f}, {smoothed_values.max():.2f}], "
        f"std={smoothed_values.std():.3f}"
    )

    print("computing opp-hand targets from states...")
    aux_targets = opp_hand_targets_from_states(state_cores, movers)
    print(
        f"  aux target distribution: mean count per color={aux_targets.mean():.2f}, "
        f"max={aux_targets.max()}"
    )

    print("training (smoothed values + opp-hand aux)...")
    net, train_stats = pretrain_aux(
        features,
        policies,
        smoothed_values,
        aux_targets,
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
        aux_weight=args.aux_weight,
    )

    net.save(
        out,
        extra={
            "source": "hp_puct_distill_smoothed_aux",
            "data_stats": stats,
            "train_stats": train_stats,
            "smooth_completions": args.smooth_completions,
            "smooth_rollouts": args.smooth_rollouts,
            "aux_weight": args.aux_weight,
        },
    )
    print(f"saved: {out} ({net.param_count():,} params)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
