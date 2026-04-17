"""Counterfactual-rollout bootstrap layer for Mandala.

Concept (per user's suggestion 2026-04-17):
  When a fast random/heuristic game ends with a close score, rewind N
  moves and try alternative actions. For each alternative, play out
  with random/heuristic policy until terminal, record the outcome.
  Whichever alternative had the best expected outcome (averaged over
  rollouts to reduce stochastic-deck variance) becomes a target action
  for that position. The position-action-outcome triple goes into the
  supervised training set.

Why this is powerful:
- Source games are CHEAP (random/heuristic, no PUCT). Generates many
  close games per second.
- Counterfactual targets are STRICTLY BETTER than what was played, on
  average. That's stronger supervision than either AZ self-play visit
  distributions (which depend on a possibly-bad net) or pure imitation
  (which can only match the heuristic).
- ENDGAME FOCUS preserved: rewinding only N moves means we're near
  terminal where the value signal is least noisy.

Usage as a Layer-2 bootstrap on top of the heuristic-imitation layer:
    1. Generate close source games (~1000) via heuristic-vs-heuristic
       with random tiebreaking.
    2. For each close game, take last N=10 plies as branch points.
    3. For each branch point, try the top-K=8 legal alternative
       actions; play out R=4 rollouts per alternative (random policy).
    4. Compute average final outcome per alternative.
    5. Output sample = (features at branch point, target_action =
       argmax over alternatives, value = best_avg_outcome).
    6. Append to the bootstrap dataset; supervised pretrain.

Run as:
    uv run python -m mlfactory.training.mandala_counterfactual \\
        --output deploy/checkpoints/mandala-cf-bootstrap.pt \\
        --source-games 1000 --close-margin 4 --rewind 10 \\
        --branch-k 8 --rollouts 4 --epochs 6
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mlfactory.agents.alphazero_mlp import AlphaZeroMLP, MLPConfig
from mlfactory.games.mandala.actions import N_TEMPLATES, legal_template_indices
from mlfactory.games.mandala.encode import (
    FEATURE_DIM,
    encode_view,
    make_history,
    record_action,
)
from mlfactory.games.mandala.env import MandalaEnv, MandalaState
from mlfactory.games.mandala.heuristic_agent import HeuristicMandalaAgent
from mlfactory.games.mandala.rules import (
    calculate_score,
    get_player_view,
    get_winner,
)


# --- Source-game generation -----------------------------------------------


def play_source_game(
    base_seed: int,
    use_heuristic: bool = True,
    epsilon: float = 0.10,
) -> dict:
    """Play one cheap game (heuristic or random) and return the full trajectory.

    Returns:
        {
            'states': [MandalaState, ...]  # before each move + final
            'history_at_each_state': [list[dict], ...] # encoder history snapshots
            'actions': [int, ...]
            'movers': [int, ...]
            'winner': int | None,
            'final_score': (s0, s1)
        }

    Trajectory is stored so that branching from any position re-uses the
    exact pre-move state and history; no need to replay from scratch.
    """
    env = MandalaEnv(rng=random.Random(base_seed))
    rng = random.Random(base_seed + 999_999)
    if use_heuristic:
        h0 = HeuristicMandalaAgent(seed=base_seed * 2)
        h1 = HeuristicMandalaAgent(seed=base_seed * 2 + 1)

    state = env.initial_state()
    states: list[MandalaState] = [state]
    histories: list[list[dict]] = [list(state.history)]  # snapshot
    actions: list[int] = []
    movers: list[int] = []
    turn = 0
    while not state.is_terminal and turn < 300:
        legal = env.legal_actions(state)
        if not legal:
            break
        if use_heuristic and rng.random() >= epsilon:
            action = (h0 if state.to_play == 0 else h1).act(env, state)
        else:
            action = rng.choice(legal)
        actions.append(action)
        movers.append(state.to_play)
        state = env.step(state, action)
        states.append(state)
        histories.append(list(state.history))
        turn += 1

    if state.is_terminal:
        winner = state.winner
        s0 = calculate_score(state.core["players"][0])
        s1 = calculate_score(state.core["players"][1])
    else:
        winner = None
        s0 = s1 = 0
    return {
        "states": states,
        "histories": histories,
        "actions": actions,
        "movers": movers,
        "winner": winner,
        "final_score": (s0, s1),
        "n_moves": len(actions),
    }


# --- Counterfactual rollouts -----------------------------------------------


def rollout_value_for_mover(
    env: MandalaEnv,
    state: MandalaState,
    mover: int,
    rng: random.Random,
    use_heuristic: bool = True,
    max_moves: int = 200,
) -> float:
    """Play out from `state` with random/heuristic policy, return value
    (+1/-1/0) from `mover`'s perspective."""
    cur = state
    if use_heuristic:
        h0 = HeuristicMandalaAgent(seed=rng.randint(0, 2**31 - 1))
        h1 = HeuristicMandalaAgent(seed=rng.randint(0, 2**31 - 1))
    n = 0
    while not cur.is_terminal and n < max_moves:
        legal = env.legal_actions(cur)
        if not legal:
            break
        if use_heuristic:
            a = (h0 if cur.to_play == 0 else h1).act(env, cur)
        else:
            a = rng.choice(legal)
        cur = env.step(cur, a)
        n += 1
    if not cur.is_terminal:
        return 0.0  # treat as draw
    w = cur.winner
    if w is None:
        return 0.0
    return 1.0 if w == mover else -1.0


def evaluate_alternatives_at(
    state: MandalaState,
    branch_k: int,
    rollouts_per_branch: int,
    use_heuristic_rollouts: bool,
    rng: random.Random,
) -> tuple[int, float, dict[int, float]]:
    """At `state`, try up to branch_k legal alternatives. For each, play
    rollouts_per_branch rollouts and average the outcome from the
    current mover's perspective. Return (best_action, best_value,
    per_action_value_dict).

    If fewer than branch_k legal actions, all are tried.
    """
    env = MandalaEnv(rng=random.Random(rng.randint(0, 2**31 - 1)))
    legal = legal_template_indices(state.core)
    if not legal:
        return -1, 0.0, {}

    # Pick branch_k actions: prefer all if few, else uniform-random subset.
    if len(legal) <= branch_k:
        chosen = legal
    else:
        chosen = rng.sample(legal, branch_k)

    mover = state.to_play
    per_action_value: dict[int, float] = {}

    for a in chosen:
        # Apply the candidate action with a fresh local rng for the
        # transition (deck reshuffles need an rng).
        try:
            next_state = env.step(state, a, rng=random.Random(rng.randint(0, 2**31 - 1)))
        except Exception:
            continue
        outcomes: list[float] = []
        for _ in range(rollouts_per_branch):
            v = rollout_value_for_mover(
                env,
                next_state,
                mover=mover,
                rng=random.Random(rng.randint(0, 2**31 - 1)),
                use_heuristic=use_heuristic_rollouts,
            )
            outcomes.append(v)
        per_action_value[a] = sum(outcomes) / len(outcomes)

    if not per_action_value:
        return -1, 0.0, {}
    best_action = max(per_action_value, key=per_action_value.get)
    return best_action, per_action_value[best_action], per_action_value


# --- Bootstrap dataset assembly --------------------------------------------


def generate_counterfactual_data(
    n_source_games: int,
    close_margin: int,
    rewind: int,
    branch_k: int,
    rollouts_per_branch: int,
    use_heuristic_rollouts: bool,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate (features, action, value) tuples via counterfactual rollouts
    on close games.

    Returns (features, actions, values, stats)."""
    all_features: list[np.ndarray] = []
    all_actions: list[int] = []
    all_values: list[float] = []
    n_close = 0
    n_branch_points = 0
    t0 = time.monotonic()

    for g in range(n_source_games):
        if g % 50 == 0 and g > 0:
            print(
                f"  source game {g}/{n_source_games}: "
                f"{n_close} close, {len(all_features)} samples, "
                f"{time.monotonic() - t0:.0f}s"
            )
        traj = play_source_game(base_seed + g)
        s0, s1 = traj["final_score"]
        margin = abs(s0 - s1)
        if traj["winner"] is None or margin > close_margin:
            continue
        n_close += 1

        # Rewind from end: take last `rewind` plies as branch points.
        n_moves = traj["n_moves"]
        start_ply = max(0, n_moves - rewind)
        branch_rng = random.Random(base_seed + g * 100 + 31)

        for ply in range(start_ply, n_moves):
            state_at_ply = traj["states"][ply]
            history_at_ply = traj["histories"][ply]

            # Compute features once per branch point.
            mover = state_at_ply.to_play
            view = get_player_view(state_at_ply.core, mover)
            features = encode_view(view, mover, history_at_ply)

            # Counterfactual rollouts.
            best_action, best_value, _ = evaluate_alternatives_at(
                state_at_ply,
                branch_k=branch_k,
                rollouts_per_branch=rollouts_per_branch,
                use_heuristic_rollouts=use_heuristic_rollouts,
                rng=branch_rng,
            )
            if best_action < 0:
                continue
            n_branch_points += 1
            all_features.append(features)
            all_actions.append(best_action)
            all_values.append(best_value)

    wall = time.monotonic() - t0
    stats = {
        "n_source_games": n_source_games,
        "n_close_games": n_close,
        "n_branch_points": n_branch_points,
        "n_samples": len(all_features),
        "wall_seconds": wall,
        "close_margin_threshold": close_margin,
    }
    return (
        np.stack(all_features, axis=0).astype(np.float32)
        if all_features
        else np.zeros((0, FEATURE_DIM), dtype=np.float32),
        np.array(all_actions, dtype=np.int64),
        np.array(all_values, dtype=np.float32),
        stats,
    )


# --- Pretrain (reused pattern from mandala_bootstrap.py) -------------------


def pretrain(
    features: np.ndarray,
    actions: np.ndarray,
    values: np.ndarray,
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
) -> tuple[AlphaZeroMLP, dict]:
    """Supervised training. If `initial_checkpoint` is set, load weights
    from there before training (Layer-2 stacks on Layer-1)."""
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
            raise ValueError(
                f"checkpoint config {prior.config} doesn't match {cfg}; "
                f"adjust --hidden / --n-blocks / --value-hidden"
            )
        net.load_state_dict(prior.state_dict())
        net = net.to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = features.shape[0]
    print(f"pretraining: {n} samples, {epochs} epochs, batch {batch_size}")
    history = []
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        perm = np_rng.permutation(n)
        epoch_p = epoch_v = epoch_correct = 0.0
        epoch_total = 0
        net.train()
        for i0 in range(0, n, batch_size):
            idx = perm[i0 : i0 + batch_size]
            x = torch.from_numpy(features[idx]).to(device)
            y_a = torch.from_numpy(actions[idx]).to(device)
            y_v = torch.from_numpy(values[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            policy_logits, value_pred = net(x)
            p_loss = F.cross_entropy(policy_logits, y_a)
            v_loss = F.mse_loss(value_pred.squeeze(-1), y_v)
            loss = p_loss + v_loss
            loss.backward()
            opt.step()
            epoch_p += p_loss.item() * idx.size
            epoch_v += v_loss.item() * idx.size
            epoch_correct += (policy_logits.argmax(dim=1) == y_a).sum().item()
            epoch_total += idx.size
        avg_p = epoch_p / epoch_total
        avg_v = epoch_v / epoch_total
        acc = epoch_correct / epoch_total
        print(
            f"  epoch {epoch}/{epochs}: policy_loss={avg_p:.4f} value_loss={avg_v:.4f} top1_acc={acc:.3f}"
        )
        history.append({"epoch": epoch, "policy": avg_p, "value": avg_v, "acc": acc})

    wall = time.monotonic() - start
    return net.cpu().eval(), {
        "wall_seconds": wall,
        "history": history,
        "final_policy_loss": history[-1]["policy"],
        "final_value_loss": history[-1]["value"],
        "final_top1_acc": history[-1]["acc"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Counterfactual-rollout bootstrap for Mandala")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--initial-checkpoint",
        default=None,
        help="Stack on top of this checkpoint (e.g. the heuristic-imitation bootstrap).",
    )
    p.add_argument("--source-games", type=int, default=1000)
    p.add_argument(
        "--close-margin",
        type=int,
        default=4,
        help="Only games where final |score difference| <= this are 'close'.",
    )
    p.add_argument(
        "--rewind",
        type=int,
        default=10,
        help="Number of late-game plies to use as branch points per close game.",
    )
    p.add_argument(
        "--branch-k",
        type=int,
        default=8,
        help="Max alternative actions to try at each branch point.",
    )
    p.add_argument(
        "--rollouts", type=int, default=4, help="Rollouts per alternative for outcome estimation."
    )
    p.add_argument(
        "--random-rollouts",
        action="store_true",
        help="Use random (vs heuristic) rollout policy for evaluating alternatives.",
    )
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Lower default than layer-1 since we may be fine-tuning a checkpoint.",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--value-hidden", type=int, default=128)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("mps unavailable, falling back to cpu")
        device = "cpu"

    print(f"generating {args.source_games} source games (heuristic, epsilon=0.10)...")
    print(f"  filtering for close games (margin <= {args.close_margin}),")
    print(
        f"  rewinding {args.rewind} plies, trying {args.branch_k} alternatives x {args.rollouts} rollouts"
    )
    features, actions, values, stats = generate_counterfactual_data(
        n_source_games=args.source_games,
        close_margin=args.close_margin,
        rewind=args.rewind,
        branch_k=args.branch_k,
        rollouts_per_branch=args.rollouts,
        use_heuristic_rollouts=not args.random_rollouts,
        base_seed=args.seed,
    )
    print(
        f"  done in {stats['wall_seconds']:.0f}s: "
        f"{stats['n_close_games']}/{stats['n_source_games']} close games, "
        f"{stats['n_branch_points']} branch points, "
        f"{stats['n_samples']} samples"
    )

    if features.shape[0] == 0:
        print("no samples generated — try --close-margin larger or --source-games more")
        return 1

    print("supervised training...")
    net, train_stats = pretrain(
        features,
        actions,
        values,
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
    )

    net.save(
        out,
        extra={
            "source": "counterfactual_bootstrap",
            "initial_checkpoint": args.initial_checkpoint,
            "data_stats": stats,
            "train_stats": train_stats,
        },
    )
    print(f"saved: {out} ({net.param_count():,} params)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
