"""AlphaZero-lite trainer for Mandala.

Parallel but Mandala-specific variant of training/trainer.py. The shared
trainer is tightly coupled to Boop's 2D-board CNN architecture, and
refactoring to dispatch on game would be a larger change than we want
while the Boop pipeline is actively serving production.

This Mandala trainer:
- Uses AlphaZeroMLP instead of AlphaZeroNet (flat feature vector).
- Encodes state via mandala.encode.encode_view (takes a player-view
  state plus action history).
- Maintains action history explicitly across self-play moves.
- Reuses play_selfplay_game (game-agnostic), train_step, ReplayBuffer,
  runner events, and all the layout/log infrastructure.
- Starts single-process (no parallel workers yet). Parallel is easy to
  add later by mirroring training/parallel.py.
- No data augmentation (Mandala has no obvious symmetry to exploit).
- Subprocess-safe: entry point matches the runner launcher's expectation
  (--run-dir + other CLI flags), emits the standard event log.
"""

from __future__ import annotations

import argparse
import math
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mlfactory.agents.alphazero.agent import AlphaZeroAgent
from mlfactory.agents.alphazero.evaluator import NetEvaluator
from mlfactory.agents.alphazero.puct import PUCTConfig
from mlfactory.agents.alphazero_mlp import AlphaZeroMLP, MLPConfig
from mlfactory.games.mandala.encode import FEATURE_DIM, encode_view, make_history
from mlfactory.games.mandala.env import MandalaEnv, MandalaState
from mlfactory.games.mandala.rules import get_player_view
from mlfactory.games.mandala.actions import N_TEMPLATES
from mlfactory.runner.events import write_event
from mlfactory.runner.layout import RunLayout
from mlfactory.training.replay_buffer import ReplayBuffer
from mlfactory.training.sample_game import GameRecord, MoveRecord
from mlfactory.training.train_step import mean_losses, train_step

# --- Graceful stop ---------------------------------------------------------
_stop_requested = False


def _handle_sigterm(signum: int, frame: object) -> None:  # noqa: ARG001
    global _stop_requested
    _stop_requested = True


# --- Config ---------------------------------------------------------------


@dataclass
class MandalaTrainerConfig:
    iters: int = 5
    selfplay_games: int = 20
    selfplay_sims: int = 50
    eval_games: int = 10
    eval_sims: int = 50
    train_batches: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    replay_capacity: int = 50_000
    warmup_samples: int = 256
    temperature_moves: int = 8
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    hidden: int = 256
    n_blocks: int = 4
    value_hidden: int = 128
    device: str = "mps"
    seed: int = 0
    samples_per_iter: int = 2
    resume_from: str | None = None
    baseline_ckpt: str | None = None
    baseline_ckpt_every: int = 5
    baseline_ckpt_games: int = 20
    stop_on_baseline_pvalue: float = 0.0


# --- Mandala-specific encoder wrapper for NetEvaluator ---------------------


class MandalaEncoderClosure:
    """Adapts encode_view() to the NetEvaluator's (state) -> (features, mask)
    contract. Takes a MandalaState, applies the get_player_view mask from
    that state's current-to-move's perspective, encodes, and returns.

    Stateless aside from identifying the active player — encode_view only
    reads from the passed state (+ its history)."""

    def __init__(self) -> None:
        pass

    def __call__(self, state: MandalaState) -> tuple[np.ndarray, np.ndarray]:
        mover = state.to_play
        view = get_player_view(state.core, mover)
        features = encode_view(view, mover, state.history)
        # Legal mask over the 150-template vocabulary.
        from mlfactory.games.mandala.actions import legal_mask

        mask = legal_mask(state.core)
        return features, mask


# --- Main -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    # Match the Boop trainer's threading hygiene: batch-1 inference is
    # MUCH faster single-threaded. See wiki/insights for the pathology.
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # Build layout from --run-dir (same convention as the Boop trainer).
    run_dir = Path(args.run_dir)
    run_id = run_dir.name
    game_name = run_dir.parent.name
    root = run_dir.parent.parent.parent
    layout = RunLayout(root=root, game=game_name, run_id=run_id)
    layout.ensure()

    cfg = _build_config(args)
    cfg.device = _resolve_device(cfg.device)

    # Deterministic RNG for the game engine. Game-level seeding is per-game
    # (seeded off cfg.seed + iter + game_index) to give self-play variety
    # while keeping reproducibility.
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    # Build net.
    mlp_cfg = MLPConfig(
        feature_dim=FEATURE_DIM,
        n_actions=N_TEMPLATES,
        hidden=cfg.hidden,
        n_blocks=cfg.n_blocks,
        value_hidden=cfg.value_hidden,
    )
    net = AlphaZeroMLP(mlp_cfg).to(cfg.device)

    # Warm-start if requested.
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-from not found: {resume_path}")
        prior_net, _ = AlphaZeroMLP.load(resume_path, map_location="cpu")
        if prior_net.config != mlp_cfg:
            raise ValueError(
                f"Checkpoint MLPConfig {prior_net.config} does not match "
                f"trainer MLPConfig {mlp_cfg}."
            )
        net.load_state_dict(prior_net.state_dict())
        net = net.to(cfg.device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Baseline checkpoint for periodic eval.
    baseline_net: AlphaZeroMLP | None = None
    baseline_tag = "baseline"
    if cfg.baseline_ckpt:
        bpath = Path(cfg.baseline_ckpt)
        if not bpath.exists():
            raise FileNotFoundError(f"--baseline-ckpt not found: {bpath}")
        baseline_net, _ = AlphaZeroMLP.load(bpath, map_location="cpu")
        baseline_net = baseline_net.cpu().eval()
        baseline_tag = f"baseline:{bpath.stem}"

    replay = ReplayBuffer(capacity=cfg.replay_capacity, rng=np_rng)

    layout.write_status("running")
    write_event(
        layout.events_path,
        "run_start",
        trainer="alphazero_mandala",
        game=game_name,
        device=cfg.device,
        iters=cfg.iters,
        config={
            "selfplay_games": cfg.selfplay_games,
            "selfplay_sims": cfg.selfplay_sims,
            "eval_games": cfg.eval_games,
            "eval_sims": cfg.eval_sims,
            "train_batches": cfg.train_batches,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "hidden": cfg.hidden,
            "n_blocks": cfg.n_blocks,
            "n_params": net.param_count(),
            "replay_capacity": cfg.replay_capacity,
            "seed": cfg.seed,
            "resume_from": cfg.resume_from,
            "baseline_ckpt": cfg.baseline_ckpt,
            "feature_dim": FEATURE_DIM,
            "n_templates": N_TEMPLATES,
        },
    )

    encoder = MandalaEncoderClosure()

    run_start = time.monotonic()
    prev_net: AlphaZeroMLP | None = None
    early_stop_hit = False

    try:
        for it in range(1, cfg.iters + 1):
            if _stop_requested or early_stop_hit:
                break
            iter_t0 = time.monotonic()
            write_event(layout.events_path, "iter_start", iter=it)

            # ---- SELFPLAY ----
            sp_t0 = time.monotonic()
            samples, sample_records, sp_stats = _run_selfplay(
                net=net,
                encoder_fn=encoder,
                cfg=cfg,
                iter_index=it,
                seed=cfg.seed + it * 100,
                layout=layout,
                game_name=game_name,
            )
            replay.extend(samples)
            sp_elapsed = time.monotonic() - sp_t0
            write_event(
                layout.events_path,
                "selfplay",
                iter=it,
                games=cfg.selfplay_games,
                samples=len(samples),
                avg_moves=round(sp_stats["avg_moves"], 2),
                p0_win_rate=round(sp_stats["p0_win_rate"], 3),
                duration_s=round(sp_elapsed, 2),
                buffer_size=len(replay),
            )
            for path in sample_records:
                write_event(
                    layout.events_path,
                    "sample_game",
                    iter=it,
                    path=str(path.relative_to(layout.root)),
                    kind="selfplay",
                )

            # ---- TRAIN ----
            if len(replay) >= cfg.warmup_samples:
                tr_t0 = time.monotonic()
                losses = []
                for _ in range(cfg.train_batches):
                    if _stop_requested:
                        break
                    batch = replay.sample(cfg.batch_size)
                    planes, policies, values = replay.stack(batch)
                    loss = train_step(net, optimizer, planes, policies, values, device=cfg.device)
                    losses.append(loss)
                tr_elapsed = time.monotonic() - tr_t0
                avg = mean_losses(losses)
                write_event(
                    layout.events_path,
                    "train",
                    iter=it,
                    batches=len(losses),
                    policy_loss=round(avg.policy, 4),
                    value_loss=round(avg.value, 4),
                    total_loss=round(avg.total, 4),
                    policy_entropy=round(avg.policy_entropy, 4),
                    value_abs_mean=round(avg.value_abs_mean, 4),
                    value_std=round(avg.value_std, 4),
                    lr=cfg.lr,
                    duration_s=round(tr_elapsed, 2),
                )
            else:
                write_event(
                    layout.events_path,
                    "log",
                    level="info",
                    msg=f"skipping training at iter {it}: buffer {len(replay)} < warmup {cfg.warmup_samples}",
                )

            # ---- CHECKPOINT ----
            ckpt_path = layout.checkpoints_dir / f"iter-{it:04d}.pt"
            net.save(ckpt_path, extra={"iter": it, "game": game_name})
            write_event(
                layout.events_path,
                "checkpoint",
                iter=it,
                path=str(ckpt_path.relative_to(layout.root)),
                is_champion=False,
            )

            # ---- EVAL ----
            ev_t0 = time.monotonic()
            # vs prev
            if prev_net is not None:
                _eval_match(
                    net=net,
                    opp_net=prev_net,
                    encoder_fn=encoder,
                    cfg=cfg,
                    n_games=cfg.eval_games,
                    tag="prev",
                    iter_index=it,
                    seed_base=cfg.seed + it * 17,
                    layout=layout,
                )
            # vs baseline (cadence-gated)
            baseline_summary = None
            if baseline_net is not None and (it == 1 or it % cfg.baseline_ckpt_every == 0):
                baseline_summary = _eval_match(
                    net=net,
                    opp_net=baseline_net,
                    encoder_fn=encoder,
                    cfg=cfg,
                    n_games=cfg.baseline_ckpt_games,
                    tag=baseline_tag,
                    iter_index=it,
                    seed_base=cfg.seed + it * 17 + 50,
                    layout=layout,
                )
            ev_elapsed = time.monotonic() - ev_t0
            write_event(
                layout.events_path,
                "log",
                level="info",
                msg=f"eval iter {it} completed in {ev_elapsed:.1f}s",
            )

            # p-value stop rule (reuse helper logic)
            if baseline_summary is not None and cfg.stop_on_baseline_pvalue > 0.0:
                pval = _binomial_p_value_one_sided(
                    baseline_summary["wins"],
                    baseline_summary["draws"],
                    baseline_summary["losses"],
                )
                write_event(
                    layout.events_path,
                    "log",
                    level="info",
                    msg=(
                        f"baseline p-value: p={pval:.4f} "
                        f"(wins={baseline_summary['wins']}, draws={baseline_summary['draws']}, "
                        f"losses={baseline_summary['losses']}, threshold={cfg.stop_on_baseline_pvalue})"
                    ),
                )
                if pval <= cfg.stop_on_baseline_pvalue:
                    write_event(
                        layout.events_path,
                        "log",
                        level="info",
                        msg=f"early stop: p={pval:.4f} <= {cfg.stop_on_baseline_pvalue}",
                    )
                    early_stop_hit = True

            # prev update
            prev_net = AlphaZeroMLP(mlp_cfg).to(cfg.device)
            prev_net.load_state_dict(net.state_dict())
            prev_net.eval()

            write_event(
                layout.events_path,
                "iter_end",
                iter=it,
                duration_s=round(time.monotonic() - iter_t0, 2),
            )

        duration = time.monotonic() - run_start
        status = "stopped" if _stop_requested else "finished"
        layout.write_status(status)
        write_event(
            layout.events_path,
            "run_end",
            status=status,
            duration_s=round(duration, 2),
        )
        return 0

    except Exception as e:  # noqa: BLE001
        layout.write_status("crashed")
        import traceback

        tb = traceback.format_exc()
        write_event(layout.events_path, "log", level="error", msg=f"crashed: {e}\n{tb}")
        write_event(
            layout.events_path,
            "run_end",
            status="crashed",
            duration_s=round(time.monotonic() - run_start, 2),
        )
        raise


# --- Helpers ---------------------------------------------------------------


def _run_selfplay(
    *,
    net,
    encoder_fn,
    cfg: MandalaTrainerConfig,
    iter_index: int,
    seed: int,
    layout: RunLayout,
    game_name: str,
):
    """Run cfg.selfplay_games self-play games sequentially. Single-process
    for simplicity; parallel can be added later."""
    from mlfactory.training.replay_buffer import Sample
    from mlfactory.training.sample_game import write_game

    samples: list[Sample] = []
    record_paths = []
    total_moves = 0
    p0_wins = 0
    n_decided = 0

    iter_samples_dir = layout.samples_dir / f"iter-{iter_index:04d}"
    iter_samples_dir.mkdir(parents=True, exist_ok=True)

    # Fresh CPU evaluator for the current net (self-play runs cpu-side).
    cpu_net = AlphaZeroMLP(net.config)
    cpu_net.load_state_dict(net.state_dict())
    cpu_net = cpu_net.cpu().eval()
    evaluator = NetEvaluator(cpu_net, encoder=encoder_fn, device="cpu", name="az-sp")

    for g in range(cfg.selfplay_games):
        if _stop_requested:
            break
        env_rng = random.Random(seed + g)
        env = MandalaEnv(rng=env_rng)

        agent = AlphaZeroAgent(
            evaluator,
            PUCTConfig(
                n_simulations=cfg.selfplay_sims,
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_epsilon=cfg.dirichlet_epsilon,
            ),
            mode="sample",
            temperature=1.0,
            temperature_moves=cfg.temperature_moves,
            add_root_noise=True,
            seed=seed + g,
            name="az-sp",
        )

        result = _play_one_game(
            env,
            agent,
            encoder_fn,
            game_name=game_name,
            iter_index=iter_index,
            game_index=g,
            seed=seed + g,
        )
        samples.extend(result["samples"])
        total_moves += result["n_moves"]
        if result["winner"] is not None:
            n_decided += 1
            if result["winner"] == 0:
                p0_wins += 1

        if g < cfg.samples_per_iter:
            path = iter_samples_dir / f"selfplay-game-{g:02d}.json"
            write_game(path, env=env, record=result["record"])
            record_paths.append(path)

    return (
        samples,
        record_paths,
        {
            "avg_moves": total_moves / max(cfg.selfplay_games, 1),
            "p0_win_rate": p0_wins / max(n_decided, 1),
        },
    )


def _play_one_game(
    env: MandalaEnv,
    agent: AlphaZeroAgent,
    encoder_fn,
    *,
    game_name: str,
    iter_index: int,
    game_index: int,
    seed: int,
) -> dict:
    """Play one self-play game. Generic shape matches
    training.selfplay.play_selfplay_game but tailored for Mandala (dict
    state, player-view encoding, imperfect-info)."""
    from mlfactory.training.replay_buffer import Sample

    agent.reset()
    state = env.initial_state()

    # Collect per-move training data.
    per_move_features: list[np.ndarray] = []
    per_move_policy: list[np.ndarray] = []
    per_move_mover: list[int] = []
    moves: list[MoveRecord] = []
    states_dumps: list[dict] = [_mandala_state_to_dump(state)]

    ply = 0
    max_moves = 300
    while not state.is_terminal and ply < max_moves:
        # Rare but real: Mandala can reach non-terminal states with zero
        # legal actions (deck exhaustion + hand exhaustion). Abort the
        # game cleanly as a draw rather than crashing PUCT.
        legal = env.legal_actions(state)
        if not legal:
            break
        features, _ = encoder_fn(state)
        action = agent.act(env, state)
        search = agent.last_search
        if search is not None:
            policy_target = search.policy_target.copy()
        else:
            policy_target = np.zeros(env.num_actions, dtype=np.float32)
            policy_target[action] = 1.0

        per_move_features.append(features)
        per_move_policy.append(policy_target)
        per_move_mover.append(state.to_play)

        moves.append(
            MoveRecord(
                ply=ply,
                to_play=state.to_play,
                action=int(action),
                visits=(
                    {int(a): int(n) for a, n in search.root_visits.items()}
                    if search is not None
                    else None
                ),
                q_values=(
                    {int(a): float(q) for a, q in search.root_q.items()}
                    if search is not None
                    else None
                ),
                root_value=(float(search.root_value) if search is not None else None),
            )
        )

        state = env.step(state, action)
        states_dumps.append(_mandala_state_to_dump(state))
        ply += 1

    winner = state.winner if state.is_terminal else None
    if winner is None:
        result = "draw"
    else:
        result = "a_win" if winner == 0 else "b_win"

    samples = []
    for feat, policy, mover in zip(per_move_features, per_move_policy, per_move_mover):
        if winner is None:
            value_target = 0.0
        else:
            value_target = 1.0 if winner == mover else -1.0
        samples.append(Sample(planes=feat, policy_target=policy, value_target=value_target))

    record = GameRecord(
        game=game_name,
        iter=iter_index,
        kind="selfplay",
        agent_a=agent.name,
        agent_b=agent.name,
        seed=seed,
        result=result,
        winner=winner,
        moves=moves,
        states=states_dumps,
        notes={"n_simulations": agent.config.n_simulations, "game_index": game_index},
    )
    return {
        "samples": samples,
        "record": record,
        "winner": winner,
        "n_moves": ply,
    }


def _mandala_state_to_dump(state: MandalaState) -> dict:
    """Compact JSON-serializable dump of a Mandala state for sample-game
    playback. We stash the raw dict state plus the history."""
    import copy

    return {
        "kind": "mandala",
        "core": copy.deepcopy(state.core),
        "history": list(state.history),
    }


def _eval_match(
    *,
    net,
    opp_net,
    encoder_fn,
    cfg: MandalaTrainerConfig,
    n_games: int,
    tag: str,
    iter_index: int,
    seed_base: int,
    layout: RunLayout,
) -> dict | None:
    """Run a colour-balanced eval match. Single-process. Emits an 'eval'
    event and returns (wins, losses, draws) summary dict (or None if empty)."""
    # CPU copies of both nets for batch-1 evaluator speed.
    cur_cpu = AlphaZeroMLP(net.config)
    cur_cpu.load_state_dict(net.state_dict())
    cur_cpu = cur_cpu.cpu().eval()
    opp_cpu = AlphaZeroMLP(opp_net.config)
    opp_cpu.load_state_dict(opp_net.state_dict())
    opp_cpu = opp_cpu.cpu().eval()

    cur_eval = NetEvaluator(cur_cpu, encoder=encoder_fn, device="cpu", name=f"az-it{iter_index}")
    opp_eval = NetEvaluator(opp_cpu, encoder=encoder_fn, device="cpu", name=tag)

    def make_agent(ev, sd):
        return AlphaZeroAgent(
            ev,
            PUCTConfig(n_simulations=cfg.eval_sims),
            mode="sample",
            temperature=1.0,
            temperature_moves=4,
            add_root_noise=False,
            seed=sd,
            name=ev.name,
        )

    wins = losses = draws = 0
    for i in range(n_games):
        if _stop_requested:
            break
        env_rng = random.Random(seed_base + i)
        env = MandalaEnv(rng=env_rng)
        a_first = i % 2 == 0
        a_agent = make_agent(cur_eval, seed_base + i)
        b_agent = make_agent(opp_eval, seed_base + 10000 + i)
        if a_first:
            p0, p1 = a_agent, b_agent
        else:
            p0, p1 = b_agent, a_agent
        p0.reset()
        p1.reset()

        state = env.initial_state()
        agents = (p0, p1)
        n = 0
        while not state.is_terminal and n < 300:
            # Defensive guard: if somehow we've run out of legal actions
            # but the state isn't terminal (shouldn't happen in Mandala
            # but we saw it in a sanity run — needs investigation), abort
            # the game as a draw rather than crashing the whole trainer.
            legal = env.legal_actions(state)
            if not legal:
                write_event(
                    layout.events_path,
                    "log",
                    level="warn",
                    msg=(
                        f"eval-match {tag} game {i}: no legal actions at "
                        f"turn {n}, phase={state.core['phase']}, "
                        f"to_play={state.to_play}. Treating as draw."
                    ),
                )
                break
            action = agents[state.to_play].act(env, state)
            state = env.step(state, action)
            n += 1

        if not state.is_terminal:
            draws += 1
        else:
            winner = state.winner
            # Map back: a is either player 0 or player 1
            a_won = (a_first and winner == 0) or (not a_first and winner == 1)
            if a_won:
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1

    total = wins + losses + draws
    score = wins / max(total, 1)
    write_event(
        layout.events_path,
        "eval",
        iter=iter_index,
        opponent=tag,
        wins=wins,
        losses=losses,
        draws=draws,
        games=total,
        score=round(score, 3),
    )
    if total == 0:
        return None
    return {"wins": wins, "losses": losses, "draws": draws, "score": score}


def _binomial_p_value_one_sided(wins: int, draws: int, losses: int) -> float:
    """Same rule used in the Boop trainer. See tests/test_training/test_binomial.py."""
    n = wins + draws + losses
    if n == 0:
        return 1.0
    score = wins + 0.5 * draws
    if abs(score - round(score)) < 1e-9:
        k = int(round(score))
    else:
        k = int(math.floor(score))
    if k <= n / 2:
        return 1.0
    total = sum(math.comb(n, i) for i in range(k, n + 1))
    return total * (0.5**n)


def _resolve_device(device: str) -> str:
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mandala AlphaZero-lite trainer")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--selfplay-games", type=int, default=20)
    p.add_argument("--selfplay-sims", type=int, default=50)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--eval-sims", type=int, default=50)
    p.add_argument("--train-batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--replay-capacity", type=int, default=50_000)
    p.add_argument("--warmup-samples", type=int, default=256)
    p.add_argument("--temperature-moves", type=int, default=8)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--value-hidden", type=int, default=128)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples-per-iter", type=int, default=2)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--baseline-ckpt", type=str, default=None)
    p.add_argument("--baseline-ckpt-every", type=int, default=5)
    p.add_argument("--baseline-ckpt-games", type=int, default=20)
    p.add_argument("--stop-on-baseline-pvalue", type=float, default=0.0)
    return p.parse_args(argv)


def _build_config(args: argparse.Namespace) -> MandalaTrainerConfig:
    return MandalaTrainerConfig(
        iters=args.iters,
        selfplay_games=args.selfplay_games,
        selfplay_sims=args.selfplay_sims,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        train_batches=args.train_batches,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        replay_capacity=args.replay_capacity,
        warmup_samples=args.warmup_samples,
        temperature_moves=args.temperature_moves,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        value_hidden=args.value_hidden,
        device=args.device,
        seed=args.seed,
        samples_per_iter=args.samples_per_iter,
        resume_from=args.resume_from,
        baseline_ckpt=args.baseline_ckpt,
        baseline_ckpt_every=args.baseline_ckpt_every,
        baseline_ckpt_games=args.baseline_ckpt_games,
        stop_on_baseline_pvalue=args.stop_on_baseline_pvalue,
    )


if __name__ == "__main__":
    sys.exit(main())
