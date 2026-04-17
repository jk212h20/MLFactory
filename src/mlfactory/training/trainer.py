"""AlphaZero-lite trainer (single-process, for Phase 3c).

One iteration = self-play batch -> training minibatches -> eval -> checkpoint.
Events are emitted at every phase boundary so the watch TUI has a live view.

Subprocess entry point:
    python -m mlfactory.training.trainer --run-dir <dir> [options]

The trainer catches SIGTERM and SIGINT, finishes the current iteration
gracefully, saves a checkpoint, emits run_end, and exits.

Design decisions locked:
- Self-play runs CPU-side (batch-1 evaluations; see wiki/insights/2026-04-16-mps-cpu-crossover-small-nets.md).
- Training runs on the configured device (MPS by default; CPU fallback).
- Eval runs CPU-side for the same reason as self-play.
- Gate-less training: the freshly trained net always becomes the self-play
  player. Eval compares against `prev` (the checkpoint from the previous
  iteration) and a fixed `mcts200` baseline; both are logged but neither
  is used as a gate.
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
from typing import NamedTuple

import numpy as np
import torch

from mlfactory.agents.alphazero.agent import AlphaZeroAgent
from mlfactory.agents.alphazero.evaluator import NetEvaluator
from mlfactory.agents.alphazero.net import AlphaZeroNet, NetConfig
from mlfactory.agents.alphazero.puct import PUCTConfig
from mlfactory.agents.mcts import MCTSAgent
from mlfactory.agents.random_agent import RandomAgent
from mlfactory.games.boop import Boop
from mlfactory.games.boop.encode import encode_state, legal_mask
from mlfactory.runner.events import write_event
from mlfactory.runner.layout import RunLayout
from mlfactory.training.augment import augment_many
from mlfactory.training.parallel import (
    EvalJob,
    SelfPlayJob,
    parallel_eval,
    parallel_selfplay,
    serialise_net,
)
from mlfactory.training.replay_buffer import ReplayBuffer
from mlfactory.training.sample_game import write_game
from mlfactory.training.selfplay import play_selfplay_game
from mlfactory.training.train_step import mean_losses, train_step


# --- Graceful stop ---------------------------------------------------------
_stop_requested = False


def _handle_sigterm(signum: int, frame: object) -> None:  # noqa: ARG001
    global _stop_requested
    _stop_requested = True


# --- Config ---------------------------------------------------------------


@dataclass
class TrainerConfig:
    iters: int = 5
    selfplay_games: int = 20
    selfplay_sims: int = 100
    eval_games: int = 20
    eval_sims: int = 100
    baseline_mcts_sims: int = 200
    train_batches: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    replay_capacity: int = 50_000
    warmup_samples: int = 256  # don't train until buffer has at least this many
    temperature_moves: int = 8
    dirichlet_alpha: float = 0.5
    dirichlet_epsilon: float = 0.25
    net_blocks: int = 4
    net_channels: int = 64
    device: str = "mps"  # "mps" | "cpu" | "cuda"
    augment: bool = True
    seed: int = 0
    # sample-game writing
    samples_per_iter: int = 2
    # parallelism
    n_workers: int = 1  # 1 = legacy single-process path
    # eval cadence: how often to run the mcts baseline (every N iters)
    mcts_eval_every: int = 1
    random_eval_every: int = 1
    # Warm-start: load net weights from this checkpoint before training.
    # Replay buffer is NOT populated from the checkpoint (starts empty).
    # If set, the checkpoint's NetConfig must match net_blocks/net_channels.
    resume_from: str | None = None
    # Optional: evaluate periodically against a fixed "baseline" checkpoint
    # (typically the strongest prior champion). Separate from the per-iter
    # eval-vs-prev and vs-mcts paths so we can track long-horizon progress
    # without moving goalposts.
    baseline_ckpt: str | None = None
    baseline_ckpt_every: int = 5  # iters between baseline-ckpt evals
    baseline_ckpt_games: int = 20  # games per baseline eval match
    # Early stop: if eval-vs-baseline_ckpt score (wins / total) reaches
    # `stop_on_baseline_win_rate` for `stop_requires_consecutive` consecutive
    # baseline evals, exit gracefully (status=finished) before hitting iters.
    # 0 or unset disables.
    #
    # Note: this is a fixed-threshold rule. Run51 demonstrated empirically
    # that with 20-game evals it false-triggered on a model that was
    # actually parity (true win rate ~50%, observed 60% by chance). For
    # honest evidence-of-strength stopping, prefer stop_on_baseline_pvalue.
    stop_on_baseline_win_rate: float = 0.0
    stop_requires_consecutive: int = 1
    # Stricter stop rule: stop when the one-sided binomial test against
    # H0: true_win_rate=0.5 (using draws as 0.5 wins each — the standard
    # win-draw-loss scoring) yields a p-value <= this threshold.
    # Combines wins + 0.5*draws as the "score" and tests via normal
    # approximation (large N) or exact binomial for small N.
    # 0 or unset disables. 0.05 is a reasonable default; 0.01 is strict.
    # Recommended: pair with --baseline-ckpt-games 40 for fair power.
    stop_on_baseline_pvalue: float = 0.0


# --- Main -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    # PyTorch CPU threading: force single-thread for batch-1 inference.
    # For self-play / eval the net does thousands of batch-1 forward passes
    # per game; PyTorch's default multi-threaded CPU kernels have per-call
    # OpenMP fork/join overhead that dwarfs the actual compute on tiny
    # (6x6) inputs. We saw this pathology in the first realistic smoke
    # (40+ min stuck in batch_norm_cpu_kernel / pthread_cond_wait). Setting
    # intra- and inter-op threads to 1 fixes it. Training itself runs on
    # MPS (or GPU), unaffected by these knobs.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Build layout from --run-dir
    run_dir = Path(args.run_dir)
    run_id = run_dir.name
    game_name = run_dir.parent.name
    root = run_dir.parent.parent.parent
    layout = RunLayout(root=root, game=game_name, run_id=run_id)
    layout.ensure()

    cfg = _build_config(args)
    # Device fallback: drop to cpu if mps/cuda unavailable.
    cfg.device = _resolve_device(cfg.device)

    env = _make_env(game_name)
    rng_np = np.random.default_rng(cfg.seed)

    # Build net and optimizer.
    net_cfg = NetConfig(
        in_channels=_encoder_channels(game_name),
        board_h=6 if game_name == "boop" else 6,
        board_w=6 if game_name == "boop" else 7,
        n_actions=env.num_actions,
        num_blocks=cfg.net_blocks,
        channels=cfg.net_channels,
    )
    net = AlphaZeroNet(net_cfg).to(cfg.device)

    # Warm-start: load weights from a prior checkpoint. Replay buffer
    # stays empty on purpose — old self-play data came from a weaker
    # policy and would drag training backwards.
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-from not found: {resume_path}")
        prior_net, prior_extra = AlphaZeroNet.load(resume_path, map_location="cpu")
        if prior_net.config != net_cfg:
            raise ValueError(
                f"Checkpoint NetConfig {prior_net.config} does not match "
                f"trainer NetConfig {net_cfg}. Adjust --net-blocks / "
                f"--net-channels to match the checkpoint."
            )
        net.load_state_dict(prior_net.state_dict())
        net = net.to(cfg.device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Encoder for this game.
    encoder = _build_encoder(game_name)

    # Replay buffer (CPU memory).
    replay = ReplayBuffer(capacity=cfg.replay_capacity, rng=rng_np)

    # Keep a snapshot of the previous iteration's net for eval-vs-prev.
    prev_net: AlphaZeroNet | None = None

    # Fixed baseline opponent net (loaded once, reused across evals).
    # Kept on CPU — eval agents always run CPU-side per the phase 3b
    # MPS/batch-1 insight.
    baseline_net: AlphaZeroNet | None = None
    baseline_net_tag: str = "baseline"
    if cfg.baseline_ckpt:
        bpath = Path(cfg.baseline_ckpt)
        if not bpath.exists():
            raise FileNotFoundError(f"--baseline-ckpt not found: {bpath}")
        baseline_net, _ = AlphaZeroNet.load(bpath, map_location="cpu")
        baseline_net = baseline_net.cpu().eval()
        baseline_net_tag = f"baseline:{bpath.stem}"

    # Baseline-stop-rule tracking.
    consecutive_baseline_hits = 0
    last_baseline_score: float | None = None

    layout.write_status("running")
    write_event(
        layout.events_path,
        "run_start",
        trainer="alphazero",
        game=game_name,
        device=cfg.device,
        iters=cfg.iters,
        config={
            "selfplay_games": cfg.selfplay_games,
            "selfplay_sims": cfg.selfplay_sims,
            "eval_games": cfg.eval_games,
            "train_batches": cfg.train_batches,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "net_blocks": cfg.net_blocks,
            "net_channels": cfg.net_channels,
            "n_params": net.param_count(),
            "replay_capacity": cfg.replay_capacity,
            "augment": cfg.augment,
            "seed": cfg.seed,
            "resume_from": cfg.resume_from,
            "baseline_ckpt": cfg.baseline_ckpt,
            "baseline_ckpt_every": cfg.baseline_ckpt_every,
            "baseline_ckpt_games": cfg.baseline_ckpt_games,
            "stop_on_baseline_win_rate": cfg.stop_on_baseline_win_rate,
            "stop_requires_consecutive": cfg.stop_requires_consecutive,
            "stop_on_baseline_pvalue": cfg.stop_on_baseline_pvalue,
        },
    )
    if cfg.resume_from:
        write_event(
            layout.events_path,
            "log",
            level="info",
            msg=f"warm-started net weights from {cfg.resume_from}",
        )
    if cfg.baseline_ckpt:
        write_event(
            layout.events_path,
            "log",
            level="info",
            msg=f"baseline opponent: {cfg.baseline_ckpt} (every {cfg.baseline_ckpt_every} iters, {cfg.baseline_ckpt_games} games)",
        )

    # Flag flipped when the early-stop-by-baseline rule fires. Distinct
    # from _stop_requested (which represents SIGTERM) so the run_end can
    # report status="finished" for a goal-reached stop rather than "stopped".
    early_stop_hit = False

    run_start = time.monotonic()
    try:
        for it in range(1, cfg.iters + 1):
            if _stop_requested or early_stop_hit:
                break
            iter_t0 = time.monotonic()
            write_event(layout.events_path, "iter_start", iter=it)

            # ---------------- 1) SELF-PLAY (CPU) ----------------
            sp_t0 = time.monotonic()
            sp_samples, sp_record_paths, sp_stats = _run_selfplay(
                env=env,
                net=net,
                encoder=encoder,
                cfg=cfg,
                iter_index=it,
                seed=cfg.seed + it * 100,
                layout=layout,
                game_name=game_name,
            )
            if cfg.augment:
                sp_samples = augment_many(sp_samples, game=game_name)
            replay.extend(sp_samples)
            sp_elapsed = time.monotonic() - sp_t0
            write_event(
                layout.events_path,
                "selfplay",
                iter=it,
                games=cfg.selfplay_games,
                samples=len(sp_samples),
                avg_moves=round(sp_stats["avg_moves"], 2),
                orange_win_rate=round(sp_stats["orange_win_rate"], 3),
                duration_s=round(sp_elapsed, 2),
                buffer_size=len(replay),
            )
            for path in sp_record_paths:
                write_event(
                    layout.events_path,
                    "sample_game",
                    iter=it,
                    path=str(path.relative_to(layout.root)),
                    kind="selfplay",
                )

            # ---------------- 2) TRAIN (device) ----------------
            if len(replay) >= cfg.warmup_samples:
                tr_t0 = time.monotonic()
                losses = _run_training(
                    net=net,
                    optimizer=optimizer,
                    replay=replay,
                    cfg=cfg,
                )
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
                    msg=(
                        f"skipping training at iter {it}: buffer {len(replay)} "
                        f"< warmup {cfg.warmup_samples}"
                    ),
                )

            # ---------------- 3) CHECKPOINT ----------------
            ckpt_path = layout.checkpoints_dir / f"iter-{it:04d}.pt"
            net.save(
                ckpt_path,
                extra={
                    "iter": it,
                    "game": game_name,
                    "buffer_size": len(replay),
                },
            )
            write_event(
                layout.events_path,
                "checkpoint",
                iter=it,
                path=str(ckpt_path.relative_to(layout.root)),
                is_champion=False,  # gate-less training — champion decided post-hoc
            )

            # ---------------- 4) EVAL ----------------
            ev_t0 = time.monotonic()
            baseline_summary = _run_eval(
                env=env,
                game_name=game_name,
                encoder=encoder,
                net=net,
                prev_net=prev_net,
                cfg=cfg,
                iter_index=it,
                seed=cfg.seed + it * 17,
                layout=layout,
                baseline_net=baseline_net,
                baseline_tag=baseline_net_tag,
            )
            ev_elapsed = time.monotonic() - ev_t0
            write_event(
                layout.events_path,
                "log",
                level="info",
                msg=f"eval iter {it} completed in {ev_elapsed:.1f}s",
            )

            # Early-stop rule 1 (legacy): fixed-threshold + consecutive count.
            # Run51 demonstrated this false-triggers on noise; prefer the
            # p-value rule below for new runs.
            if baseline_summary is not None and cfg.stop_on_baseline_win_rate > 0.0:
                last_baseline_score = baseline_summary.score
                if baseline_summary.score >= cfg.stop_on_baseline_win_rate:
                    consecutive_baseline_hits += 1
                else:
                    consecutive_baseline_hits = 0
                if consecutive_baseline_hits >= cfg.stop_requires_consecutive:
                    write_event(
                        layout.events_path,
                        "log",
                        level="info",
                        msg=(
                            f"early stop (winrate rule): baseline win rate >= "
                            f"{cfg.stop_on_baseline_win_rate} for "
                            f"{consecutive_baseline_hits} consecutive evals "
                            f"(latest score={baseline_summary.score:.3f})"
                        ),
                    )
                    early_stop_hit = True

            # Early-stop rule 2 (statistical): one-sided binomial p-value
            # against H0: true_winrate=0.5. Stops only when the observation
            # is significantly inconsistent with parity. Single-trigger;
            # the threshold itself does the work that "consecutive" did
            # for the loose rule.
            if baseline_summary is not None and cfg.stop_on_baseline_pvalue > 0.0:
                pval = _binomial_p_value_one_sided(
                    baseline_summary.wins,
                    baseline_summary.draws,
                    baseline_summary.losses,
                )
                # Always emit the diagnostic so the watcher can see it.
                write_event(
                    layout.events_path,
                    "log",
                    level="info",
                    msg=(
                        f"baseline p-value: p={pval:.4f} "
                        f"(wins={baseline_summary.wins}, draws={baseline_summary.draws}, "
                        f"losses={baseline_summary.losses}, threshold={cfg.stop_on_baseline_pvalue})"
                    ),
                )
                if pval <= cfg.stop_on_baseline_pvalue:
                    write_event(
                        layout.events_path,
                        "log",
                        level="info",
                        msg=(
                            f"early stop (pvalue rule): one-sided binomial "
                            f"p={pval:.4f} <= {cfg.stop_on_baseline_pvalue} "
                            f"(score={baseline_summary.score:.3f}, "
                            f"n={baseline_summary.total})"
                        ),
                    )
                    early_stop_hit = True

            # Update prev_net for next iter's eval.
            prev_net = _clone_net(net, cfg.device)

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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaZero-lite trainer")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--selfplay-games", type=int, default=20)
    p.add_argument("--selfplay-sims", type=int, default=100)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-sims", type=int, default=100)
    p.add_argument("--baseline-mcts-sims", type=int, default=200)
    p.add_argument("--train-batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--replay-capacity", type=int, default=50_000)
    p.add_argument("--warmup-samples", type=int, default=256)
    p.add_argument("--temperature-moves", type=int, default=8)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    p.add_argument("--net-blocks", type=int, default=4)
    p.add_argument("--net-channels", type=int, default=64)
    p.add_argument("--device", default="mps")
    p.add_argument("--samples-per-iter", type=int, default=2)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Worker processes for self-play/eval. 1 = single-process.",
    )
    p.add_argument(
        "--mcts-eval-every",
        type=int,
        default=1,
        help="Run mcts baseline eval every N iters (1 = every iter).",
    )
    p.add_argument(
        "--random-eval-every", type=int, default=1, help="Run random baseline eval every N iters."
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint to warm-start net weights from. "
        "NetConfig shape must match --net-blocks / --net-channels.",
    )
    p.add_argument(
        "--baseline-ckpt",
        type=str,
        default=None,
        help="Path to a fixed opponent checkpoint. Evaluated every "
        "--baseline-ckpt-every iters. Tracks long-horizon progress.",
    )
    p.add_argument(
        "--baseline-ckpt-every",
        type=int,
        default=5,
        help="Iters between baseline-checkpoint evaluations.",
    )
    p.add_argument(
        "--baseline-ckpt-games",
        type=int,
        default=20,
        help="Games per baseline-checkpoint evaluation match.",
    )
    p.add_argument(
        "--stop-on-baseline-win-rate",
        type=float,
        default=0.0,
        help="If >0, stop training once eval-vs-baseline_ckpt reaches this "
        "fraction for --stop-requires-consecutive evaluations in a row.",
    )
    p.add_argument(
        "--stop-requires-consecutive",
        type=int,
        default=1,
        help="Number of consecutive baseline evals that must meet the "
        "stop-on-baseline-win-rate threshold before stopping.",
    )
    p.add_argument(
        "--stop-on-baseline-pvalue",
        type=float,
        default=0.0,
        help="If >0, stop training when one-sided binomial p-value vs "
        "H0:p=0.5 on the baseline-ckpt eval is <= this. 0.05 standard, "
        "0.01 strict. Pair with --baseline-ckpt-games >= 40 for power.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _build_config(args: argparse.Namespace) -> TrainerConfig:
    return TrainerConfig(
        iters=args.iters,
        selfplay_games=args.selfplay_games,
        selfplay_sims=args.selfplay_sims,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        baseline_mcts_sims=args.baseline_mcts_sims,
        train_batches=args.train_batches,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        replay_capacity=args.replay_capacity,
        warmup_samples=args.warmup_samples,
        temperature_moves=args.temperature_moves,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        net_blocks=args.net_blocks,
        net_channels=args.net_channels,
        device=args.device,
        augment=not args.no_augment,
        seed=args.seed,
        samples_per_iter=args.samples_per_iter,
        n_workers=args.n_workers,
        mcts_eval_every=args.mcts_eval_every,
        random_eval_every=args.random_eval_every,
        resume_from=args.resume_from,
        baseline_ckpt=args.baseline_ckpt,
        baseline_ckpt_every=args.baseline_ckpt_every,
        baseline_ckpt_games=args.baseline_ckpt_games,
        stop_on_baseline_win_rate=args.stop_on_baseline_win_rate,
        stop_requires_consecutive=args.stop_requires_consecutive,
        stop_on_baseline_pvalue=args.stop_on_baseline_pvalue,
    )


def _resolve_device(device: str) -> str:
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _make_env(game_name: str):
    if game_name == "boop":
        return Boop()
    raise ValueError(f"unsupported game '{game_name}' for AZ trainer (Phase 3)")


def _encoder_channels(game_name: str) -> int:
    if game_name == "boop":
        from mlfactory.games.boop.encode import N_PLANES

        return N_PLANES
    raise ValueError(game_name)


def _build_encoder(game_name: str):
    if game_name == "boop":

        def enc(state):
            return encode_state(state), legal_mask(state)

        return enc
    raise ValueError(game_name)


def _clone_net(net: AlphaZeroNet, device: str) -> AlphaZeroNet:
    clone = AlphaZeroNet(net.config).to(device)
    clone.load_state_dict(net.state_dict())
    clone.eval()
    return clone


def _make_selfplay_agent(
    net: AlphaZeroNet, encoder, cfg: TrainerConfig, seed: int
) -> AlphaZeroAgent:
    # Self-play uses CPU device for net evaluation (batch-1 is faster on CPU).
    cpu_net = AlphaZeroNet(net.config)
    cpu_net.load_state_dict(net.state_dict())
    cpu_net = cpu_net.cpu().eval()
    ev = NetEvaluator(cpu_net, encoder=encoder, device="cpu", name=f"net-it")
    return AlphaZeroAgent(
        ev,
        PUCTConfig(
            n_simulations=cfg.selfplay_sims,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_epsilon=cfg.dirichlet_epsilon,
        ),
        mode="sample",
        temperature=1.0,
        temperature_moves=cfg.temperature_moves,
        add_root_noise=True,
        seed=seed,
        name="az-sp",
    )


def _make_eval_agent(
    net: AlphaZeroNet, encoder, cfg: TrainerConfig, seed: int, name: str
) -> AlphaZeroAgent:
    """Eval agent: sample-mode with low temperature for first few moves then greedy.

    A fully-deterministic greedy-vs-greedy matchup can loop forever on games
    without repetition rules (like Boop), because the net + greedy selection
    is a pure function of state. A small amount of early-game sampling both
    breaks cycles and produces meaningful variance across the 10 eval games.
    """
    cpu_net = AlphaZeroNet(net.config)
    cpu_net.load_state_dict(net.state_dict())
    cpu_net = cpu_net.cpu().eval()
    ev = NetEvaluator(cpu_net, encoder=encoder, device="cpu", name=name)
    return AlphaZeroAgent(
        ev,
        PUCTConfig(n_simulations=cfg.eval_sims),
        mode="sample",
        temperature=1.0,
        temperature_moves=4,  # first 4 moves sampled, then greedy
        add_root_noise=False,
        seed=seed,
        name=name,
    )


def _run_selfplay(
    *,
    env,
    net,
    encoder,
    cfg: TrainerConfig,
    iter_index: int,
    seed: int,
    layout: RunLayout,
    game_name: str,
):
    """Play selfplay_games games, extract samples, save a few for replay.

    With n_workers > 1, games run in parallel worker processes.
    """
    samples_acc = []
    record_paths: list[Path] = []
    total_moves = 0
    orange_wins = 0
    n_decided = 0

    iter_samples_dir = layout.samples_dir / f"iter-{iter_index:04d}"
    iter_samples_dir.mkdir(parents=True, exist_ok=True)

    if cfg.n_workers > 1:
        net_bytes, net_cfg = serialise_net(net)
        jobs = [
            SelfPlayJob(
                game_name=game_name,
                net_state_dict_bytes=net_bytes,
                net_config_dict=net_cfg,
                n_simulations=cfg.selfplay_sims,
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_epsilon=cfg.dirichlet_epsilon,
                temperature=1.0,
                temperature_moves=cfg.temperature_moves,
                add_root_noise=True,
                iter_index=iter_index,
                game_index=g,
                seed=seed + g,
                max_moves=500,
                record_visits=True,
            )
            for g in range(cfg.selfplay_games)
        ]
        results = parallel_selfplay(jobs, n_workers=cfg.n_workers)
    else:
        results = []
        for g in range(cfg.selfplay_games):
            if _stop_requested:
                break
            agent = _make_selfplay_agent(net, encoder, cfg, seed=seed + g)
            results.append(
                play_selfplay_game(
                    env,
                    agent,
                    encoder=encoder,
                    game_name=game_name,
                    iter_index=iter_index,
                    seed=seed + g,
                    game_index=g,
                )
            )

    # Sort by game_index so sample-game indices are stable.
    results.sort(key=lambda r: r.record.notes.get("game_index", 0))

    for result in results:
        samples_acc.extend(result.samples)
        total_moves += result.n_moves
        if result.winner is not None:
            n_decided += 1
            if result.winner == 0:
                orange_wins += 1
        g = int(result.record.notes.get("game_index", 0))
        if g < cfg.samples_per_iter:
            path = iter_samples_dir / f"selfplay-game-{g:02d}.json"
            write_game(path, env=env, record=result.record)
            record_paths.append(path)

    n_games_completed = len(results)
    avg_moves = total_moves / max(n_games_completed, 1)
    orange_rate = orange_wins / max(n_decided, 1)
    return (
        samples_acc,
        record_paths,
        {
            "avg_moves": avg_moves,
            "orange_win_rate": orange_rate,
            "games_completed": n_games_completed,
        },
    )


def _run_training(*, net, optimizer, replay: ReplayBuffer, cfg: TrainerConfig):
    losses = []
    for _ in range(cfg.train_batches):
        if _stop_requested:
            break
        batch = replay.sample(cfg.batch_size)
        planes, policies, values = replay.stack(batch)
        loss = train_step(net, optimizer, planes, policies, values, device=cfg.device)
        losses.append(loss)
    return losses


def _run_eval(
    *,
    env,
    game_name: str,
    encoder,
    net: AlphaZeroNet,
    prev_net: AlphaZeroNet | None,
    cfg: TrainerConfig,
    iter_index: int,
    seed: int,
    layout: RunLayout,
    baseline_net: AlphaZeroNet | None = None,
    baseline_tag: str = "baseline",
) -> EvalSummary | None:
    """Evaluate vs prev checkpoint and (periodically) vs mcts baseline + random
    + optional fixed baseline-ckpt.

    With n_workers > 1, each eval match's games run in parallel.
    Cadence: `mcts_eval_every`, `random_eval_every`, `baseline_ckpt_every`
    control how often each baseline runs (always on iter 1 and on multiples
    of the cadence).

    Returns the baseline-ckpt win rate if it was evaluated this iter,
    else None. Used by the caller to implement early-stopping rules.
    """
    iter_samples_dir = layout.samples_dir / f"iter-{iter_index:04d}"
    iter_samples_dir.mkdir(parents=True, exist_ok=True)

    # --- Eval vs prev (cheap AZ-vs-AZ, every iter) ---
    if prev_net is not None:
        _eval_match_parallel(
            env=env,
            game_name=game_name,
            encoder=encoder,
            net=net,
            cfg=cfg,
            opponent_kind="az",
            opp_net=prev_net,
            opp_sims=cfg.eval_sims,
            opp_name="az-prev",
            tag="prev",
            n_games=cfg.eval_games,
            iter_index=iter_index,
            seed_base=seed,
            layout=layout,
        )
    else:
        write_event(
            layout.events_path,
            "log",
            level="info",
            msg=f"skipping eval-vs-prev at iter {iter_index} (no previous net)",
        )

    # --- Eval vs fixed baseline checkpoint (cadence-gated) ---
    baseline_summary: EvalSummary | None = None
    if baseline_net is not None and (iter_index == 1 or iter_index % cfg.baseline_ckpt_every == 0):
        baseline_summary = _eval_match_parallel(
            env=env,
            game_name=game_name,
            encoder=encoder,
            net=net,
            cfg=cfg,
            opponent_kind="az",
            opp_net=baseline_net,
            opp_sims=cfg.eval_sims,
            opp_name=baseline_tag,
            tag=baseline_tag,
            n_games=cfg.baseline_ckpt_games,
            iter_index=iter_index,
            seed_base=seed + 50,
            layout=layout,
        )

    # --- Eval vs MCTS baseline (cadence-gated) ---
    if iter_index == 1 or iter_index % cfg.mcts_eval_every == 0:
        _eval_match_parallel(
            env=env,
            game_name=game_name,
            encoder=encoder,
            net=net,
            cfg=cfg,
            opponent_kind="mcts",
            opp_net=None,
            opp_sims=cfg.baseline_mcts_sims,
            opp_name=f"mcts{cfg.baseline_mcts_sims}",
            tag=f"mcts{cfg.baseline_mcts_sims}",
            n_games=cfg.eval_games,
            iter_index=iter_index,
            seed_base=seed + 100,
            layout=layout,
        )
    else:
        write_event(
            layout.events_path,
            "log",
            level="info",
            msg=f"skipping mcts eval at iter {iter_index} (every {cfg.mcts_eval_every})",
        )

    # --- Eval vs random (cheap sanity, cadence-gated) ---
    if iter_index == 1 or iter_index % cfg.random_eval_every == 0:
        _eval_match_parallel(
            env=env,
            game_name=game_name,
            encoder=encoder,
            net=net,
            cfg=cfg,
            opponent_kind="random",
            opp_net=None,
            opp_sims=None,
            opp_name="random",
            tag="random",
            n_games=max(cfg.eval_games // 2, 6),
            iter_index=iter_index,
            seed_base=seed + 200,
            layout=layout,
        )

    return baseline_summary


def _eval_match_parallel(
    *,
    env,
    game_name: str,
    encoder,
    net: AlphaZeroNet,
    cfg: TrainerConfig,
    opponent_kind: str,
    opp_net: AlphaZeroNet | None,
    opp_sims: int | None,
    opp_name: str,
    tag: str,
    n_games: int,
    iter_index: int,
    seed_base: int,
    layout: RunLayout,
) -> EvalSummary | None:
    """Run one colour-balanced eval match, parallel if n_workers>1, sequential otherwise.

    Emits a single 'eval' event with aggregated results and returns an
    EvalSummary (wins, losses, draws) so callers can do statistical
    reasoning beyond raw win rate. Returns None if no games were played.
    """
    # Colour-balanced assignment: games alternate a-as-player-0 / a-as-player-1.
    a_is_p0 = [i % 2 == 0 for i in range(n_games)]

    if cfg.n_workers > 1:
        a_bytes, a_cfg = serialise_net(net)
        b_bytes = b_cfg = None
        if opp_net is not None:
            b_bytes, b_cfg = serialise_net(opp_net)
        jobs = [
            EvalJob(
                game_name=game_name,
                a_net_state_dict_bytes=a_bytes,
                a_net_config_dict=a_cfg,
                a_sims=cfg.eval_sims,
                a_temperature_moves=4,
                a_name=f"az-it{iter_index}",
                a_seed=seed_base + i,
                opponent_kind=opponent_kind,  # type: ignore[arg-type]
                b_net_state_dict_bytes=b_bytes,
                b_net_config_dict=b_cfg,
                b_sims=opp_sims,
                b_name=opp_name,
                b_seed=seed_base + 1000 + i,
                a_is_player_0=a_is_p0[i],
            )
            for i in range(n_games)
        ]
        game_results = parallel_eval(jobs, n_workers=cfg.n_workers, should_stop=_is_stop_requested)
        wins = sum(1 for r in game_results if r.a_won)
        losses = sum(1 for r in game_results if r.b_won)
        draws = sum(1 for r in game_results if r.drew)
        total = wins + losses + draws
    else:
        # Sequential fallback — delegate to the original play_match.
        cur_agent = _make_eval_agent(net, encoder, cfg, seed=seed_base, name=f"az-it{iter_index}")
        if opponent_kind == "az":
            assert opp_net is not None
            opp_agent = _make_eval_agent(opp_net, encoder, cfg, seed=seed_base + 1, name=opp_name)
        elif opponent_kind == "mcts":
            assert opp_sims is not None
            opp_agent = MCTSAgent(n_simulations=opp_sims, seed=seed_base + 1, name=opp_name)
        else:  # random
            opp_agent = RandomAgent(name=opp_name, seed=seed_base + 1)
        from mlfactory.tools.arena import play_match

        r = play_match(env, cur_agent, opp_agent, n_games=n_games, should_stop=_is_stop_requested)
        wins, losses, draws = int(r.a_wins), int(r.b_wins), int(r.draws)
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
    return EvalSummary(wins=wins, losses=losses, draws=draws) if total > 0 else None


class EvalSummary(NamedTuple):
    """Aggregated result of one eval match. Returned by _eval_match_parallel
    so callers can do statistical reasoning beyond raw win rate."""

    wins: int
    losses: int
    draws: int

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        return self.wins / max(self.total, 1)


def _is_stop_requested() -> bool:
    return _stop_requested


def _binomial_p_value_one_sided(wins: int, draws: int, losses: int) -> float:
    """One-sided binomial test against H0: true win rate = 0.5.

    Uses standard win-draw-loss scoring where each draw counts as 0.5
    points. We round the score to the nearest half-integer of "successes"
    out of total games and compute P(X >= observed | p=0.5) under
    Binomial(n, 0.5).

    Edge cases:
      - 0 games -> p=1.0 (no evidence)
      - score == n -> p = 0.5^n (all wins, exact)
      - score <= n/2 -> p = 1.0 (no evidence of strength above 50%)

    Returns p in [0, 1]. Lower = stronger evidence the model exceeds 0.5.
    """
    n = wins + draws + losses
    if n == 0:
        return 1.0
    # Score in points; treat each draw as 0.5
    score = wins + 0.5 * draws
    # Convert to integer "successes" by rounding up — half-points become
    # the conservative (closer-to-null) integer when ambiguous, so we
    # never overstate evidence. With even draws this is exact.
    # Equivalent: k = ceil(score). If score is exact integer, use it.
    if abs(score - round(score)) < 1e-9:
        k = int(round(score))
    else:
        # Unusual: odd number of draws. Round DOWN (toward null) for
        # conservatism; we'd rather under-report evidence than over-report.
        k = int(math.floor(score))
    if k <= n / 2:
        return 1.0
    # P(X >= k | p=0.5) = sum_{i=k..n} C(n,i) * 0.5^n
    # Equivalently 0.5^n * sum_{i=k..n} C(n,i)
    log_half_n = n * math.log(0.5)
    # Compute log of sum of binomials for numerical stability with large n
    # but n is small here (40-100), exact arithmetic is fine
    total = 0
    for i in range(k, n + 1):
        total += math.comb(n, i)
    return total * (0.5**n)


if __name__ == "__main__":
    sys.exit(main())
