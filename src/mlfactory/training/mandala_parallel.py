"""Parallel self-play for Mandala.

Mirrors the Boop parallel worker pattern (training/parallel.py) but
tailored to Mandala's MLP net + feature-based encoding + history.

Design:
- spawn context (macOS-safe; matches Boop setup)
- each worker sets torch single-threaded to avoid OpenMP batch-1 overhead
- workers receive net state_dict as bytes + MLPConfig as dict; rebuild
  net locally. No torch modules or closures cross the process line.
- worker plays ONE complete self-play game, returns (samples, record,
  winner, n_moves) back to parent.
"""

from __future__ import annotations

import io
import multiprocessing as mp
from dataclasses import dataclass

import torch


@dataclass
class MandalaSelfPlayJob:
    """Inputs for one self-play game run in a worker."""

    net_state_dict_bytes: bytes
    net_config_dict: dict
    n_simulations: int
    dirichlet_alpha: float
    dirichlet_epsilon: float
    temperature: float
    temperature_moves: int
    iter_index: int
    game_index: int
    seed: int
    game_name: str = "mandala"


def _worker_init() -> None:
    """Run once per worker at startup. Single-thread torch so batch-1
    forward passes don't fight themselves via OpenMP."""
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run_selfplay_job(job: MandalaSelfPlayJob) -> dict:
    """Worker entry point. Rebuilds net + env locally, plays game,
    returns a dict of {samples, record, winner, n_moves}.

    Imports happen inside the worker because torch + our modules can be
    slow to import; spawn context means each worker pays the import cost
    but we avoid pickling imports in the parent."""
    import random

    from mlfactory.agents.alphazero.agent import AlphaZeroAgent
    from mlfactory.agents.alphazero.evaluator import NetEvaluator
    from mlfactory.agents.alphazero.puct import PUCTConfig
    from mlfactory.agents.alphazero_mlp import AlphaZeroMLP, MLPConfig
    from mlfactory.games.mandala.env import MandalaEnv
    from mlfactory.training.trainer_mandala import (
        MandalaEncoderClosure,
        _play_one_game,
    )

    # Rebuild net from serialized state.
    cfg = MLPConfig(**job.net_config_dict)
    net = AlphaZeroMLP(cfg)
    state_dict = torch.load(
        io.BytesIO(job.net_state_dict_bytes),
        map_location="cpu",
        weights_only=True,
    )
    net.load_state_dict(state_dict)
    net = net.cpu().eval()

    encoder = MandalaEncoderClosure()
    evaluator = NetEvaluator(net, encoder=encoder, device="cpu", name="az-sp")

    agent = AlphaZeroAgent(
        evaluator,
        PUCTConfig(
            n_simulations=job.n_simulations,
            dirichlet_alpha=job.dirichlet_alpha,
            dirichlet_epsilon=job.dirichlet_epsilon,
        ),
        mode="sample",
        temperature=job.temperature,
        temperature_moves=job.temperature_moves,
        add_root_noise=True,
        seed=job.seed,
        name="az-sp",
    )

    env = MandalaEnv(rng=random.Random(job.seed))
    return _play_one_game(
        env,
        agent,
        encoder,
        game_name=job.game_name,
        iter_index=job.iter_index,
        game_index=job.game_index,
        seed=job.seed,
    )


def serialise_mlp(net) -> tuple[bytes, dict]:
    """Pack AlphaZeroMLP state into (bytes, config_dict) picklable payload."""
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    cfg = net.config
    cfg_dict = {
        "feature_dim": cfg.feature_dim,
        "n_actions": cfg.n_actions,
        "hidden": cfg.hidden,
        "n_blocks": cfg.n_blocks,
        "value_hidden": cfg.value_hidden,
    }
    return buf.getvalue(), cfg_dict


def parallel_selfplay(
    jobs: list[MandalaSelfPlayJob],
    n_workers: int,
) -> list[dict]:
    """Run all jobs in parallel workers. Returns results in completion order.

    Results retain their game_index via record.notes, so callers can sort
    back to submission order if needed.
    """
    if not jobs:
        return []
    if n_workers <= 1:
        return [_run_selfplay_job(j) for j in jobs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        return list(pool.imap_unordered(_run_selfplay_job, jobs))
