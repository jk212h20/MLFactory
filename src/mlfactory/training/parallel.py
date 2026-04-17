"""Parallel self-play and evaluation via multiprocessing.

Each worker process plays one complete game using a replica of the current
net (reconstructed from a pickled state_dict) and returns the samples +
GameRecord. The parent process orchestrates via a Pool.

Design points:
- Workers run on **CPU only**. Self-play / eval are batch-1 inference which
  is faster on CPU than MPS (see wiki/insights/2026-04-16-mps-cpu-crossover-small-nets.md).
  This also avoids the "MPS context can't be shared across processes" issue.
- Each worker calls `torch.set_num_threads(1)` at startup to prevent the
  OpenMP batch-1 pathology (see the same insight).
- We use spawn (not fork) on macOS because torch doesn't tolerate fork
  cleanly (shared CUDA/MPS handles break; allocator is not fork-safe).
- Only serialisable types cross the process boundary: bytes (state_dict
  pickled), config dicts, seeds, game names. No live objects (net, agent,
  env) — those are rebuilt inside the worker.
- One worker = one game at a time. Workers are reused within a call via
  `Pool.map` (imap_unordered gives us progress updates).
"""

from __future__ import annotations

import io
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from typing import Callable, Literal

import torch

from mlfactory.agents.alphazero.agent import AlphaZeroAgent
from mlfactory.agents.alphazero.evaluator import NetEvaluator
from mlfactory.agents.alphazero.net import AlphaZeroNet
from mlfactory.agents.alphazero.puct import PUCTConfig
from mlfactory.agents.mcts import MCTSAgent
from mlfactory.agents.random_agent import RandomAgent
from mlfactory.tools.arena import play_game
from mlfactory.training.sample_game import GameRecord, MoveRecord, state_to_dict
from mlfactory.training.selfplay import SelfPlayResult, play_selfplay_game


# -- shared job specs -------------------------------------------------------
#
# All fields must be picklable (no torch modules, no closures).


@dataclass
class SelfPlayJob:
    """One self-play game to run in a worker."""

    game_name: str  # "boop" etc.
    net_state_dict_bytes: bytes  # torch.save(net.state_dict()) output
    net_config_dict: dict  # NetConfig fields, used to reconstruct net
    n_simulations: int
    dirichlet_alpha: float
    dirichlet_epsilon: float
    temperature: float
    temperature_moves: int
    add_root_noise: bool
    iter_index: int
    game_index: int
    seed: int
    max_moves: int = 500
    record_visits: bool = True


@dataclass
class EvalJob:
    """One eval game in a worker.

    opponent_kind ∈ {"az", "mcts", "random"}. Everything needed to rebuild
    the two agents is in this struct; nothing else crosses the process line.
    """

    game_name: str
    # Agent A (always the current net under evaluation)
    a_net_state_dict_bytes: bytes
    a_net_config_dict: dict
    a_sims: int
    a_temperature_moves: int
    a_name: str
    a_seed: int
    # Agent B (the opponent)
    opponent_kind: Literal["az", "mcts", "random"]
    b_net_state_dict_bytes: bytes | None = None
    b_net_config_dict: dict | None = None
    b_sims: int | None = None
    b_name: str = "opp"
    b_seed: int = 0
    # How colours are assigned
    a_is_player_0: bool = True
    # Game cap (to survive cycles that somehow slip past the agent config)
    move_cap: int | None = None


@dataclass
class EvalGameResult:
    """Result of one eval game."""

    a_is_player_0: bool
    winner: int | None  # 0 or 1 (player index) or None for draw
    moves_played: int
    a_won: bool
    b_won: bool
    drew: bool


# -- worker-side: reconstruct objects from job specs -----------------------


def _worker_init() -> None:
    """Run once per worker process on startup.

    Sets torch to single-threaded per the phase-3b MPS/OpenMP insight.
    `set_num_interop_threads` can only be called once per process before
    any parallel work starts — wrapping in try/except so we don't die if
    the child inherited a torch that's already done some work.
    """
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _make_env(game_name: str):
    if game_name == "boop":
        from mlfactory.games.boop import Boop

        return Boop()
    raise ValueError(f"unsupported game in worker: {game_name}")


def _make_encoder(game_name: str) -> Callable:
    if game_name == "boop":
        from mlfactory.games.boop.encode import encode_state, legal_mask

        def enc(state):
            return encode_state(state), legal_mask(state)

        return enc
    raise ValueError(game_name)


def _net_from_job(net_state_dict_bytes: bytes, net_config_dict: dict) -> AlphaZeroNet:
    from mlfactory.agents.alphazero.net import NetConfig

    cfg = NetConfig(**net_config_dict)
    net = AlphaZeroNet(cfg)
    state_dict = torch.load(io.BytesIO(net_state_dict_bytes), map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    return net.cpu().eval()


# -- worker-side entry points ----------------------------------------------


def _run_selfplay_job(job: SelfPlayJob) -> SelfPlayResult:
    """Worker entry point for one self-play game. Must be top-level for pickling."""
    env = _make_env(job.game_name)
    encoder = _make_encoder(job.game_name)
    net = _net_from_job(job.net_state_dict_bytes, job.net_config_dict)
    ev = NetEvaluator(net, encoder=encoder, device="cpu", name="net")
    agent = AlphaZeroAgent(
        ev,
        PUCTConfig(
            n_simulations=job.n_simulations,
            dirichlet_alpha=job.dirichlet_alpha,
            dirichlet_epsilon=job.dirichlet_epsilon,
        ),
        mode="sample",
        temperature=job.temperature,
        temperature_moves=job.temperature_moves,
        add_root_noise=job.add_root_noise,
        seed=job.seed,
        name="az-sp",
    )
    return play_selfplay_game(
        env,
        agent,
        encoder=encoder,
        game_name=job.game_name,
        iter_index=job.iter_index,
        game_index=job.game_index,
        seed=job.seed,
        max_moves=job.max_moves,
        record_visits=job.record_visits,
    )


def _run_eval_job(job: EvalJob) -> EvalGameResult:
    """Worker entry point for one eval game."""
    env = _make_env(job.game_name)
    encoder = _make_encoder(job.game_name)

    # Agent A is always the az-under-test.
    a_net = _net_from_job(job.a_net_state_dict_bytes, job.a_net_config_dict)
    a_ev = NetEvaluator(a_net, encoder=encoder, device="cpu", name=job.a_name)
    a = AlphaZeroAgent(
        a_ev,
        PUCTConfig(n_simulations=job.a_sims),
        mode="sample",
        temperature=1.0,
        temperature_moves=job.a_temperature_moves,
        add_root_noise=False,
        seed=job.a_seed,
        name=job.a_name,
    )

    # Agent B is opponent_kind-specific.
    if job.opponent_kind == "az":
        assert job.b_net_state_dict_bytes is not None
        assert job.b_net_config_dict is not None
        assert job.b_sims is not None
        b_net = _net_from_job(job.b_net_state_dict_bytes, job.b_net_config_dict)
        b_ev = NetEvaluator(b_net, encoder=encoder, device="cpu", name=job.b_name)
        b = AlphaZeroAgent(
            b_ev,
            PUCTConfig(n_simulations=job.b_sims),
            mode="sample",
            temperature=1.0,
            temperature_moves=job.a_temperature_moves,
            add_root_noise=False,
            seed=job.b_seed,
            name=job.b_name,
        )
    elif job.opponent_kind == "mcts":
        assert job.b_sims is not None
        b = MCTSAgent(n_simulations=job.b_sims, seed=job.b_seed, name=job.b_name)
    elif job.opponent_kind == "random":
        b = RandomAgent(name=job.b_name, seed=job.b_seed)
    else:
        raise ValueError(f"unknown opponent_kind: {job.opponent_kind}")

    # Colour assignment: if a_is_player_0, a plays player 0.
    agent_0 = a if job.a_is_player_0 else b
    agent_1 = b if job.a_is_player_0 else a

    winner, moves = play_game(env, agent_0, agent_1, move_cap=job.move_cap)

    a_won = (winner == 0 and job.a_is_player_0) or (winner == 1 and not job.a_is_player_0)
    b_won = (winner == 1 and job.a_is_player_0) or (winner == 0 and not job.a_is_player_0)
    drew = winner is None

    return EvalGameResult(
        a_is_player_0=job.a_is_player_0,
        winner=winner,
        moves_played=moves,
        a_won=a_won,
        b_won=b_won,
        drew=drew,
    )


# -- parent-side: serialise net + orchestrate pool -------------------------


def serialise_net(net: AlphaZeroNet) -> tuple[bytes, dict]:
    """Pack a net's weights + config into picklable bytes + dict."""
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    net_bytes = buf.getvalue()
    cfg_dict = {
        "in_channels": net.config.in_channels,
        "board_h": net.config.board_h,
        "board_w": net.config.board_w,
        "n_actions": net.config.n_actions,
        "num_blocks": net.config.num_blocks,
        "channels": net.config.channels,
        "policy_channels": net.config.policy_channels,
        "value_channels": net.config.value_channels,
        "value_hidden": net.config.value_hidden,
    }
    return net_bytes, cfg_dict


def _make_pool(n_workers: int):
    """Create a spawn-context Pool with proper init. macOS-safe."""
    ctx = mp.get_context("spawn")
    return ctx.Pool(processes=n_workers, initializer=_worker_init)


def parallel_selfplay(
    jobs: list[SelfPlayJob],
    n_workers: int,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[SelfPlayResult]:
    """Run all self-play jobs in parallel workers; return results in order of completion.

    `progress_callback(completed, total)` is called after each worker returns.
    Raises if any worker raises (after all submitted jobs have been attempted).
    """
    if not jobs:
        return []
    if n_workers <= 1:
        return [_run_selfplay_job(j) for j in jobs]
    results: list[SelfPlayResult] = []
    with _make_pool(n_workers) as pool:
        total = len(jobs)
        for i, r in enumerate(pool.imap_unordered(_run_selfplay_job, jobs), 1):
            results.append(r)
            if progress_callback is not None:
                try:
                    progress_callback(i, total)
                except Exception:  # noqa: BLE001
                    pass
    return results


def parallel_eval(
    jobs: list[EvalJob],
    n_workers: int,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> list[EvalGameResult]:
    """Run all eval jobs in parallel. Like parallel_selfplay but for EvalJob."""
    if not jobs:
        return []
    if n_workers <= 1:
        out: list[EvalGameResult] = []
        for j in jobs:
            if should_stop is not None and should_stop():
                break
            out.append(_run_eval_job(j))
        return out
    results: list[EvalGameResult] = []
    with _make_pool(n_workers) as pool:
        for r in pool.imap_unordered(_run_eval_job, jobs):
            results.append(r)
            if should_stop is not None and should_stop():
                # Terminate remaining workers.
                pool.terminate()
                break
    return results
