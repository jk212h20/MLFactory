#!/usr/bin/env python
"""Mandala matchmaking eval: pit two agents head-to-head, alternate seats.

Designed for Phase 2 gate (distilled net vs heuristic / HP-PUCT) but
general enough to compare any two agents from a small registry.

Output: JSON-ish summary line on stdout + per-game records.

Always alternates which player goes first across seeds (seat-balanced) so
no agent gets a positional advantage.

Usage:
    uv run python scripts/mandala_eval.py \\
        --a net:deploy/checkpoints/mandala-pimc-distill.pt[:raw|:puct50|:puct200] \\
        --b heuristic \\
        --games 40 --workers 8

Agent specs:
    heuristic                      - Rule-based heuristic agent (greedy)
    hp_puct:N                      - Heuristic-prior PUCT with N sims
    pimc:DxS                       - PIMC, D determinizations x S sims each
    net:PATH:raw                   - Raw distilled-policy argmax (no search)
    net:PATH:puct:N                - AlphaZero net + PUCT with N sims
    random                         - Uniform random over legal actions
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from dataclasses import asdict, dataclass


# --- Agent registry -------------------------------------------------------


def _build_agent(spec: str, seed: int):
    """Spec parser. Constructs and returns the agent.

    Imports happen inside so workers don't pay torch-import cost
    until their first job.
    """
    parts = spec.split(":")
    kind = parts[0]

    if kind == "random":
        return _RandomAgent(seed=seed)

    if kind == "heuristic":
        from mlfactory.games.mandala.heuristic_agent import HeuristicMandalaAgent

        return HeuristicMandalaAgent(seed=seed)

    if kind == "hp_puct":
        n_sims = int(parts[1])
        from mlfactory.agents.alphazero.agent import AlphaZeroAgent
        from mlfactory.agents.alphazero.puct import PUCTConfig
        from mlfactory.games.mandala.env import MandalaEnv
        from mlfactory.games.mandala.heuristic_evaluator import HeuristicPriorEvaluator

        env = MandalaEnv(rng=random.Random(seed))
        ev = HeuristicPriorEvaluator(
            env, prior_temperature=1.0, rollout_policy="random", rng_seed=seed
        )
        return AlphaZeroAgent(
            ev,
            PUCTConfig(n_simulations=n_sims),
            mode="greedy",
            seed=seed,
            name=f"hp_puct{n_sims}",
        )

    if kind == "pimc":
        dets, sims = parts[1].split("x")
        from mlfactory.games.mandala.pimc_agent import PIMCMandalaAgent

        return PIMCMandalaAgent(
            n_determinizations=int(dets),
            sims_per_det=int(sims),
            mode="greedy",
            seed=seed,
        )

    if kind == "net":
        path = parts[1]
        sub = parts[2] if len(parts) > 2 else "raw"
        if sub == "raw":
            return _RawNetAgent(path=path, seed=seed)
        if sub == "puct":
            n_sims = int(parts[3])
            return _NetPUCTAgent(path=path, n_sims=n_sims, seed=seed)
        raise ValueError(f"unknown net subspec: {sub}")

    raise ValueError(f"unknown agent kind: {kind!r}")


class _RandomAgent:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.name = "random"

    def reset(self) -> None:
        pass

    def act(self, env, state):
        legal = env.legal_actions(state)
        return self._rng.choice(legal)


class _RawNetAgent:
    """Plays argmax of the trained net's policy head over legal actions.

    Pure no-search. Tests whether the distilled policy alone captures
    the teacher's wisdom.
    """

    def __init__(self, path: str, seed: int) -> None:
        import numpy as np
        import torch

        from mlfactory.agents.alphazero_mlp import AlphaZeroMLP
        from mlfactory.games.mandala.actions import legal_mask
        from mlfactory.games.mandala.encode import encode_view
        from mlfactory.games.mandala.rules import get_player_view

        self._np = np
        self._torch = torch
        self._encode_view = encode_view
        self._get_player_view = get_player_view
        self._legal_mask_fn = legal_mask

        net, _ = AlphaZeroMLP.load(path, map_location="cpu")
        self._net = net.eval()
        self._rng = random.Random(seed)
        self.name = f"net_raw[{path.rsplit('/', 1)[-1]}]"

    def reset(self) -> None:
        pass

    def act(self, env, state):
        mover = state.to_play
        view = self._get_player_view(state.core, mover)
        feats = self._encode_view(view, mover, state.history)
        mask = self._legal_mask_fn(state.core)
        x = self._torch.from_numpy(feats[None, :]).float()
        with self._torch.no_grad():
            logits, _ = self._net(x)
        logits = logits[0].cpu().numpy()
        # Mask illegal -> -inf
        masked = self._np.where(mask.astype(bool), logits, -self._np.inf)
        if not self._np.isfinite(masked).any():
            # No legal action got a finite logit; fall back to uniform.
            legal_idx = self._np.where(mask)[0]
            return int(self._rng.choice(legal_idx.tolist()))
        return int(masked.argmax())


class _NetPUCTAgent:
    """Distilled net + PUCT search on top. Tests net-as-evaluator."""

    def __init__(self, path: str, n_sims: int, seed: int) -> None:
        from mlfactory.agents.alphazero.agent import AlphaZeroAgent
        from mlfactory.agents.alphazero.evaluator import NetEvaluator
        from mlfactory.agents.alphazero.puct import PUCTConfig
        from mlfactory.agents.alphazero_mlp import AlphaZeroMLP
        from mlfactory.training.trainer_mandala import MandalaEncoderClosure

        net, _ = AlphaZeroMLP.load(path, map_location="cpu")
        evaluator = NetEvaluator(net, MandalaEncoderClosure(), device="cpu", name="net")
        self._inner = AlphaZeroAgent(
            evaluator,
            PUCTConfig(n_simulations=n_sims),
            mode="greedy",
            seed=seed,
            name=f"net_puct{n_sims}[{path.rsplit('/', 1)[-1]}]",
        )
        self.name = self._inner.name

    def reset(self) -> None:
        self._inner.reset()

    def act(self, env, state):
        return self._inner.act(env, state)


# --- Game runner ----------------------------------------------------------


@dataclass
class GameResult:
    seed: int
    a_seat: int  # 0 or 1
    winner: int | None  # 0/1 (game perspective) or None for draw
    n_moves: int
    a_won: int  # 1/0/-1 for draw -> count separately

    @property
    def a_outcome(self) -> str:
        if self.winner is None:
            return "draw"
        return "a_win" if self.winner == self.a_seat else "b_win"


def _play_one(args: tuple[str, str, int, int]) -> dict:
    """Play one game between agents specified by spec strings.

    a_seat = 0 means agent A is player 0 (first to move).
    """
    a_spec, b_spec, seed, a_seat = args

    from mlfactory.games.mandala.env import MandalaEnv

    env = MandalaEnv(rng=random.Random(seed))
    a = _build_agent(a_spec, seed=seed)
    b = _build_agent(b_spec, seed=seed + 50_000)
    a.reset()
    b.reset()

    if a_seat == 0:
        agents = (a, b)
    else:
        agents = (b, a)

    state = env.initial_state()
    n = 0
    while not state.is_terminal and n < 300:
        legal = env.legal_actions(state)
        if not legal:
            break
        action = agents[state.to_play].act(env, state)
        state = env.step(state, action)
        n += 1

    winner = state.winner if state.is_terminal else None
    if winner is None:
        a_won = 0  # draw
        outcome = "draw"
    elif winner == a_seat:
        a_won = 1
        outcome = "a_win"
    else:
        a_won = -1
        outcome = "b_win"

    return {
        "seed": seed,
        "a_seat": a_seat,
        "winner": winner,
        "n_moves": n,
        "a_won": a_won,
        "outcome": outcome,
    }


def _worker_init() -> None:
    import torch

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _binom_p_two_sided(wins: int, n: int) -> float:
    """Two-sided p-value vs H0: p=0.5. Exact binomial."""
    from math import comb

    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    pmf = [comb(n, i) * (0.5**n) for i in range(k + 1)]
    tail = sum(pmf)
    p = 2 * tail
    return min(p, 1.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Mandala two-agent eval")
    p.add_argument("--a", required=True, help="agent A spec (e.g. net:path:raw)")
    p.add_argument("--b", required=True, help="agent B spec (e.g. heuristic)")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json-out", default=None, help="optional JSON summary file")
    args = p.parse_args(argv)

    # Build job list with alternating seats.
    jobs: list[tuple[str, str, int, int]] = []
    for g in range(args.games):
        a_seat = g % 2
        jobs.append((args.a, args.b, args.seed + g, a_seat))

    t0 = time.monotonic()
    if args.workers <= 1:
        results = [_play_one(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_worker_init) as pool:
            results = []
            for r in pool.imap_unordered(_play_one, jobs):
                results.append(r)
                if not args.quiet and len(results) % max(1, args.games // 10) == 0:
                    a_wins = sum(1 for x in results if x["a_won"] == 1)
                    b_wins = sum(1 for x in results if x["a_won"] == -1)
                    draws = sum(1 for x in results if x["a_won"] == 0)
                    print(
                        f"  [{len(results)}/{args.games}] a={a_wins} b={b_wins} draws={draws}",
                        flush=True,
                    )

    wall = time.monotonic() - t0

    a_wins = sum(1 for r in results if r["a_won"] == 1)
    b_wins = sum(1 for r in results if r["a_won"] == -1)
    draws = sum(1 for r in results if r["a_won"] == 0)
    n_decisive = a_wins + b_wins
    a_winrate_overall = a_wins / max(1, len(results))
    a_winrate_decisive = a_wins / max(1, n_decisive)
    p_value = _binom_p_two_sided(a_wins, n_decisive) if n_decisive > 0 else 1.0

    # Per-seat breakdown (sanity check there's no first-move bias hiding things)
    a_wins_seat0 = sum(1 for r in results if r["a_seat"] == 0 and r["a_won"] == 1)
    a_wins_seat1 = sum(1 for r in results if r["a_seat"] == 1 and r["a_won"] == 1)
    n_seat0 = sum(1 for r in results if r["a_seat"] == 0)
    n_seat1 = sum(1 for r in results if r["a_seat"] == 1)

    summary = {
        "a": args.a,
        "b": args.b,
        "games": args.games,
        "wall_seconds": wall,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_winrate_overall": a_winrate_overall,
        "a_winrate_decisive": a_winrate_decisive,
        "p_value_vs_50_50": p_value,
        "by_seat": {
            "a_as_p0": {"wins": a_wins_seat0, "of": n_seat0},
            "a_as_p1": {"wins": a_wins_seat1, "of": n_seat1},
        },
    }
    print(json.dumps(summary, indent=2))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"wrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
