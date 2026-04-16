"""AlphaZero agent: PUCT search + evaluator, playing under the Agent protocol.

Picks moves via PUCT. In competition/eval mode uses the greedy most-visited
action ("robust child"). In self-play mode (temperature > 0) samples from
visits^(1/T), which encourages early-game diversity.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from mlfactory.agents.alphazero.evaluator import Evaluator
from mlfactory.agents.alphazero.puct import PUCTConfig, PUCTSearch, SearchResult
from mlfactory.core.env import Action, Env, State


Mode = Literal["greedy", "sample"]


class AlphaZeroAgent:
    """Agent protocol implementation using PUCT + a pluggable evaluator.

    Parameters
    ----------
    evaluator : prior + value oracle (net-backed or uniform).
    config    : PUCT hyperparameters; defaults fine for Boop @ 100-200 sims.
    mode      : "greedy" (pick argmax visits, deterministic) or "sample"
                (sample from visits^(1/T), used in self-play).
    temperature : only used when mode=='sample'. 1.0 = proportional to visits;
                smaller values sharpen; 0 collapses to greedy.
    temperature_moves : for self-play, move index after which temperature is
                set to 0 (i.e., greedy). AlphaZero uses 30 on 19x19 Go; for
                Boop (typical 30-60 move games) 8 is a reasonable default.
    add_root_noise : add Dirichlet noise at the root (self-play).
    seed      : RNG seed for PUCT + sampling.
    name      : display name.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        config: PUCTConfig | None = None,
        *,
        mode: Mode = "greedy",
        temperature: float = 1.0,
        temperature_moves: int = 8,
        add_root_noise: bool = False,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or PUCTConfig()
        self.mode = mode
        self.temperature = temperature
        self.temperature_moves = temperature_moves
        self.add_root_noise = add_root_noise
        self._rng = np.random.default_rng(seed)
        self.name = name or f"az({getattr(evaluator, 'name', '?')}-{self.config.n_simulations})"
        # Per-game move counter, reset by `reset()`.
        self._move_idx = 0
        # Expose last search for introspection / sample-game logging.
        self.last_search: SearchResult | None = None

    def reset(self) -> None:
        self._move_idx = 0
        self.last_search = None

    def act(self, env: Env, state: State) -> Action:
        search = PUCTSearch(env, self.evaluator, self.config, rng=self._rng)
        result = search.search(state, add_root_noise=self.add_root_noise)
        self.last_search = result

        action = self._choose_action(result)
        self._move_idx += 1
        return action

    # ------------------------------------------------------------------
    def _choose_action(self, result: SearchResult) -> Action:
        visits = result.root_visits
        if not visits:
            raise ValueError("PUCT returned no visited actions")

        if self.mode == "greedy" or self._move_idx >= self.temperature_moves:
            # Greedy: most-visited; ties broken by higher Q.
            def _key(item: tuple[int, int]) -> tuple[int, float]:
                a, n = item
                return (n, result.root_q.get(a, 0.0))

            return max(visits.items(), key=_key)[0]

        # Sample: visits^(1/T)
        if self.temperature <= 1e-6:
            return max(visits.items(), key=lambda kv: kv[1])[0]
        actions = list(visits.keys())
        counts = np.array([visits[a] for a in actions], dtype=np.float64)
        if counts.sum() == 0:
            # defensive: fall back to uniform over actions
            idx = int(self._rng.integers(len(actions)))
            return actions[idx]
        if self.temperature == 1.0:
            probs = counts / counts.sum()
        else:
            probs = np.power(counts, 1.0 / self.temperature)
            probs /= probs.sum()
        idx = int(self._rng.choice(len(actions), p=probs))
        return actions[idx]
