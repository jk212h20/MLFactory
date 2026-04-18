"""Game probes for the classifier — wrap each game's engine in the
GameProbe interface so the classifier can analyze it uniformly.

Add a new probe per game; the classifier itself stays game-agnostic.
"""

from __future__ import annotations

import json
import random
from typing import Any


# --- Boop probe ------------------------------------------------------------


class BoopProbe:
    """Probe for Boop (perfect info, deterministic, 2D-board game)."""

    name = "boop"
    heuristic_agent = None  # we don't have a stand-alone heuristic for Boop

    def __init__(self) -> None:
        from mlfactory.games.boop import Boop

        self.env = Boop()
        self._n_actions = self.env.num_actions

    def create_initial(self, seed: int) -> Any:
        # Boop is deterministic; seed unused by initial state.
        return self.env.initial_state()

    def legal_actions(self, state: Any) -> list:
        return self.env.legal_actions(state)

    def step(self, state: Any, action: Any, seed: int) -> Any:
        return self.env.step(state, action)

    def is_terminal(self, state: Any) -> bool:
        return state.is_terminal

    def winner(self, state: Any) -> int | None:
        return state.winner

    def to_play(self, state: Any) -> int:
        return state.to_play

    def player_view(self, state: Any, player_index: int) -> Any:
        return state  # perfect info

    def state_size_bytes(self, state: Any) -> int:
        # Rough estimate: serialise to a JSON-friendly tuple.
        try:
            return len(
                json.dumps(
                    {
                        "board": list(state.board),
                        "orange_pool": list(state.orange_pool),
                        "gray_pool": list(state.gray_pool),
                        "to_play": state.to_play,
                        "phase": state.phase,
                    }
                )
            )
        except Exception:
            return len(repr(state))

    def num_actions_total(self) -> int:
        return self._n_actions


# --- Mandala probe ---------------------------------------------------------


class MandalaProbe:
    """Probe for Mandala (imperfect info, stochastic deck, card game)."""

    name = "mandala"

    def __init__(self) -> None:
        from mlfactory.games.mandala.actions import N_TEMPLATES
        from mlfactory.games.mandala.heuristic_agent import HeuristicMandalaAgent
        from mlfactory.games.mandala.env import MandalaEnv

        self.env_cls = MandalaEnv
        self.n_actions = N_TEMPLATES
        self._heuristic_agent_cls = HeuristicMandalaAgent
        # Bind a heuristic agent for the classifier's heuristic-vs-random
        # measurement.
        self.heuristic_agent = self._make_heuristic_agent

    def _make_heuristic_agent(self, state: Any) -> Any:
        agent = self._heuristic_agent_cls(seed=0)
        env = self.env_cls()
        return agent.act(env, state)

    def create_initial(self, seed: int) -> Any:
        env = self.env_cls(rng=random.Random(seed))
        return env.initial_state()

    def legal_actions(self, state: Any) -> list:
        env = self.env_cls()
        return env.legal_actions(state)

    def step(self, state: Any, action: Any, seed: int) -> Any:
        env = self.env_cls(rng=random.Random(seed))
        return env.step(state, action)

    def is_terminal(self, state: Any) -> bool:
        return state.is_terminal

    def winner(self, state: Any) -> int | None:
        return state.winner

    def to_play(self, state: Any) -> int:
        return state.to_play

    def player_view(self, state: Any, player_index: int) -> Any:
        from mlfactory.games.mandala.rules import get_player_view
        from mlfactory.games.mandala.env import MandalaState

        view_core = get_player_view(state.core, player_index)
        return MandalaState(core=view_core, history=list(state.history))

    def state_size_bytes(self, state: Any) -> int:
        try:
            return len(json.dumps(state.core, default=str))
        except Exception:
            return len(repr(state.core))

    def num_actions_total(self) -> int:
        return self.n_actions
