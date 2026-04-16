"""Uniform-random agent. The baseline everything must beat."""

from __future__ import annotations

import random

from mlfactory.core.env import Action, Env, State


class RandomAgent:
    """Picks a uniformly random legal action."""

    def __init__(self, name: str = "random", seed: int | None = None) -> None:
        self.name = name
        self.rng = random.Random(seed)

    def act(self, env: Env, state: State) -> Action:
        legal = env.legal_actions(state)
        if not legal:
            raise ValueError("no legal actions in this state")
        return self.rng.choice(legal)

    def reset(self) -> None:
        pass
