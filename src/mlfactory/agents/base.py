"""The Agent protocol: anything that can pick an action given a state."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mlfactory.core.env import Action, Env, State


@runtime_checkable
class Agent(Protocol):
    """An agent produces an action given an env and state.

    Agents are expected to be long-lived: they may carry hyperparameters,
    trained weights, search trees, RNG state, etc. But `act()` itself
    must not mutate the input state.
    """

    name: str

    def act(self, env: Env, state: State) -> Action:
        """Choose and return a legal action in `state`."""
        ...

    def reset(self) -> None:
        """Called at the start of every new game. Optional; default is a no-op."""
        ...
