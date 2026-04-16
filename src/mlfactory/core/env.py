"""The Env protocol: a minimal interface every game in MLFactory implements.

Design principles:
- **Immutable states**. `step()` returns a new state; it never mutates. This is non-negotiable
  because MCTS explores many branches from the same node.
- **Side-to-move in the state**. `to_play` is always read from the state; agents never track it.
- **Legal action mask is authoritative**. Agents must call `legal_actions()` and never guess.
- **Value is from the perspective of the side-to-move at terminal**. If `to_play` loses, value
  is -1; if wins, +1; draw is 0. Callers flip signs as they walk game histories.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Action IDs are plain integers. Game adapters are responsible for
# encoding/decoding them onto their native action representation.
Action = int

# Player IDs: 0 or 1. The state's `to_play` tells us whose turn it is.
Player = int


@runtime_checkable
class State(Protocol):
    """A game state. Must be hashable and cheaply clonable.

    Implementations are typically frozen dataclasses or tuples of arrays.
    """

    @property
    def to_play(self) -> Player:
        """Which player (0 or 1) is to move next."""
        ...

    @property
    def is_terminal(self) -> bool:
        """Whether the game is over."""
        ...

    @property
    def winner(self) -> Player | None:
        """Winner if the game is over and someone won; None for ongoing or draw."""
        ...


@runtime_checkable
class Env(Protocol):
    """A two-player, perfect-information, deterministic game environment.

    All methods are pure: they take a state (and optionally an action) and return
    a new state or derived value. The env itself holds no mutable per-game state.
    """

    name: str
    num_actions: int

    def initial_state(self) -> State:
        """The starting state of a new game."""
        ...

    def legal_actions(self, state: State) -> list[Action]:
        """The list of legal action IDs from `state`. Empty if terminal."""
        ...

    def step(self, state: State, action: Action) -> State:
        """Return the successor state after `action` is taken from `state`.

        Must raise `ValueError` if the action is illegal in `state`.
        Must NOT mutate `state`.
        """
        ...

    def terminal_value(self, state: State) -> float:
        """At a terminal state, the value from perspective of side-to-move.

        By convention:
        - +1.0 if the side-to-move has just won (unusual — typically the side that JUST moved won)
        - -1.0 if the side-to-move has lost (common case: opponent's winning move ended game)
        - 0.0 for draws.

        The typical implementation: at terminal, `to_play` is the player who would move *next*
        if the game continued. If their opponent just made the winning move, `to_play` sees -1.

        Must raise `ValueError` if called on a non-terminal state.
        """
        ...

    def render(self, state: State) -> str:
        """Human-readable string representation."""
        ...
