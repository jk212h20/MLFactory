"""Vanilla UCT Monte Carlo Tree Search.

Pure Python; no neural net. Uses uniform-random playouts (light playouts).

Implementation details:
- Tree nodes track visit count `N`, total value `W`, parent, action-to-reach.
- **Values are from the perspective of the player who MOVED INTO this node.**
  i.e., node.W / node.N is the expected value for the player who just moved here.
  This is the cleanest convention and makes UCT selection a simple maximisation.
- At expansion time we add one child per simulation (standard).
- Rollout: uniform-random legal moves until terminal. Then compute reward from
  the terminal state's perspective and walk back up flipping signs.

See also:
- wiki/techniques/mcts-uct.md — the spec this implements
- wiki/sources/browne2012-mcts-survey.md — theory
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from mlfactory.core.env import Action, Env, State


@dataclass
class _Node:
    parent: _Node | None
    action: Action | None  # action that led from parent to this node
    state: State
    to_play_at_entry: int  # player who was to move at this node (== state.to_play)
    visits: int = 0
    value_sum: float = 0.0  # from perspective of the player who MOVED INTO this node
    # Children are created on expansion.
    children: list[_Node] = field(default_factory=list)
    # Unexpanded legal actions at this node.
    untried: list[Action] = field(default_factory=list)
    terminal: bool = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    @property
    def is_fully_expanded(self) -> bool:
        return not self.untried and bool(self.children)


def _uct_score(child: _Node, parent_visits: int, c: float) -> float:
    """UCT: exploitation + exploration. Higher = more attractive."""
    if child.visits == 0:
        return math.inf
    exploit = child.value_sum / child.visits
    explore = c * math.sqrt(math.log(parent_visits) / child.visits)
    return exploit + explore


class MCTSAgent:
    """Vanilla UCT MCTS with uniform-random rollouts.

    Parameters
    ----------
    n_simulations : int
        Number of playouts per move.
    c : float
        Exploration constant. sqrt(2) ~= 1.4142 is the theoretical value
        for rewards in [0, 1]. Our rewards are in [-1, +1]; we keep the
        same constant empirically (rescale in future if needed).
    seed : int | None
        RNG seed.
    name : str | None
        Display name; defaults to f"mcts{n_simulations}".
    """

    def __init__(
        self,
        n_simulations: int = 200,
        c: float = math.sqrt(2),
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        self.n_simulations = n_simulations
        self.c = c
        self.rng = random.Random(seed)
        self.name = name or f"mcts{n_simulations}"

    def reset(self) -> None:
        # Vanilla MCTS builds a fresh tree per move. Nothing to carry.
        pass

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def act(self, env: Env, state: State) -> Action:
        legal = env.legal_actions(state)
        if not legal:
            raise ValueError("no legal actions")
        if len(legal) == 1:
            return legal[0]

        root = _Node(
            parent=None,
            action=None,
            state=state,
            to_play_at_entry=state.to_play,
        )
        root.untried = list(legal)

        for _ in range(self.n_simulations):
            self._simulate(env, root)

        # Robust child: pick the most-visited action.
        # (More stable than mean value at modest budgets.)
        best = max(root.children, key=lambda n: (n.visits, n.mean_value))
        return best.action  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # One MCTS iteration
    # ------------------------------------------------------------------
    def _simulate(self, env: Env, root: _Node) -> None:
        # 1. SELECT: walk down using UCT until we find a node that's
        #    not fully expanded OR terminal.
        node = root
        while not node.terminal and node.is_fully_expanded:
            node = max(
                node.children,
                key=lambda ch: _uct_score(ch, node.visits, self.c),
            )

        # 2. EXPAND: if not terminal, add one child.
        if not node.terminal:
            if not node.untried and not node.children:
                # First visit to this node; populate untried.
                legal = env.legal_actions(node.state)
                if not legal:
                    # Reached a state the env considers terminal.
                    node.terminal = True
                else:
                    node.untried = list(legal)

            if node.untried:
                action = node.untried.pop(self.rng.randrange(len(node.untried)))
                next_state = env.step(node.state, action)
                child = _Node(
                    parent=node,
                    action=action,
                    state=next_state,
                    to_play_at_entry=next_state.to_play,
                )
                if next_state.is_terminal:
                    child.terminal = True
                else:
                    child.untried = list(env.legal_actions(next_state))
                node.children.append(child)
                node = child

        # 3. SIMULATE (rollout from `node`).
        reward_for_mover = self._rollout(env, node)

        # 4. BACKPROP. `reward_for_mover` is from the perspective of the
        #    player who just moved INTO `node`. Walk back up; every ancestor's
        #    perspective flips each ply.
        cur: _Node | None = node
        value = reward_for_mover
        while cur is not None:
            cur.visits += 1
            cur.value_sum += value
            value = -value  # flip for the parent (the other player)
            cur = cur.parent

    # ------------------------------------------------------------------
    # Random rollout
    # ------------------------------------------------------------------
    def _rollout(self, env: Env, node: _Node) -> float:
        """Play uniform-random moves from node.state to terminal.

        Returns the reward for the player who just moved INTO `node`.

        - If `node` is already terminal: reward is -env.terminal_value(state),
          because terminal_value is from state.to_play's perspective (the player
          who would move next), and the mover is the *other* player.
        - Otherwise: random playout, then same sign adjustment.
        """
        state = node.state
        if state.is_terminal:
            # terminal_value is from perspective of state.to_play (the side that would move next)
            # The "mover into this node" was 1 - state.to_play, i.e. opposite perspective.
            return -env.terminal_value(state)

        # Random rollout
        while not state.is_terminal:
            legal = env.legal_actions(state)
            action = legal[self.rng.randrange(len(legal))]
            state = env.step(state, action)

        # Terminal reached: flip sign to put it in the mover-into-node's perspective.
        # After the rollout, `state.to_play` is the side that would go if game continued.
        # The mover into `node` had to_play == node.to_play_at_entry.
        # Reward for mover = +1 if node.to_play_at_entry won, -1 if lost, 0 if draw.
        if state.winner is None:
            return 0.0
        return 1.0 if state.winner == node.to_play_at_entry else -1.0
