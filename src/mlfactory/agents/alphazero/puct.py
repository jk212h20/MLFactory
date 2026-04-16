"""PUCT search (AlphaZero-style Monte Carlo Tree Search with neural priors).

Differences from vanilla UCT (`agents/mcts.py`):
- **Priors from evaluator**, not uniform. Expansion creates a child for every
  legal action up front; the prior P(a|s) weights the exploration term.
- **Value from evaluator, not rollouts.** No random playouts. At a new leaf
  we ask the evaluator for value V(s) and use that as the backup value.
- **PUCT formula**: U(s,a) = Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
- **Dirichlet noise** can be added at the root to encourage exploration
  during self-play.

Sign convention (identical to vanilla MCTS, see wiki/insights/2026-04-16-mcts-sign-bug.md):
    node.value_sum / node.visits = expected reward for the player who MOVED INTO this node.
The evaluator returns value from the mover's perspective at the leaf.
Mover-at-leaf = leaf.state.to_play. Mover-into-leaf = 1 - leaf.state.to_play.
So at the leaf, reward_for_mover_into_leaf = -evaluator_value.

We then backprop exactly as in vanilla MCTS: flip sign each ply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mlfactory.agents.alphazero.evaluator import EvalResult, Evaluator
from mlfactory.core.env import Action, Env, State


@dataclass(frozen=True)
class PUCTConfig:
    """Tunables for one PUCT search."""

    n_simulations: int = 200
    c_puct: float = 1.5
    # Dirichlet noise at root (AlphaZero paper: alpha ~= 10/avg_moves).
    dirichlet_alpha: float = 0.5
    dirichlet_epsilon: float = 0.0  # 0 disables noise; typically 0.25 in self-play
    # Numerical stability: tiny FPU value for visit-count ratios
    eps: float = 1e-8


@dataclass
class _PUCTNode:
    state: State
    to_play_at_entry: int  # player to move from this state
    parent: "_PUCTNode | None" = None
    action_from_parent: Action | None = None
    # Children by action index. Lazily materialised at expansion time.
    children: dict[Action, "_PUCTNode"] = field(default_factory=dict)
    # Prior P(a|s) for each legal action, populated at expansion.
    priors: dict[Action, float] = field(default_factory=dict)
    # MCTS statistics
    visits: int = 0
    value_sum: float = 0.0  # from perspective of mover-into-node
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value_for_mover_into: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


@dataclass(frozen=True)
class SearchResult:
    """Output of a PUCT search from the root."""

    root_visits: dict[Action, int]  # visit count per legal action
    root_q: dict[Action, float]  # mean value per legal action (mover-into-child perspective)
    root_value: float  # mean value at root (mover-into-root's perspective)
    # Policy target for training: normalized visit distribution (n_actions,)
    policy_target: np.ndarray
    # Full-length legal mask (n_actions,) bool
    legal_mask: np.ndarray
    total_sims: int


class PUCTSearch:
    """A configured, reusable PUCT searcher.

    Per-search state lives in the tree passed around internally; the object
    itself is stateless across calls to `search()` (aside from RNG).
    """

    def __init__(
        self,
        env: Env,
        evaluator: Evaluator,
        config: PUCTConfig | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.env = env
        self.evaluator = evaluator
        self.config = config or PUCTConfig()
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def search(self, state: State, *, add_root_noise: bool = False) -> SearchResult:
        """Run n_simulations from `state` and return aggregated statistics.

        Parameters
        ----------
        state : starting state; must not be terminal (caller checks).
        add_root_noise : mix Dirichlet noise into root priors (self-play).
        """
        if state.is_terminal:
            raise ValueError("cannot search from a terminal state")

        root = _PUCTNode(
            state=state,
            to_play_at_entry=state.to_play,
        )
        # Expand the root immediately so we have priors.
        root_eval = self.evaluator.evaluate(state)
        self._expand(root, root_eval)

        # Optionally inject Dirichlet noise at the root.
        if add_root_noise and self.config.dirichlet_epsilon > 0.0 and root.priors:
            self._add_dirichlet_noise(root)

        # Run simulations.
        for _ in range(self.config.n_simulations):
            self._simulate(root)

        # Aggregate visits into a policy target over the full action space.
        legal_mask = np.zeros(self.env.num_actions, dtype=bool)
        visits = np.zeros(self.env.num_actions, dtype=np.float32)
        root_visits_map: dict[Action, int] = {}
        root_q_map: dict[Action, float] = {}
        for action, child in root.children.items():
            visits[action] = child.visits
            legal_mask[action] = True
            root_visits_map[action] = child.visits
            root_q_map[action] = child.q

        total = visits.sum()
        if total > 0:
            policy_target = (visits / total).astype(np.float32)
        else:
            policy_target = np.zeros_like(visits)

        return SearchResult(
            root_visits=root_visits_map,
            root_q=root_q_map,
            root_value=root.q,
            policy_target=policy_target,
            legal_mask=legal_mask,
            total_sims=self.config.n_simulations,
        )

    # ------------------------------------------------------------------
    # One simulation
    # ------------------------------------------------------------------
    def _simulate(self, root: _PUCTNode) -> None:
        # 1. SELECT: descend until we hit an unexpanded or terminal leaf.
        node = root
        path: list[_PUCTNode] = [node]
        while node.is_expanded and not node.is_terminal:
            action, child = self._select_child(node)
            if child is None:
                # Expand a new child for this action.
                next_state = self.env.step(node.state, action)
                child = _PUCTNode(
                    state=next_state,
                    to_play_at_entry=next_state.to_play,
                    parent=node,
                    action_from_parent=action,
                )
                node.children[action] = child
            node = child
            path.append(node)

        # 2. EVALUATE / EXPAND leaf.
        leaf = node
        if leaf.is_terminal:
            # Reward for mover-into-leaf at a terminal state.
            leaf_value = leaf.terminal_value_for_mover_into
        elif leaf.state.is_terminal:
            leaf.is_terminal = True
            # Game just ended via a step from parent. Compute reward for
            # mover-into-leaf = the player who just made the move.
            mover_into = 1 - leaf.state.to_play
            if leaf.state.winner is None:
                leaf.terminal_value_for_mover_into = 0.0
            else:
                leaf.terminal_value_for_mover_into = (
                    1.0 if leaf.state.winner == mover_into else -1.0
                )
            leaf_value = leaf.terminal_value_for_mover_into
        else:
            # Non-terminal leaf: ask the evaluator.
            ev = self.evaluator.evaluate(leaf.state)
            self._expand(leaf, ev)
            # Evaluator's value is from mover-at-leaf's perspective.
            # Mover-into-leaf is the other player, so flip sign.
            leaf_value = -ev.value

        # 3. BACKPROP: walk path root -> leaf, flipping sign each ply from
        # the leaf's perspective. `leaf_value` is reward for mover-into-leaf.
        # The ply-parity flips ensure each ancestor gets its own mover-into value.
        value = leaf_value
        for n in reversed(path):
            n.visits += 1
            n.value_sum += value
            value = -value

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------
    def _expand(self, node: _PUCTNode, ev: EvalResult) -> None:
        """Populate `node.priors` and mark as expanded.

        Children are NOT created here; they are created lazily on first visit
        (see `_simulate`). This saves memory for unvisited branches and
        matches the common AlphaZero lazy-expansion pattern.
        """
        legal = self.env.legal_actions(node.state)
        if not legal:
            # Terminal per env (e.g., no legal moves).
            node.is_terminal = True
            if node.state.winner is None:
                node.terminal_value_for_mover_into = 0.0
            else:
                mover_into = 1 - node.state.to_play
                node.terminal_value_for_mover_into = (
                    1.0 if node.state.winner == mover_into else -1.0
                )
            return

        # Extract legal priors from the evaluator's output. The evaluator
        # guarantees priors are zero on illegal actions and sum to 1 over
        # legal, so we just read them off.
        priors = ev.priors
        for a in legal:
            node.priors[a] = float(priors[a])
        # Defensive renormalisation (evaluators may have rounding drift).
        s = sum(node.priors.values())
        if s > 0:
            for a in node.priors:
                node.priors[a] /= s
        else:
            # Degenerate: all priors zero on legal actions -> uniform.
            u = 1.0 / len(legal)
            for a in legal:
                node.priors[a] = u
        node.is_expanded = True

    def _add_dirichlet_noise(self, root: _PUCTNode) -> None:
        """Inject Dirichlet noise into root priors for exploration."""
        alpha = self.config.dirichlet_alpha
        eps = self.config.dirichlet_epsilon
        actions = list(root.priors.keys())
        noise = self.rng.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise):
            root.priors[a] = (1 - eps) * root.priors[a] + eps * float(n)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _select_child(self, node: _PUCTNode) -> tuple[Action, _PUCTNode | None]:
        """Select child maximising PUCT score. Returns (action, child-or-None).

        If the child hasn't been materialised yet (never visited), the second
        element is None and the caller creates it.
        """
        c_puct = self.config.c_puct
        total_visits = node.visits
        # sqrt(total_visits) is shared across all actions; AlphaZero uses
        # sqrt(parent.visits) which is >= 1 once the node has been visited.
        sqrt_parent = math.sqrt(max(total_visits, 1))

        best_action: Action | None = None
        best_child: _PUCTNode | None = None
        best_score = -math.inf
        for action, prior in node.priors.items():
            child = node.children.get(action)
            if child is None:
                q = 0.0
                visits = 0
            else:
                # IMPORTANT: from the parent's perspective, we want to pick
                # the action that maximizes reward for the parent's to-play.
                # `child.q` is the mean reward from mover-into-child's
                # perspective. Mover-into-child == parent's to-play. So
                # `child.q` is already in the right frame and we MAXIMIZE it.
                q = child.q
                visits = child.visits
            u = c_puct * prior * sqrt_parent / (1 + visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_action is not None, "root with no priors"
        return best_action, best_child
