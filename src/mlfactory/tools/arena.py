"""Arena: run matches between agents, report win rates, compute ELO.

Design rules:
- Every match between A and B plays **half the games with A as player 0 and half as B as player 0**.
  This is non-negotiable: a side-advantage bug is catastrophic and the arena must not hide it.
- Draws count as 0.5 for each side (ELO-standard).
- Confidence intervals use the Wilson score interval for proportions.
- ELO is computed from the full round-robin using iterative fitting (Bradley-Terry-ish).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from mlfactory.agents.base import Agent
from mlfactory.core.env import Env, State


@dataclass
class MatchResult:
    """Result of a single match (one game) between two agents."""

    agent_a_name: str
    agent_b_name: str
    a_is_player_0: bool  # colour assignment for this game
    winner: int | None  # 0 or 1 (player index), None for draw
    moves: int

    @property
    def a_won(self) -> bool:
        if self.winner is None:
            return False
        return (self.winner == 0) == self.a_is_player_0

    @property
    def b_won(self) -> bool:
        if self.winner is None:
            return False
        return not self.a_won

    @property
    def drawn(self) -> bool:
        return self.winner is None


@dataclass
class PairwiseResult:
    """Aggregated results for A vs B across many games."""

    agent_a_name: str
    agent_b_name: str
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.a_wins + self.b_wins + self.draws

    @property
    def a_score(self) -> float:
        """A's score: wins + 0.5*draws."""
        return self.a_wins + 0.5 * self.draws

    @property
    def a_win_rate(self) -> float:
        return self.a_score / max(self.total, 1)

    def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval (95% CI by default) for A's win rate.

        Treats the match as n Bernoulli trials with success = A's score normalised.
        Draws contribute 0.5, which slightly distorts the binomial assumption; for
        reporting purposes this is fine and standard practice.
        """
        n = self.total
        if n == 0:
            return (0.0, 1.0)
        p = self.a_win_rate
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))


def play_game(env: Env, agent_0: Agent, agent_1: Agent) -> tuple[int | None, int]:
    """Play one game. Returns (winner_player_index | None, moves_played)."""
    agent_0.reset()
    agent_1.reset()
    state: State = env.initial_state()
    agents = (agent_0, agent_1)
    moves = 0
    # Move cap to prevent infinite games from bugs (Connect 4 max = 42).
    move_cap = env.num_actions * 100
    while not state.is_terminal and moves < move_cap:
        a = agents[state.to_play].act(env, state)
        state = env.step(state, a)
        moves += 1
    if moves >= move_cap:
        return (None, moves)  # treat as draw
    return (state.winner, moves)


def play_match(
    env: Env,
    agent_a: Agent,
    agent_b: Agent,
    n_games: int = 100,
    swap_colours: bool = True,
    progress: bool = False,
) -> PairwiseResult:
    """Play n_games between A and B. Colour-balanced by default.

    If n_games is odd, the extra game goes to A-as-player-0.
    """
    result = PairwiseResult(agent_a_name=agent_a.name, agent_b_name=agent_b.name)
    games = []
    for i in range(n_games):
        a_is_player_0 = (i % 2 == 0) if swap_colours else True
        if a_is_player_0:
            winner, moves = play_game(env, agent_a, agent_b)
        else:
            winner, moves = play_game(env, agent_b, agent_a)
        games.append(
            MatchResult(
                agent_a_name=agent_a.name,
                agent_b_name=agent_b.name,
                a_is_player_0=a_is_player_0,
                winner=winner,
                moves=moves,
            )
        )
        if games[-1].a_won:
            result.a_wins += 1
        elif games[-1].b_won:
            result.b_wins += 1
        else:
            result.draws += 1
        if progress and (i + 1) % max(1, n_games // 10) == 0:
            print(
                f"  [{agent_a.name} vs {agent_b.name}] game {i + 1}/{n_games} "
                f"A:{result.a_wins} B:{result.b_wins} D:{result.draws}"
            )
    return result


# --- ELO computation ---------------------------------------------------


def compute_elo(
    pairwise: list[PairwiseResult],
    base: float = 1500.0,
    anchor_name: str | None = None,
    n_iters: int = 1000,
    lr: float = 8.0,
) -> dict[str, float]:
    """Fit ELO ratings from a set of pairwise match results.

    Uses gradient descent on the Bradley-Terry log-likelihood. Simple and works well
    for small agent pools.

    Parameters
    ----------
    pairwise : list of PairwiseResult
    base : float
        The rating everyone starts at. Also what `anchor_name` (if provided) gets fixed to.
    anchor_name : str | None
        If provided, this agent's rating is pinned to `base` throughout.
    n_iters : int
        Number of optimisation steps.
    lr : float
        Learning rate.
    """
    names = set()
    for p in pairwise:
        names.add(p.agent_a_name)
        names.add(p.agent_b_name)
    ratings = dict.fromkeys(names, base)

    for _ in range(n_iters):
        grads = dict.fromkeys(names, 0.0)
        for p in pairwise:
            if p.total == 0:
                continue
            ra = ratings[p.agent_a_name]
            rb = ratings[p.agent_b_name]
            # Expected score for A under Elo logistic:
            expected = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            actual = p.a_score / p.total
            # Gradient of log-likelihood w.r.t. ra is proportional to (actual - expected)
            delta = (actual - expected) * p.total
            grads[p.agent_a_name] += delta
            grads[p.agent_b_name] -= delta
        for name in names:
            if anchor_name is not None and name == anchor_name:
                continue
            ratings[name] += lr * grads[name] / max(len(pairwise), 1)

    if anchor_name:
        # Shift so the anchor ends exactly at `base` (in case gradient steps drifted).
        shift = base - ratings[anchor_name]
        ratings = {n: r + shift for n, r in ratings.items()}

    return ratings


# --- Tournament --------------------------------------------------------


@dataclass
class TournamentResult:
    """Result of a full round-robin tournament."""

    agents: list[str]
    pairwise: list[PairwiseResult] = field(default_factory=list)
    elo: dict[str, float] = field(default_factory=dict)
    games_per_match: int = 0
    wall_seconds: float = 0.0

    def matrix(self) -> list[list[str]]:
        """Win-rate matrix as strings for pretty printing. agents[i] vs agents[j]."""
        by_pair: dict[tuple[str, str], PairwiseResult] = {}
        for p in self.pairwise:
            by_pair[(p.agent_a_name, p.agent_b_name)] = p
        rows = []
        for a in self.agents:
            row = []
            for b in self.agents:
                if a == b:
                    row.append("  -  ")
                    continue
                if (a, b) in by_pair:
                    p = by_pair[(a, b)]
                    row.append(f"{p.a_win_rate:.2%}")
                elif (b, a) in by_pair:
                    p = by_pair[(b, a)]
                    row.append(f"{1 - p.a_win_rate:.2%}")
                else:
                    row.append("  ?  ")
            rows.append(row)
        return rows


def round_robin(
    env: Env,
    agents: list[Agent],
    games_per_match: int = 100,
    progress: bool = False,
) -> TournamentResult:
    """Full round-robin. Every pair of agents plays `games_per_match` games (colour-balanced)."""
    start = time.monotonic()
    pairwise: list[PairwiseResult] = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if progress:
                print(f"\n[match] {agents[i].name} vs {agents[j].name}")
            result = play_match(
                env, agents[i], agents[j], n_games=games_per_match, progress=progress
            )
            pairwise.append(result)
    elapsed = time.monotonic() - start
    elo = compute_elo(pairwise, anchor_name=agents[0].name)
    return TournamentResult(
        agents=[a.name for a in agents],
        pairwise=pairwise,
        elo=elo,
        games_per_match=games_per_match,
        wall_seconds=elapsed,
    )
