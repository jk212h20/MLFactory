"""Self-play: generate training samples by having the net play itself.

One self-play game produces a list of `Sample`s, one per position visited.
At each position we record:
- encoded planes
- PUCT visit distribution (the policy target)
- the final game result, seen from that position's mover's perspective

The final value target for a position is +1 if the mover at that position
won, -1 if they lost, 0 for a draw. This is the standard AlphaZero approach.

We also optionally record a `GameRecord` for disk serialisation (sample
games) while we're at it — it's cheap since we're already traversing states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from mlfactory.agents.alphazero.agent import AlphaZeroAgent
from mlfactory.core.env import Env
from mlfactory.training.replay_buffer import Sample
from mlfactory.training.sample_game import GameRecord, MoveRecord, state_to_dict


@dataclass
class SelfPlayResult:
    """Output of one self-play game."""

    samples: list[Sample]  # training data from the game
    record: GameRecord  # full serialisable record for replay
    winner: int | None  # 0, 1, or None (draw)
    n_moves: int


# Signature for a game-specific encoder: state -> (planes, legal_mask)
EncoderFn = Callable[[object], tuple[np.ndarray, np.ndarray]]


def play_selfplay_game(
    env: Env,
    agent: AlphaZeroAgent,
    encoder: EncoderFn,
    *,
    game_name: str,
    game_index: int = 0,
    iter_index: int | None = None,
    seed: int | None = None,
    max_moves: int = 500,
    record_visits: bool = True,
) -> SelfPlayResult:
    """Play one self-play game and return training samples + a GameRecord.

    The caller is responsible for configuring the agent's search (temperature,
    noise). Both "sides" are the same agent — self-play.
    """
    agent.reset()

    state = env.initial_state()
    # Per-position (planes, policy_target, mover) triples; value filled in
    # at the end of the game.
    per_move_planes: list[np.ndarray] = []
    per_move_policy: list[np.ndarray] = []
    per_move_mover: list[int] = []

    moves: list[MoveRecord] = []
    states: list[dict] = [state_to_dict(state)]

    ply = 0
    while not state.is_terminal and ply < max_moves:
        planes, _ = encoder(state)
        action = agent.act(env, state)
        search = agent.last_search

        # Training sample policy target = MCTS visit distribution.
        if search is not None:
            policy_target = search.policy_target.copy()
        else:
            # Agent didn't expose a search; fall back to one-hot on action.
            policy_target = np.zeros(env.num_actions, dtype=np.float32)
            policy_target[action] = 1.0

        per_move_planes.append(planes)
        per_move_policy.append(policy_target)
        per_move_mover.append(state.to_play)

        move_record = MoveRecord(
            ply=ply,
            to_play=state.to_play,
            action=int(action),
            visits=(
                {int(a): int(n) for a, n in search.root_visits.items()}
                if (record_visits and search is not None)
                else None
            ),
            q_values=(
                {int(a): float(q) for a, q in search.root_q.items()}
                if (record_visits and search is not None)
                else None
            ),
            root_value=(float(search.root_value) if search is not None else None),
        )
        moves.append(move_record)

        state = env.step(state, action)
        states.append(state_to_dict(state))
        ply += 1

    # Determine result from terminal state.
    winner = state.winner if state.is_terminal else None
    if winner is None:
        result = "draw"
    else:
        result = "a_win" if winner == 0 else "b_win"

    # Build training samples with value targets.
    samples: list[Sample] = []
    for planes, policy, mover in zip(per_move_planes, per_move_policy, per_move_mover):
        if winner is None:
            value_target = 0.0
        else:
            value_target = 1.0 if winner == mover else -1.0
        samples.append(Sample(planes=planes, policy_target=policy, value_target=value_target))

    record = GameRecord(
        game=game_name,
        iter=iter_index,
        kind="selfplay",
        agent_a=agent.name,
        agent_b=agent.name,
        seed=seed,
        result=result,
        winner=winner,
        moves=moves,
        states=states,
        notes={"n_simulations": agent.config.n_simulations, "game_index": game_index},
    )

    return SelfPlayResult(
        samples=samples,
        record=record,
        winner=winner,
        n_moves=ply,
    )
