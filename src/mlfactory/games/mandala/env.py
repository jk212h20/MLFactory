"""Env adapter: expose the Mandala engine under the MLFactory Env protocol.

The engine uses dict-based state (matching the JS source). The Env protocol
expects immutable state objects with `to_play`, `is_terminal`, `winner`
properties. We wrap the engine's dict state in a lightweight MandalaState
dataclass that mirrors those properties and keeps a reference to the
underlying dict, while also carrying an action history list for encoding.

Key design decisions:
- `Action` is the template index (int in [0, N_TEMPLATES)), matching the
  generic MLFactory action protocol.
- legal_actions(state) returns the legal template indices.
- step(state, action) materialises the template into a concrete engine
  action using the actual current hand, applies perform_action, and
  appends the template to the history.
- terminal_value uses the REAL game outcome (win/loss/draw), NOT the
  score. That's the primary value signal. The expected_final_score
  feature in the encoder gives the net a supplementary shaping signal.
- **Seed / RNG handling**: Mandala has stochastic transitions (reshuffles
  happen when the deck empties). The engine uses random.Random by default.
  For reproducible self-play we thread an rng through the env. The trainer
  passes one per game.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from mlfactory.core.env import Action, Env, Player
from mlfactory.games.mandala.actions import (
    N_TEMPLATES,
    legal_template_indices,
    template_to_engine_action,
)
from mlfactory.games.mandala.rules import (
    calculate_score,
    create_game,
    get_player_view,
    get_winner,
    perform_action,
)


@dataclass
class MandalaState:
    """Immutable-ish wrapper around the engine's dict state.

    `core` holds the full state (ground truth). `history` is a list of
    {template_index, actor_index} dicts for the encoder. Neither field
    should be mutated after construction.

    For thread / MCTS safety, callers clone before mutating. `MandalaEnv.step`
    already deep-copies, so you get a fresh core on every step.
    """

    core: dict
    history: list[dict] = field(default_factory=list)

    @property
    def to_play(self) -> Player:
        return int(self.core["currentPlayerIndex"])

    @property
    def is_terminal(self) -> bool:
        return self.core["phase"] == "ended"

    @property
    def winner(self) -> Player | None:
        if not self.is_terminal:
            return None
        w = get_winner(self.core)
        if w is None:
            return None
        # engine winnerId is "p0" or "p1"; convert to int index
        wid = w["winnerId"]
        # Handle ties: engine always returns a winner (tie-broken by fewer cup
        # cards). We respect that — no None for ties.
        return 0 if wid == "p0" else 1


class MandalaEnv:
    """MLFactory Env protocol implementation for Mandala.

    `name`, `num_actions`, `initial_state`, `legal_actions`, `step`,
    `terminal_value`, `render` all implemented. See env.py in core for
    the protocol.

    Because Mandala has stochastic transitions, callers must pass an
    RNG to `step` if they want reproducibility. A convenience overload
    lets you bind an rng at construction time.
    """

    name: str = "mandala"
    num_actions: int = N_TEMPLATES

    def __init__(self, rng: random.Random | None = None) -> None:
        """Create the env.

        rng : optional. If provided, used for initial_state() and for all
              step() calls that don't pass their own rng. If None, a
              fresh unseeded Random() is used per call — non-reproducible.
        """
        self._rng = rng

    def _get_rng(self, override: random.Random | None) -> random.Random:
        if override is not None:
            return override
        if self._rng is not None:
            return self._rng
        return random.Random()

    def initial_state(self, rng: random.Random | None = None) -> MandalaState:
        """Return the starting state."""
        r = self._get_rng(rng)
        core = create_game("p0", "p1", rng=r)
        return MandalaState(core=core, history=[])

    def legal_actions(self, state: MandalaState) -> list[Action]:
        return legal_template_indices(state.core)

    def step(
        self,
        state: MandalaState,
        action: Action,
        rng: random.Random | None = None,
    ) -> MandalaState:
        """Apply action template index to state, return new state.

        The action is materialised into a concrete engine action using the
        state's current hand, applied via perform_action, and appended to
        the history.
        """
        if state.is_terminal:
            raise ValueError("cannot step a terminal state")

        from mlfactory.games.mandala.actions import index_to_template

        template = index_to_template(action)
        engine_action = template_to_engine_action(template, state.core)

        r = self._get_rng(rng)
        result = perform_action(state.core, engine_action, rng=r)
        if not result["success"]:
            raise ValueError(
                f"illegal action: template_index={action}, engine_action="
                f"{engine_action}, error={result.get('error')}"
            )

        # History: record this action tagged with the player who took it.
        # IMPORTANT: `state.to_play` was the actor (before the step advanced).
        new_history = list(state.history)
        new_history.append({"template_index": int(action), "actor_index": state.to_play})

        return MandalaState(core=result["newState"], history=new_history)

    def terminal_value(self, state: MandalaState) -> float:
        """Value from perspective of the side-to-move at this state.

        At terminal: side-to-move is whoever would move next if the game
        continued. Winner is state.winner. So:
        - winner == side-to-move: +1.0
        - winner is opponent:     -1.0
        - draw (engine doesn't produce real draws, but be defensive): 0.0
        """
        if not state.is_terminal:
            raise ValueError("terminal_value called on non-terminal state")
        w = state.winner
        if w is None:
            return 0.0
        return 1.0 if w == state.to_play else -1.0

    def render(self, state: MandalaState) -> str:
        """Compact human-readable summary."""
        s = state.core
        lines = [
            f"phase={s['phase']}  turn={s['currentPlayerIndex']}  #{s['turnNumber']}  "
            f"deck={len(s['deck'])} discard={len(s['discardPile'])}  "
            f"trigger={s.get('endGameTrigger')}",
        ]
        for p_idx, p in enumerate(s["players"]):
            hand = ",".join(
                c["color"][0].upper() if c["color"] != "hidden" else "?" for c in p["hand"]
            )
            cup = ",".join(
                c["color"][0].upper() if c["color"] != "hidden" else "?" for c in p["cup"]
            )
            river = "".join(c[0].upper() if c else "." for c in p["river"])
            sc = calculate_score(p) if "hidden" not in {c.get("color") for c in p["cup"]} else "?"
            lines.append(f"  p{p_idx}: hand=[{hand}] cup=[{cup}] river=[{river}] score={sc}")
        for m_idx, m in enumerate(s["mandalas"]):
            mnt = ",".join(c["color"][0].upper() for c in m["mountain"])
            f0 = ",".join(c["color"][0].upper() for c in m["fields"][0])
            f1 = ",".join(c["color"][0].upper() for c in m["fields"][1])
            lines.append(f"  m{m_idx}: mountain=[{mnt}] f0=[{f0}] f1=[{f1}]")
        if s.get("destruction"):
            d = s["destruction"]
            lines.append(
                f"  destruction: mandala={d['mandalaIndex']} claimer={d['currentClaimerIndex']} "
                f"remaining={d['remainingColors']}"
            )
        return "\n".join(lines)
