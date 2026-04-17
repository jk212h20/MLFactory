"""Rule-based heuristic Mandala bot.

Not strong, but strictly better than random. Intended uses:
- Fixed external baseline for training evals ("does the trained net beat
  the heuristic bot?").
- Bootstrap opponent: generate supervised training data so the net doesn't
  have to cold-start from uniform randomness (which costs a lot of
  iterations on a game with 150-template action space and ~100-move games).

Strategy (aligned with the rules analysis + the user's observations):

  1. Don't claim colors matching your starting cup cards early if you can
     avoid it — starting cup cards have latent value ~3, so putting them
     in low river slots destroys value. Prefer claims of colors your
     known cup doesn't already contain heavily.
  2. During destruction, rank claim options by (a) whether they add a new
     color to your river (first-of-color goes to river = long-term
     scoring slot), (b) how many cards the claim pulls from the mountain
     (more cards = more future cup points if river matches), (c) avoid
     colors you already have 2+ of in cup (diminishing returns).
  3. In normal play, prefer grow_field with multiples (bigger field =
     better claim priority when the mandala completes). Among
     grow_field options, pick the largest legal count that doesn't
     reduce your hand below 2 (we leave options for next turn).
  4. Prefer build_mountain for singleton colors (drops one card, draws 3,
     hand-health).
  5. Only discard_redraw if nothing else useful; prefer discarding
     colors that are already blocked in both mandalas.

This is shallow but gets us a real floor. AZ should beat it after even
modest training.
"""

from __future__ import annotations

import random
from collections import Counter

from mlfactory.games.mandala.actions import (
    Template,
    legal_template_indices,
    index_to_template,
    template_to_index,
)
from mlfactory.games.mandala.env import MandalaEnv, MandalaState
from mlfactory.games.mandala.rules import (
    COLORS,
    can_play_color_to_field,
    can_play_color_to_mountain,
)


class HeuristicMandalaAgent:
    """Rule-based agent. Implements the Agent protocol minimally:
    name + act(env, state) -> action (int template index).

    reset() is a no-op; the heuristic is stateless beyond rng.
    """

    def __init__(self, seed: int | None = None, name: str = "heuristic") -> None:
        self._rng = random.Random(seed)
        self.name = name
        self.last_search = None  # for code that expects it; always None here

    def reset(self) -> None:
        pass

    def act(self, env: MandalaEnv, state: MandalaState) -> int:
        legal = legal_template_indices(state.core)
        if not legal:
            raise ValueError("no legal actions")
        scored = [(self._score_template(state, index_to_template(t)), t) for t in legal]
        # Stable-ish tie-breaking: highest score wins, ties broken by rng.
        best_score = max(s for s, _ in scored)
        top = [t for s, t in scored if s == best_score]
        return self._rng.choice(top)

    # ------------------------------------------------------------------
    # Scoring heuristics
    # ------------------------------------------------------------------

    def _score_template(self, state: MandalaState, t: Template) -> float:
        """Bigger score = more attractive."""
        if t.kind == "claim_color":
            return self._score_claim(state, t.color)
        if t.kind == "build_mountain":
            return self._score_build_mountain(state, t)
        if t.kind == "grow_field":
            return self._score_grow_field(state, t)
        if t.kind == "discard_redraw":
            return self._score_discard_redraw(state, t)
        return 0.0

    # --- claim scoring ---

    def _score_claim(self, state: MandalaState, color: str) -> float:
        """Prefer claims that add NEW colors to river and pull many cards."""
        me = state.core["players"][state.to_play]
        river = me["river"]
        cup_colors = Counter(c["color"] for c in me["cup"])

        dest = state.core.get("destruction")
        if dest is None:
            return 0.0
        mandala = state.core["mandalas"][dest["mandalaIndex"]]
        # How many cards of this color in the mountain?
        n_cards = sum(1 for c in mandala["mountain"] if c["color"] == color)

        base = 0.0
        if color not in river:
            # First-of-color goes to next empty river slot. The later
            # the empty slot, the more valuable existing cup cards of
            # this color will be.
            try:
                first_empty = river.index(None)
            except ValueError:
                first_empty = -1
            if first_empty == -1:
                base += 0.5  # river full — all to cup, low value
            else:
                # river slot value is first_empty + 1, but each extra cup
                # card of this color is worth that slot value. And we
                # already have cup_colors[color] cup cards of this color.
                slot_value = first_empty + 1
                base += slot_value * (n_cards - 1 + cup_colors.get(color, 0))
                # Penalize filling river slot 1 or 2 unless we're ONLY
                # going to get it (starting cup bias — don't claim early
                # what might later fill slots 4-5).
                if first_empty <= 1:
                    base -= 1.0
        else:
            # Color already in river; all n_cards go to cup.
            slot = river.index(color)
            base += (slot + 1) * n_cards
        # Slight tiebreak preferring more cards.
        return base + 0.01 * n_cards

    # --- build_mountain scoring ---

    def _score_build_mountain(self, state: MandalaState, t: Template) -> float:
        """Prefer build when we have singletons or blocked colors."""
        me = state.core["players"][state.to_play]
        color_counts = Counter(c["color"] for c in me["hand"])
        count_of_color = color_counts[t.color]

        base = 2.0  # build is generally healthy (draws 3)
        # Strongly prefer using singleton colors for build (fewer are wasted).
        if count_of_color == 1:
            base += 3.0
        elif count_of_color >= 3:
            base -= 2.0  # wasteful; these are better for grow_field
        # Favor building toward a mandala that's missing this color more —
        # less important; small bonus.
        return base

    # --- grow_field scoring ---

    def _score_grow_field(self, state: MandalaState, t: Template) -> float:
        """Prefer grow with bigger counts (more field dominance)."""
        me = state.core["players"][state.to_play]
        color_counts = Counter(c["color"] for c in me["hand"])
        mandala = state.core["mandalas"][t.mandala_index]

        # Base: grow scales with count (more field = better claim order).
        base = 1.0 + 2.0 * t.count
        # Extra bonus if this is close to "using most of a color we have".
        if t.count == color_counts[t.color]:
            base += 1.0
        # Penalize leaving your hand very small.
        remaining = len(me["hand"]) - t.count
        if remaining <= 1:
            base -= 1.5
        # Penalize growing into a field of a mandala where opponent's
        # field is much bigger (we won't win the claim race).
        opp_idx = 1 - state.to_play
        my_field = len(mandala["fields"][state.to_play])
        opp_field = len(mandala["fields"][opp_idx])
        if opp_field > my_field + t.count + 1:
            base -= 2.0
        return base

    # --- discard_redraw scoring ---

    def _score_discard_redraw(self, state: MandalaState, t: Template) -> float:
        """Discard is generally last-resort. Prefer discarding colors that
        are blocked in both mandalas (no build + no grow legal)."""
        base = 0.0
        m0 = state.core["mandalas"][0]
        m1 = state.core["mandalas"][1]
        build_blocked_0 = not can_play_color_to_mountain(m0, t.color)
        build_blocked_1 = not can_play_color_to_mountain(m1, t.color)
        field_blocked_0 = not can_play_color_to_field(m0, state.to_play, t.color)
        field_blocked_1 = not can_play_color_to_field(m1, state.to_play, t.color)
        blocked_score = (
            int(build_blocked_0)
            + int(build_blocked_1)
            + int(field_blocked_0)
            + int(field_blocked_1)
        )
        base += blocked_score  # 0..4
        # Prefer discarding fewer cards unless all 4 zones blocked.
        base -= 0.2 * t.count
        # Always dominated by other actions unless we can't build/grow.
        return base - 1.0  # baseline negative so this is rarely chosen
