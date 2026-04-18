"""State encoder for Mandala: player-view state -> fixed-size feature vector.

Design choices:
- The encoder only ever sees a **player view** (produced by get_player_view).
  Features are computable purely from what a player knows. This is the
  imperfect-information departure from Boop's encoder.
- Features are color-multiset counts per public zone, plus Bayesian
  residuals per color (what's still in the unseen pool = opp hand ∪ opp
  starting cup ∪ deck). This matches the poker-solver-style opp-range
  approach: give the net the raw material to learn opp hand inference.
- Recent action history is encoded so the net can condition on "what did
  opponent just do" — essential for opp-range reasoning.
- Current score is NOT exposed as a raw feature, because claiming a color
  early to fill the first river slot looks like a gain under naive
  scoring but is usually a loss once you consider the starting cup cards
  have latent ~3-point value. Instead we expose an EXPECTED-FINAL-SCORE
  estimate that treats unknown-color cup cards (mostly the 2 starting
  cards) as worth ~3 each. The net can still learn to disagree; the
  feature just reduces the noise floor.

History-of-actions is stored externally because get_player_view returns
only the current state. Callers of this encoder pass `history` explicitly.
See make_history() and record_action() helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mlfactory.games.mandala.actions import N_TEMPLATES, Template, template_to_index
from mlfactory.games.mandala.rules import (
    CARDS_PER_COLOR,
    COLORS,
    INITIAL_CUP_SIZE,
    MAX_HAND_SIZE,
    RIVER_SIZE,
)


# --- Feature-layout constants ----------------------------------------------

_N_COLORS = len(COLORS)
_COLOR_IDX = {c: i for i, c in enumerate(COLORS)}

# Zones that contribute color-count features to the encoding.
# Order MUST stay stable once models are trained; document any change.
_ZONES: list[str] = [
    "my_hand",
    "my_cup_visible",  # my cup I see everything; kept separate for clarity
    "opp_cup_visible",  # only non-starting cup cards are visible to me
    "discard",
    "mandala0_mountain",
    "mandala0_field_me",
    "mandala0_field_opp",
    "mandala1_mountain",
    "mandala1_field_me",
    "mandala1_field_opp",
]
_N_ZONES = len(_ZONES)

# History: last K (turn, action_template_index) entries.
HISTORY_K = 8

# Expected latent value of an unknown-color cup card.
# Rationale: a cup card scores `river_slot + 1` (1..6) if its color is in
# the river, else 0. For a color appearing in a river slot, the expected
# slot position is (0+1+2+3+4+5)/6 = 2.5, so value = 3.5. Multiply by the
# probability the color appears in the river. With 6 colors and 6 slots,
# ~P(color in river) ≈ 1 (in practice nearly all colors end up in
# someone's river). So ~3.5 is a generous upper estimate; real games
# average closer to 3. We use 3.0 as a conservative default.
UNKNOWN_CUP_VALUE: float = 3.0

# --- Dimension computation -------------------------------------------------

# Zone color counts: (N_ZONES, N_COLORS)
_ZONE_COUNT_DIM = _N_ZONES * _N_COLORS

# Residual color counts in the unseen pool (opp hand + opp starting cup + deck):
# one scalar per color = N_COLORS
_RESIDUAL_DIM = _N_COLORS

# Rivers: 2 players × 6 slots × one-hot over (N_COLORS + 1 for empty)
_RIVER_DIM = 2 * RIVER_SIZE * (_N_COLORS + 1)

# Phase: one-hot over {playing, destroying, ended}
_PHASE_DIM = 3

# Turn-related scalars:
#   - is_my_turn (0/1)
#   - is_my_claim (0/1, during destruction)
#   - destruction_active (0/1)
#   - destruction_remaining_mask over N_COLORS (which colors remain claimable)
#   - destruction_mandala_index (0/1/-1)
_TURN_SCALAR_DIM = 3 + _N_COLORS + 1

# Expected-final-score for both players: 2 scalars, normalized to ~[0, 1]
_SCORE_DIM = 2

# Committed-points-per-color, per player.
#
# A skilled human evaluating "what is my score right now" computes this
# explicitly: for each color in their river slot S (0-indexed), every cup
# card of that color is worth S+1 points. Summed over colors, this is
# the player's CURRENT committed score (no speculation about future
# claims). The mountain cards collected through the game and the order
# they were collected drive this number more than the visible board.
#
# We expose:
#   - committed_points_per_color: 6 scalars per player = 12. Lets the net
#     reason about specific colors ("I have a lot of red committed").
#   - total_committed: 1 scalar per player = 2. Aggregate.
#   - score_margin: 1 scalar (me_total - opp_total_visible). The single
#     most game-state-relevant scalar a strong human computes.
# For opp, the visible-cup-only portion is computed (we hide their
# starting cup cards). The encoder does NOT speculate about hidden cup
# colors here — that's what expected_final_score does separately.
_COMMITTED_PER_COLOR_DIM = 2 * _N_COLORS  # 12
_COMMITTED_TOTAL_DIM = 2  # me, opp
_SCORE_MARGIN_DIM = 1

# Per-color SEEN counts: total cards of each color visible to me across
# all visible zones. Complements the residual (which says how many are
# UNSEEN); the seen count gives the net a direct view of "what's been
# played out so far" for forward planning.
_SEEN_COUNT_DIM = _N_COLORS  # 6

# Sizes (normalized to rough ranges) for scalars the net might not derive
# easily from zone counts:
#   my_hand_size, opp_hand_size, deck_size, discard_size, my_cup_size,
#   opp_cup_size, turn_number, end_game_trigger_flag, end_game_trigger_type,
#   my_cumulative_draws, opp_cumulative_draws
# The cumulative-draws scalars (last two) are normalized by 50 — a typical
# game has ~30-50 cards drawn per player. They give the net an explicit
# signal of "how much has this player's hand been refreshed", which helps
# opp-hand inference: a player who has barely drawn is still mostly
# holding cards from their starting deal.
_SIZE_SCALAR_DIM = 11

# Action history: K entries, each one-hot over N_TEMPLATES + 1 presence bit.
# To keep the feature vector from exploding, encode each history step as:
#   - one-hot template index (N_TEMPLATES)
#   - a scalar bit saying "this was by me" (1.0) or "by opponent" (0.0)
_HISTORY_STEP_DIM = N_TEMPLATES + 1
_HISTORY_DIM = HISTORY_K * _HISTORY_STEP_DIM

FEATURE_DIM = (
    _ZONE_COUNT_DIM
    + _RESIDUAL_DIM
    + _RIVER_DIM
    + _PHASE_DIM
    + _TURN_SCALAR_DIM
    + _SCORE_DIM
    + _COMMITTED_PER_COLOR_DIM
    + _COMMITTED_TOTAL_DIM
    + _SCORE_MARGIN_DIM
    + _SEEN_COUNT_DIM
    + _SIZE_SCALAR_DIM
    + _HISTORY_DIM
)


# --- History helpers -------------------------------------------------------
#
# History is a list of {"template_index": int, "actor_index": int} dicts.
# Recent K entries are exposed as one-hot template indexes in the encoder.
#
# In addition we track CUMULATIVE counters that don't get clipped:
#   - draws_by_player: list of 2 ints, total cards each player has drawn
#     since the start of the game (over all build_mountain + discard_redraw
#     actions). Useful for opp-hand inference: a player who has drawn lots
#     of cards has a "fresher" hand drawn from more of the residual pool.
#
# To keep backward compat with code that treated history as a plain list,
# we use a list with an optional `_meta` dict attached as an attribute (set
# via setattr). New code uses `make_history()` / `record_action()` /
# `history_meta()` rather than touching the structure directly.


def make_history() -> list[dict]:
    """Fresh empty history with attached cumulative counters."""
    h: list[dict] = []
    # Use a list subclass so we can attach attributes; plain `list` doesn't
    # accept setattr in CPython.
    h = _HistoryList()
    h._meta = {"draws_by_player": [0, 0]}
    return h


class _HistoryList(list):
    """list subclass that accepts attribute attachment for cumulative
    counters. Functionally identical to list for all consumers."""

    pass


def history_meta(history: list[dict] | None) -> dict:
    """Return cumulative counters attached to a history list, or defaults
    if absent (backward compat with plain-list histories from older code)."""
    if history is None:
        return {"draws_by_player": [0, 0]}
    meta = getattr(history, "_meta", None)
    if meta is None:
        return {"draws_by_player": [0, 0]}
    return meta


def _draws_for_template(template_index: int) -> int:
    """Number of cards drawn as a side effect of an action of this
    template kind. Mandala draw rules:
    - build_mountain: draws up to 3 (we use 3 as the upper bound; actual
      may be fewer if hand was already at MAX_HAND_SIZE or deck was
      empty, but for a cumulative-info-leak feature 3 is a fine
      approximation).
    - discard_redraw: draws exactly `count` cards.
    - grow_field, claim_color: no draw.
    """
    from mlfactory.games.mandala.actions import index_to_template

    t = index_to_template(template_index)
    if t.kind == "build_mountain":
        return 3
    if t.kind == "discard_redraw":
        return t.count
    return 0


def record_action(history: list[dict], template_index: int, actor_index: int) -> list[dict]:
    """Append (template_index, actor_index) to history. Returns the updated
    list (also mutates in place for convenience).

    Also updates cumulative counters attached to the history (see
    history_meta()). Falls back to plain-list behaviour if the history
    wasn't created via make_history()."""
    history.append({"template_index": template_index, "actor_index": actor_index})
    # Clip the recent-actions tail (keep last 2*HISTORY_K).
    if len(history) > 2 * HISTORY_K:
        del history[: len(history) - 2 * HISTORY_K]
    # Update cumulative draws (if this history has the meta dict).
    meta = getattr(history, "_meta", None)
    if meta is not None:
        meta["draws_by_player"][actor_index] += _draws_for_template(template_index)
    return history


# --- Feature computation ---------------------------------------------------


def _color_counts(cards: list[dict]) -> np.ndarray:
    """Count cards-per-color for a list of cards. Ignores 'hidden' cards."""
    counts = np.zeros(_N_COLORS, dtype=np.float32)
    for c in cards:
        color = c.get("color")
        if color in _COLOR_IDX:
            counts[_COLOR_IDX[color]] += 1
    return counts


def _river_one_hot(river: list[str | None]) -> np.ndarray:
    """For each of 6 river slots, emit a (N_COLORS+1)-length one-hot
    where the last slot means 'empty'."""
    out = np.zeros((RIVER_SIZE, _N_COLORS + 1), dtype=np.float32)
    for i, color in enumerate(river):
        if color is None:
            out[i, _N_COLORS] = 1.0
        else:
            out[i, _COLOR_IDX[color]] = 1.0
    return out


def committed_points_per_color(player: dict) -> np.ndarray:
    """For each color, points already-committed in this player's cup
    given their current river. Cup card of color C is worth (river_slot_of_C
    + 1) points if C is in the river, else 0. Hidden cup cards
    (opp's starting cards) contribute 0 — they're handled by the
    expected_final_score latent estimate elsewhere.

    Returns an array of shape (N_COLORS,) of point values."""
    out = np.zeros(_N_COLORS, dtype=np.float32)
    river_positions: dict[str, int] = {}
    for i, color in enumerate(player["river"]):
        if color is not None:
            river_positions[color] = i
    for card in player["cup"]:
        color = card.get("color")
        if color is None or color == "hidden":
            continue
        slot = river_positions.get(color)
        if slot is None:
            continue
        out[_COLOR_IDX[color]] += slot + 1
    return out


def expected_final_score(player: dict) -> float:
    """Heuristic estimate of a player's final score from their current
    river + cup state.

    Rules:
    - Cards in the cup whose color IS in the river: actual river-index+1
      points each (already committed).
    - Cards in the cup whose color is NOT yet in the river: worth
      UNKNOWN_CUP_VALUE (latent expected value).
    - 'hidden' cards (opponent's starting cup cards we can't see):
      worth UNKNOWN_CUP_VALUE each.

    Note: this heuristic intentionally does NOT double-count the first
    card of a color that filled a river slot. Each cup card contributes
    exactly once (engine stores it in cup; river stores a color marker).
    """
    river_positions: dict[str, int] = {}
    for i, color in enumerate(player["river"]):
        if color is not None:
            river_positions[color] = i

    total = 0.0
    for card in player["cup"]:
        color = card.get("color")
        if color == "hidden" or color is None:
            total += UNKNOWN_CUP_VALUE
        elif color in river_positions:
            total += river_positions[color] + 1  # slot 0 = 1 point, slot 5 = 6 points
        else:
            # Color is in cup but not yet claimed into river — worth ~UNKNOWN_CUP_VALUE
            # if we imagine claiming it later (but only if we later add the color to
            # our river). A common case during mid-game.
            total += UNKNOWN_CUP_VALUE

    return total


# --- Main encode() ---------------------------------------------------------


def encode_view(
    view: dict,
    my_player_index: int,
    history: list[dict] | None = None,
) -> np.ndarray:
    """Encode a player-view state into a 1-D float32 feature vector.

    Parameters
    ----------
    view : dict
        The result of `get_player_view(state, my_player_index)`.
    my_player_index : int
        Which slot (0 or 1) the encoded perspective belongs to.
    history : list[dict] | None
        Recent action history, newest last. If None, treated as empty
        (equivalent to a fresh game).

    Returns
    -------
    features : np.ndarray (FEATURE_DIM,) float32
    """
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    cursor = 0

    me = view["players"][my_player_index]
    opp = view["players"][1 - my_player_index]
    md0 = view["mandalas"][0]
    md1 = view["mandalas"][1]

    # Zone color counts. Normalized by CARDS_PER_COLOR so each feature is in [0, 1].
    # This matches how the net expects dense, roughly-unit-scaled inputs.
    zones_cards: dict[str, list[dict]] = {
        "my_hand": me["hand"],
        "my_cup_visible": me["cup"],
        "opp_cup_visible": [c for c in opp["cup"] if c.get("color") != "hidden"],
        "discard": view["discardPile"],
        "mandala0_mountain": md0["mountain"],
        "mandala0_field_me": md0["fields"][my_player_index],
        "mandala0_field_opp": md0["fields"][1 - my_player_index],
        "mandala1_mountain": md1["mountain"],
        "mandala1_field_me": md1["fields"][my_player_index],
        "mandala1_field_opp": md1["fields"][1 - my_player_index],
    }
    for zone in _ZONES:
        counts = _color_counts(zones_cards[zone])
        features[cursor : cursor + _N_COLORS] = counts / CARDS_PER_COLOR
        cursor += _N_COLORS

    # Residual color counts in the unseen pool. For each color,
    #   residual = CARDS_PER_COLOR - (sum of visible counts of that color
    #              across all zones I can see, plus my own hidden starting
    #              cup cards if applicable).
    # Since my cup is fully known to me and opp's starting cup is not,
    # the "visible" accounting already excludes exactly the hidden cards.
    # Residual is distributed over {opp hand, opp starting cup, deck}.
    visible_total = np.zeros(_N_COLORS, dtype=np.float32)
    for zone in _ZONES:
        visible_total += _color_counts(zones_cards[zone])
    residual = CARDS_PER_COLOR - visible_total
    # Clip defensively (shouldn't go negative but let's be safe).
    residual = np.clip(residual, 0, CARDS_PER_COLOR).astype(np.float32)
    features[cursor : cursor + _N_COLORS] = residual / CARDS_PER_COLOR
    cursor += _N_COLORS

    # Rivers (both players'). Me first, then opp.
    my_river = _river_one_hot(me["river"]).flatten()
    opp_river = _river_one_hot(opp["river"]).flatten()
    features[cursor : cursor + my_river.size] = my_river
    cursor += my_river.size
    features[cursor : cursor + opp_river.size] = opp_river
    cursor += opp_river.size

    # Phase one-hot.
    phase = view["phase"]
    phase_idx = {"playing": 0, "destroying": 1, "ended": 2}.get(phase, 0)
    features[cursor + phase_idx] = 1.0
    cursor += _PHASE_DIM

    # Turn scalars.
    is_my_turn = 1.0 if view.get("currentPlayerIndex") == my_player_index else 0.0
    dest = view.get("destruction")
    is_my_claim = 0.0
    destruction_active = 0.0
    dest_mask = np.zeros(_N_COLORS, dtype=np.float32)
    dest_mandala_idx = -1.0
    if dest is not None:
        destruction_active = 1.0
        is_my_claim = 1.0 if dest.get("currentClaimerIndex") == my_player_index else 0.0
        for color in dest.get("remainingColors", []):
            dest_mask[_COLOR_IDX[color]] = 1.0
        dest_mandala_idx = float(dest.get("mandalaIndex", -1))
    features[cursor] = is_my_turn
    features[cursor + 1] = is_my_claim
    features[cursor + 2] = destruction_active
    features[cursor + 3 : cursor + 3 + _N_COLORS] = dest_mask
    features[cursor + 3 + _N_COLORS] = dest_mandala_idx
    cursor += _TURN_SCALAR_DIM

    # Expected final score for both players. Normalize by a rough max (42
    # = 6 slots × 7 cards-avg × upper points) so values land in ~[0, 1].
    my_score = expected_final_score(me)
    opp_score = expected_final_score(opp)
    score_norm = 40.0
    features[cursor] = my_score / score_norm
    features[cursor + 1] = opp_score / score_norm
    cursor += _SCORE_DIM

    # Committed-points-per-color (12 scalars: 6 colors × {me, opp}).
    # Each scalar = points already committed for that player from cup
    # cards of that color, given current river. Normalized by 18 (the
    # max possible: all 18 cards of a colour in cup, river slot 6 → 18*6=108,
    # but practical max far lower; 18 is a soft cap).
    me_committed_per_color = committed_points_per_color(me)
    opp_committed_per_color = committed_points_per_color(opp)  # opp's hidden cup → 0
    committed_norm = 18.0
    features[cursor : cursor + _N_COLORS] = me_committed_per_color / committed_norm
    cursor += _N_COLORS
    features[cursor : cursor + _N_COLORS] = opp_committed_per_color / committed_norm
    cursor += _N_COLORS

    # Total committed per player (2 scalars). Net-redundant with above
    # but explicit aggregates help small MLPs.
    me_committed_total = float(me_committed_per_color.sum())
    opp_committed_total = float(opp_committed_per_color.sum())
    total_norm = 40.0
    features[cursor + 0] = me_committed_total / total_norm
    features[cursor + 1] = opp_committed_total / total_norm
    cursor += _COMMITTED_TOTAL_DIM

    # Score margin (1 scalar): the most game-state-relevant single number.
    # Includes ALL signal a strong human computes ('I'm ahead by 7'),
    # mixing committed totals with the latent estimates of unknown cup
    # cards. Scale: in [-1, 1] roughly.
    margin = (me_committed_total + my_score - opp_committed_total - opp_score) / total_norm
    features[cursor] = max(-1.0, min(1.0, margin))
    cursor += _SCORE_MARGIN_DIM

    # Per-color SEEN counts: how many of each color have I observed
    # (across all my visible zones combined). Complements residual
    # (residual = 18 - seen). Net can use either or both for inference
    # about what's left in opp's hand / deck.
    seen = visible_total.copy()
    features[cursor : cursor + _N_COLORS] = seen / CARDS_PER_COLOR
    cursor += _N_COLORS

    # Size scalars (normalized).
    features[cursor + 0] = len(me["hand"]) / MAX_HAND_SIZE
    features[cursor + 1] = len(opp["hand"]) / MAX_HAND_SIZE
    features[cursor + 2] = len(view["deck"]) / 108
    features[cursor + 3] = len(view["discardPile"]) / 108
    features[cursor + 4] = len(me["cup"]) / 40  # cups can grow large
    features[cursor + 5] = len(opp["cup"]) / 40
    features[cursor + 6] = view.get("turnNumber", 1) / 100
    end_trigger = view.get("endGameTrigger")
    features[cursor + 7] = 1.0 if end_trigger else 0.0
    features[cursor + 8] = (
        1.0
        if end_trigger == "sixth_river_color"
        else (0.5 if end_trigger == "deck_exhausted" else 0.0)
    )
    # Cumulative draws by each player. Helps opp-hand inference: a player
    # who has barely drawn is mostly still holding their initial deal,
    # while a player who has drawn many cards has a fresh hand from a
    # broader sample of the residual pool.
    meta = history_meta(history)
    draws = meta["draws_by_player"]
    draws_norm = 50.0
    features[cursor + 9] = min(draws[my_player_index] / draws_norm, 1.0)
    features[cursor + 10] = min(draws[1 - my_player_index] / draws_norm, 1.0)
    cursor += _SIZE_SCALAR_DIM

    # History (last HISTORY_K entries, padded left with zeros if fewer).
    hist = history or []
    tail = hist[-HISTORY_K:]
    pad = HISTORY_K - len(tail)
    for slot in range(pad):
        # empty slot: zeros (already initialized)
        cursor += _HISTORY_STEP_DIM
    for entry in tail:
        tmpl_idx = entry["template_index"]
        if 0 <= tmpl_idx < N_TEMPLATES:
            features[cursor + tmpl_idx] = 1.0
        # actor bit: 1 if me, 0 if opp
        features[cursor + N_TEMPLATES] = 1.0 if entry["actor_index"] == my_player_index else 0.0
        cursor += _HISTORY_STEP_DIM

    assert cursor == FEATURE_DIM, (cursor, FEATURE_DIM)
    return features


def legal_mask_from_view(view: dict) -> np.ndarray:
    """Convenience: run the legal-template mask on a view.

    The view has the same structure as a full state, and legal_mask in
    actions.py only uses public / player-accessible fields, so this is
    the same computation."""
    from mlfactory.games.mandala.actions import legal_mask as _mask

    return _mask(view)
