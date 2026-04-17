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

# Sizes (normalized to rough ranges) for scalars the net might not derive
# easily from zone counts:
#   my_hand_size, opp_hand_size, deck_size, discard_size, my_cup_size,
#   opp_cup_size, turn_number, end_game_trigger_flag, end_game_trigger_type
_SIZE_SCALAR_DIM = 9

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
    + _SIZE_SCALAR_DIM
    + _HISTORY_DIM
)


# --- History helpers -------------------------------------------------------


def make_history() -> list[dict]:
    """Fresh empty history. Shared across a game's encoding calls."""
    return []


def record_action(history: list[dict], template_index: int, actor_index: int) -> list[dict]:
    """Append (template_index, actor_index) to history. Returns the updated
    list (also mutates in place for convenience)."""
    history.append({"template_index": template_index, "actor_index": actor_index})
    # Clip to HISTORY_K most recent (keep it bounded so features stay stable).
    if len(history) > 2 * HISTORY_K:
        del history[: len(history) - 2 * HISTORY_K]
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
