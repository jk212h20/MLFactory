"""Mandala game engine — Python port of mandala-web/game.js.

State is represented as plain dicts/lists matching the JS shape 1:1, so
state objects can be compared byte-for-byte with JSON-serialised JS state
for parity testing.

Key differences from the JS source:
- `createGame` accepts an optional `rng` (random.Random instance) so
  JS↔Python parity can be established with matched RNG streams. If not
  provided, a fresh Random() is created.
- The shuffle algorithm is Fisher–Yates matching the JS implementation
  exactly: we call `rng.random()` in the same order as JS's
  `Math.random()` inside the loop, producing identical permutations for
  identical streams. The parity harness injects a shared generator.
- All mutation-producing functions deepcopy the input state, matching JS's
  structuredClone discipline.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any

# -------------------- CONSTANTS --------------------

COLORS: list[str] = ["red", "orange", "yellow", "green", "purple", "black"]
CARDS_PER_COLOR: int = 18
MAX_HAND_SIZE: int = 8
INITIAL_HAND_SIZE: int = 6
INITIAL_CUP_SIZE: int = 2
INITIAL_MOUNTAIN_SIZE: int = 2
RIVER_SIZE: int = 6

# Total cards in the deck: 6 colors × 18 cards = 108.
TOTAL_CARDS: int = len(COLORS) * CARDS_PER_COLOR


# -------------------- UTILITIES --------------------


def create_deck() -> list[dict]:
    """Produce an ordered 108-card deck with JS-compatible ids.

    Ids are `<color>-<running_counter>` with the counter incremented across
    all cards so each id is globally unique (matches game.js:25).
    """
    deck: list[dict] = []
    counter = 0
    for color in COLORS:
        for _ in range(CARDS_PER_COLOR):
            deck.append({"id": f"{color}-{counter}", "color": color})
            counter += 1
    return deck


def shuffle_deck(deck: list[dict], rng: random.Random) -> list[dict]:
    """Fisher–Yates in-place shuffle of a copy. Matches game.js:31-38.

    The JS reference picks `j = Math.floor(Math.random() * (i + 1))` for each
    i from len-1 down to 1. To produce an identical permutation we consume
    `rng.random()` in the same order.
    """
    shuffled = list(deck)
    for i in range(len(shuffled) - 1, 0, -1):
        j = math.floor(rng.random() * (i + 1))
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return shuffled


def draw_cards(state: dict, count: int, rng: random.Random) -> tuple[list[dict], dict]:
    """Draw `count` cards from state.deck, reshuffling discardPile into deck if
    needed. Returns (cards_drawn, new_state). Matches game.js:40-67.

    If the deck empties and gets reshuffled, sets endGameTrigger to
    'deck_exhausted' (if not already set).
    """
    new_state = dict(state)
    # Shallow-copy the mutable containers we touch so callers' original is safe.
    new_state["deck"] = list(state["deck"])
    new_state["discardPile"] = list(state["discardPile"])

    cards: list[dict] = []
    for _ in range(count):
        if len(new_state["deck"]) == 0:
            if len(new_state["discardPile"]) == 0:
                break  # nothing left to draw
            new_state["deck"] = shuffle_deck(new_state["discardPile"], rng)
            new_state["discardPile"] = []
            if not new_state.get("endGameTrigger"):
                new_state["endGameTrigger"] = "deck_exhausted"
        card = new_state["deck"][0]
        new_state["deck"] = new_state["deck"][1:]
        cards.append(card)

    return cards, new_state


def _create_player(player_id: str) -> dict:
    return {
        "id": player_id,
        "hand": [],
        "cup": [],
        "river": [None] * RIVER_SIZE,
        # startingCupCount is added at deal time (see create_game).
    }


def _create_mandala() -> dict:
    return {"mountain": [], "fields": [[], []]}


# -------------------- GAME INITIALIZATION --------------------


def create_game(
    player1_id: str = "p0",
    player2_id: str = "p1",
    rng: random.Random | None = None,
) -> dict:
    """Create a fresh game. Matches game.js:89-128.

    `rng` controls both the initial shuffle and later reshuffles. If None,
    a fresh unseeded Random is used. Deterministic for a given seeded rng.
    """
    if rng is None:
        rng = random.Random()

    deck = shuffle_deck(create_deck(), rng)

    players = [_create_player(player1_id), _create_player(player2_id)]

    # Deal hands (INITIAL_HAND_SIZE each).
    for p in range(2):
        players[p]["hand"] = deck[:INITIAL_HAND_SIZE]
        deck = deck[INITIAL_HAND_SIZE:]

    # Deal cups (starting cups, hidden from opponent).
    for p in range(2):
        cup_cards = deck[:INITIAL_CUP_SIZE]
        players[p]["cup"] = cup_cards
        players[p]["startingCupCount"] = INITIAL_CUP_SIZE
        deck = deck[INITIAL_CUP_SIZE:]

    # Create mandalas and deal initial mountain cards (2 per mandala).
    mandalas = [_create_mandala(), _create_mandala()]
    for m in range(2):
        mandalas[m]["mountain"] = deck[:INITIAL_MOUNTAIN_SIZE]
        deck = deck[INITIAL_MOUNTAIN_SIZE:]

    return {
        "deck": deck,
        "discardPile": [],
        "players": players,
        "mandalas": mandalas,
        "currentPlayerIndex": 0,
        "phase": "playing",
        "endGameTrigger": None,
        "destruction": None,
        "lastMandalaPlayerIndex": None,
        "turnNumber": 1,
    }


# -------------------- RULE-OF-COLOR CHECKS --------------------


def _colors_in_mountain(mandala: dict) -> set[str]:
    return {c["color"] for c in mandala["mountain"]}


def _colors_in_field(mandala: dict, player_index: int) -> set[str]:
    return {c["color"] for c in mandala["fields"][player_index]}


def _colors_in_mandala(mandala: dict) -> set[str]:
    colors = set()
    for c in mandala["mountain"]:
        colors.add(c["color"])
    for c in mandala["fields"][0]:
        colors.add(c["color"])
    for c in mandala["fields"][1]:
        colors.add(c["color"])
    return colors


def can_play_color_to_mountain(mandala: dict, color: str) -> bool:
    """Matches game.js:150-159. A color is blocked from the mountain if it's
    in EITHER field — but it's fine if it's already in the mountain."""
    if color in _colors_in_field(mandala, 0):
        return False
    if color in _colors_in_field(mandala, 1):
        return False
    return True


def can_play_color_to_field(mandala: dict, player_index: int, color: str) -> bool:
    """Matches game.js:161-175. Blocked if the color is in the mountain or in
    the opponent's field. Own field is fine."""
    if color in _colors_in_mountain(mandala):
        return False
    opponent_index = 1 - player_index
    if color in _colors_in_field(mandala, opponent_index):
        return False
    return True


def _is_mandala_complete(mandala: dict) -> bool:
    return len(_colors_in_mandala(mandala)) == 6


# -------------------- ACTION VALIDATION --------------------


def validate_build_mountain(state: dict, card_id: str, mandala_index: int) -> dict:
    if state["phase"] != "playing":
        return {"valid": False, "error": "Cannot build mountain during this phase"}
    player = state["players"][state["currentPlayerIndex"]]
    card = next((c for c in player["hand"] if c["id"] == card_id), None)
    if card is None:
        return {"valid": False, "error": "Card not in hand"}
    mandala = state["mandalas"][mandala_index]
    if not can_play_color_to_mountain(mandala, card["color"]):
        return {
            "valid": False,
            "error": f"Cannot play {card['color']} to this mountain (Rule of Color)",
        }
    return {"valid": True}


def validate_grow_field(state: dict, card_ids: list[str], mandala_index: int) -> dict:
    if state["phase"] != "playing":
        return {"valid": False, "error": "Cannot grow field during this phase"}
    if len(card_ids) == 0:
        return {"valid": False, "error": "Must play at least one card"}

    player = state["players"][state["currentPlayerIndex"]]
    cards = []
    for cid in card_ids:
        c = next((c for c in player["hand"] if c["id"] == cid), None)
        if c is None:
            return {"valid": False, "error": "Some cards not in hand"}
        cards.append(c)

    color = cards[0]["color"]
    if any(c["color"] != color for c in cards):
        return {"valid": False, "error": "All cards must be the same color"}

    mandala = state["mandalas"][mandala_index]
    if not can_play_color_to_field(mandala, state["currentPlayerIndex"], color):
        return {
            "valid": False,
            "error": f"Cannot play {color} to this field (Rule of Color)",
        }

    # JS check: cardIds.length >= player.hand.length  →  must keep at least 1 in hand
    if len(card_ids) >= len(player["hand"]):
        return {"valid": False, "error": "Must keep at least 1 card in hand"}

    return {"valid": True}


def validate_discard_redraw(state: dict, card_ids: list[str]) -> dict:
    if state["phase"] != "playing":
        return {"valid": False, "error": "Cannot discard during this phase"}
    if len(card_ids) == 0:
        return {"valid": False, "error": "Must discard at least one card"}

    player = state["players"][state["currentPlayerIndex"]]
    cards = []
    for cid in card_ids:
        c = next((c for c in player["hand"] if c["id"] == cid), None)
        if c is None:
            return {"valid": False, "error": "Some cards not in hand"}
        cards.append(c)

    color = cards[0]["color"]
    if any(c["color"] != color for c in cards):
        return {"valid": False, "error": "All cards must be the same color"}

    return {"valid": True}


def validate_claim_color(state: dict, color: str) -> dict:
    if state["phase"] != "destroying":
        return {"valid": False, "error": "Can only claim colors during destruction phase"}
    if not state.get("destruction"):
        return {"valid": False, "error": "No destruction in progress"}
    if state["destruction"]["currentClaimerIndex"] != state["currentPlayerIndex"]:
        return {"valid": False, "error": "Not your turn to claim"}
    if color not in state["destruction"]["remainingColors"]:
        return {"valid": False, "error": "Color not available to claim"}
    return {"valid": True}


# -------------------- ACTION EXECUTION --------------------


def _execute_build_mountain(
    state: dict, card_id: str, mandala_index: int, rng: random.Random
) -> dict:
    validation = validate_build_mountain(state, card_id, mandala_index)
    if not validation["valid"]:
        return {"success": False, "error": validation["error"], "newState": state}

    new_state = copy.deepcopy(state)
    player_index = new_state["currentPlayerIndex"]
    player = new_state["players"][player_index]
    card_index = next(i for i, c in enumerate(player["hand"]) if c["id"] == card_id)
    card = player["hand"][card_index]

    # Remove from hand, add to mountain.
    player["hand"].pop(card_index)
    new_state["mandalas"][mandala_index]["mountain"].append(card)
    new_state["lastMandalaPlayerIndex"] = player_index

    # Draw up to 3, capped at MAX_HAND_SIZE.
    cards_to_draw = min(3, MAX_HAND_SIZE - len(player["hand"]))
    drawn, new_state = draw_cards(new_state, cards_to_draw, rng)
    new_state["players"][player_index]["hand"].extend(drawn)

    if _is_mandala_complete(new_state["mandalas"][mandala_index]):
        new_state = _start_destruction(new_state, mandala_index)
    else:
        new_state["currentPlayerIndex"] = 1 - player_index
        new_state["turnNumber"] += 1

    return {"success": True, "newState": new_state}


def _execute_grow_field(
    state: dict, card_ids: list[str], mandala_index: int, rng: random.Random
) -> dict:
    validation = validate_grow_field(state, card_ids, mandala_index)
    if not validation["valid"]:
        return {"success": False, "error": validation["error"], "newState": state}

    new_state = copy.deepcopy(state)
    player_index = new_state["currentPlayerIndex"]
    player = new_state["players"][player_index]

    cards_to_play = []
    for cid in card_ids:
        idx = next(i for i, c in enumerate(player["hand"]) if c["id"] == cid)
        cards_to_play.append(player["hand"].pop(idx))

    new_state["mandalas"][mandala_index]["fields"][player_index].extend(cards_to_play)
    new_state["lastMandalaPlayerIndex"] = player_index
    # No draw for grow_field.

    if _is_mandala_complete(new_state["mandalas"][mandala_index]):
        new_state = _start_destruction(new_state, mandala_index)
    else:
        new_state["currentPlayerIndex"] = 1 - player_index
        new_state["turnNumber"] += 1

    return {"success": True, "newState": new_state}


def _execute_discard_redraw(state: dict, card_ids: list[str], rng: random.Random) -> dict:
    validation = validate_discard_redraw(state, card_ids)
    if not validation["valid"]:
        return {"success": False, "error": validation["error"], "newState": state}

    new_state = copy.deepcopy(state)
    player_index = new_state["currentPlayerIndex"]
    player = new_state["players"][player_index]

    discarded = []
    for cid in card_ids:
        idx = next(i for i, c in enumerate(player["hand"]) if c["id"] == cid)
        discarded.append(player["hand"].pop(idx))
    new_state["discardPile"].extend(discarded)

    drawn, new_state = draw_cards(new_state, len(discarded), rng)
    new_state["players"][player_index]["hand"].extend(drawn)

    new_state["currentPlayerIndex"] = 1 - player_index
    new_state["turnNumber"] += 1

    return {"success": True, "newState": new_state}


# -------------------- DESTRUCTION PHASE --------------------


def _start_destruction(state: dict, mandala_index: int) -> dict:
    new_state = copy.deepcopy(state)
    mandala = new_state["mandalas"][mandala_index]

    field0_count = len(mandala["fields"][0])
    field1_count = len(mandala["fields"][1])
    if field0_count > field1_count:
        first_claimer = 0
    elif field1_count > field0_count:
        first_claimer = 1
    else:
        # Tie → the player who did NOT play last goes first.
        first_claimer = 1 if new_state["lastMandalaPlayerIndex"] == 0 else 0

    # Order of remaining colors must match JS: JS uses Array.from(new Set(...))
    # which preserves first-seen order from the mountain.
    seen: list[str] = []
    for c in mandala["mountain"]:
        if c["color"] not in seen:
            seen.append(c["color"])

    new_state["phase"] = "destroying"
    new_state["destruction"] = {
        "mandalaIndex": mandala_index,
        "currentClaimerIndex": first_claimer,
        "remainingColors": seen,
    }
    new_state["currentPlayerIndex"] = first_claimer

    return new_state


def _execute_claim_color(state: dict, color: str, rng: random.Random) -> dict:
    validation = validate_claim_color(state, color)
    if not validation["valid"]:
        return {"success": False, "error": validation["error"], "newState": state}

    new_state = copy.deepcopy(state)
    destruction = new_state["destruction"]
    mandala_index = destruction["mandalaIndex"]
    mandala = new_state["mandalas"][mandala_index]
    player_index = destruction["currentClaimerIndex"]
    player = new_state["players"][player_index]

    # Pull all cards of this color from the mountain.
    claimed = [c for c in mandala["mountain"] if c["color"] == color]
    mandala["mountain"] = [c for c in mandala["mountain"] if c["color"] != color]

    if len(mandala["fields"][player_index]) == 0:
        # No field cards → all claimed cards discarded (penalty).
        new_state["discardPile"].extend(claimed)
    else:
        # Find this color in the claimer's river.
        river = player["river"]
        if color in river:
            # Already in river — all cards go to cup.
            player["cup"].extend(claimed)
        else:
            # New color — first card goes to next empty river slot.
            try:
                first_empty = river.index(None)
            except ValueError:
                first_empty = -1
            if first_empty != -1:
                river[first_empty] = color
                # Rest go to cup.
                player["cup"].extend(claimed[1:])
                if first_empty == 5:
                    new_state["endGameTrigger"] = "sixth_river_color"
            else:
                # River full — shouldn't happen, defensive.
                player["cup"].extend(claimed)

    destruction["remainingColors"] = [c for c in destruction["remainingColors"] if c != color]

    if len(destruction["remainingColors"]) == 0:
        new_state = _finish_destruction(new_state, rng)
    else:
        destruction["currentClaimerIndex"] = 1 - player_index
        new_state["currentPlayerIndex"] = destruction["currentClaimerIndex"]

    return {"success": True, "newState": new_state}


def _finish_destruction(state: dict, rng: random.Random) -> dict:
    new_state = copy.deepcopy(state)
    mandala_index = new_state["destruction"]["mandalaIndex"]
    mandala = new_state["mandalas"][mandala_index]

    # Fields discarded into the pile.
    new_state["discardPile"].extend(mandala["fields"][0])
    new_state["discardPile"].extend(mandala["fields"][1])
    mandala["fields"] = [[], []]

    if new_state.get("endGameTrigger"):
        new_state["phase"] = "ended"
        new_state["destruction"] = None
        return new_state

    # Refill mountain from deck (may reshuffle).
    mountain_cards, new_state = draw_cards(new_state, INITIAL_MOUNTAIN_SIZE, rng)
    new_state["mandalas"][mandala_index]["mountain"] = mountain_cards

    new_state["phase"] = "playing"
    new_state["destruction"] = None
    new_state["currentPlayerIndex"] = 1 - new_state["lastMandalaPlayerIndex"]
    new_state["turnNumber"] += 1

    return new_state


# -------------------- MAIN ACTION HANDLER --------------------


def perform_action(state: dict, action: dict, rng: random.Random | None = None) -> dict:
    """Apply an action. Matches game.js:553-566.

    The JS engine does not take an RNG here — it uses the global Math.random.
    We accept an optional rng for determinism; falls back to a fresh Random()
    if none is supplied.
    """
    if rng is None:
        rng = random.Random()

    action_type = action.get("type")
    if action_type == "build_mountain":
        return _execute_build_mountain(state, action["cardId"], action["mandalaIndex"], rng)
    if action_type == "grow_field":
        return _execute_grow_field(state, action["cardIds"], action["mandalaIndex"], rng)
    if action_type == "discard_redraw":
        return _execute_discard_redraw(state, action["cardIds"], rng)
    if action_type == "claim_color":
        return _execute_claim_color(state, action["color"], rng)
    return {"success": False, "error": "Unknown action type", "newState": state}


# -------------------- VALID ACTIONS ENUMERATION --------------------


def get_valid_actions(state: dict) -> dict:
    """Enumerate every legal action. Matches game.js:572-640.

    Returns a dict with keys buildMountain, growField, discardRedraw,
    claimColor. Each value is a list of action-parameter dicts (not full
    action objects — caller prepends the 'type' field).
    """
    result = {
        "buildMountain": [],
        "growField": [],
        "discardRedraw": [],
        "claimColor": [],
    }

    if state["phase"] == "ended":
        return result

    player = state["players"][state["currentPlayerIndex"]]

    if state["phase"] == "destroying":
        d = state.get("destruction")
        if d and d["currentClaimerIndex"] == state["currentPlayerIndex"]:
            result["claimColor"] = list(d["remainingColors"])
        return result

    # Playing phase: group hand by color, preserving order (JS Map preserves
    # insertion order; we mirror that with a dict).
    hand_by_color: dict[str, list[dict]] = {}
    for card in player["hand"]:
        hand_by_color.setdefault(card["color"], []).append(card)

    # Build-mountain: one option per (card, mandala).
    for card in player["hand"]:
        for m in (0, 1):
            if can_play_color_to_mountain(state["mandalas"][m], card["color"]):
                result["buildMountain"].append({"cardId": card["id"], "mandalaIndex": m})

    # Grow-field: must keep at least 1 in hand.
    if len(player["hand"]) > 1:
        for color, cards in hand_by_color.items():
            for m in (0, 1):
                if can_play_color_to_field(
                    state["mandalas"][m], state["currentPlayerIndex"], color
                ):
                    max_count = min(len(cards), len(player["hand"]) - 1)
                    for count in range(1, max_count + 1):
                        result["growField"].append(
                            {
                                "cardIds": [c["id"] for c in cards[:count]],
                                "mandalaIndex": m,
                            }
                        )

    # Discard-redraw: 1..k of each color in hand.
    for color, cards in hand_by_color.items():
        for count in range(1, len(cards) + 1):
            result["discardRedraw"].append({"cardIds": [c["id"] for c in cards[:count]]})

    return result


# -------------------- SCORING --------------------


def calculate_score(player: dict) -> int:
    """Matches game.js:646-657. Cards whose color is in the river score
    (river-index + 1) points; others score 0."""
    score = 0
    for card in player["cup"]:
        try:
            idx = player["river"].index(card["color"])
        except ValueError:
            continue
        score += idx + 1
    return score


def get_winner(state: dict) -> dict | None:
    """Matches game.js:659-680. Returns {winnerId, scores} or None if the game
    hasn't ended."""
    if state["phase"] != "ended":
        return None
    score0 = calculate_score(state["players"][0])
    score1 = calculate_score(state["players"][1])
    if score0 > score1:
        winner_id = state["players"][0]["id"]
    elif score1 > score0:
        winner_id = state["players"][1]["id"]
    else:
        cup0 = len(state["players"][0]["cup"])
        cup1 = len(state["players"][1]["cup"])
        winner_id = state["players"][0]["id"] if cup0 < cup1 else state["players"][1]["id"]
    return {"winnerId": winner_id, "scores": [score0, score1]}


# -------------------- PLAYER VIEW (HIDE HIDDEN INFO) --------------------


_HIDDEN_CARD = {"id": "hidden", "color": "hidden"}


def get_player_view(state: dict, player_index: int) -> dict:
    """Return a version of `state` with opponent's hand, opponent's starting
    cup cards, and the deck contents masked. Matches game.js:690-714.

    Note: the number of hidden cards is preserved (length-equal to hide
    count), so the opponent's hand size and starting-cup size are still
    observable; only the identities/colors are hidden.
    """
    view = copy.deepcopy(state)
    opponent_index = 1 - player_index
    opponent = view["players"][opponent_index]

    opponent["hand"] = [dict(_HIDDEN_CARD) for _ in opponent["hand"]]

    starting = opponent.get("startingCupCount", INITIAL_CUP_SIZE)
    if len(opponent["cup"]) <= starting:
        opponent["cup"] = [dict(_HIDDEN_CARD) for _ in opponent["cup"]]
    else:
        opponent["cup"] = [dict(_HIDDEN_CARD) for _ in range(starting)] + opponent["cup"][starting:]

    view["deck"] = [dict(_HIDDEN_CARD) for _ in view["deck"]]

    return view
