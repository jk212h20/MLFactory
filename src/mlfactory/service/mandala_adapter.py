"""Convert between mandala-web JSON game state and Python MandalaState.

The mandala-web Node server's `getPlayerView(state, playerIndex)` returns a
JSON state that is structurally identical to the Python engine's dict
state (the Python engine in src/mlfactory/games/mandala/rules.py is a
faithful port of mandala-web/game.js). The shape we receive over the
wire from the site:

    {
      "deck":       [{id, color}, ...],          # may be hidden cards in view
      "discardPile":[{id, color}, ...],
      "players": [
        {id, hand: [...], cup: [...], river: [c|null x6], startingCupCount},
        {id, hand: [...], cup: [...], river: [c|null x6], startingCupCount}
      ],
      "mandalas": [{mountain, fields:[[],[]]}, {mountain, fields:[[],[]]}],
      "currentPlayerIndex": 0|1,
      "phase":              "playing"|"destroying"|"ended",
      "endGameTrigger":     null|"deck_exhausted"|"sixth_river_color",
      "destruction":        null|{mandalaIndex, currentClaimerIndex, remainingColors},
      "lastMandalaPlayerIndex": null|0|1,
      "turnNumber":         int
    }

Hidden cards (opponent hand, opponent starting cup cards, deck contents) are
serialized as `{id: "hidden", color: "hidden"}` by the site's view filter
(matches our Python `get_player_view`). The bot's own hand is fully
visible — that's exactly what the encoder needs.

Action history is OPTIONAL. The site MAY include a top-level "history"
field (a list of `{templateIndex, actorIndex}` dicts) reconstructed from
the game so far. If absent, we play with empty history (the trained
encoder still works, it just loses the history feature signal — modest
quality loss but the bot still plays).
"""

from __future__ import annotations

from typing import Any

from mlfactory.games.mandala.actions import (
    Template,
    index_to_template,
    template_to_engine_action,
)
from mlfactory.games.mandala.encode import _HistoryList
from mlfactory.games.mandala.env import MandalaState


# -- JSON -> MandalaState ---------------------------------------------------


def parse_mandala_state(payload: dict[str, Any]) -> MandalaState:
    """Parse a mandala-web JSON state into our internal MandalaState.

    The payload IS the engine's dict state (mandala-web's game.js and our
    rules.py both use the same shape). We accept an optional sibling
    `history` field for the encoder's history features.

    The state must be from the bot's own perspective — i.e. its own hand
    must be fully visible (real card ids). The opponent's hidden zones may
    be filled with `{id: "hidden", color: "hidden"}` placeholders.
    """
    if "players" not in payload or "mandalas" not in payload:
        raise ValueError("payload missing required keys 'players'/'mandalas'")
    if not isinstance(payload["players"], list) or len(payload["players"]) != 2:
        raise ValueError("payload.players must be a list of length 2")
    if not isinstance(payload["mandalas"], list) or len(payload["mandalas"]) != 2:
        raise ValueError("payload.mandalas must be a list of length 2")

    # Defensive copy so the caller's dict isn't aliased into our state.
    # We do NOT deep-clone every card; the engine itself uses
    # structuredClone-equivalent (copy.deepcopy) on every step.
    core = {
        "deck": list(payload.get("deck", [])),
        "discardPile": list(payload.get("discardPile", [])),
        "players": [dict(p) for p in payload["players"]],
        "mandalas": [
            {
                "mountain": list(m.get("mountain", [])),
                "fields": [list(m["fields"][0]), list(m["fields"][1])],
            }
            for m in payload["mandalas"]
        ],
        "currentPlayerIndex": int(payload["currentPlayerIndex"]),
        "phase": payload.get("phase", "playing"),
        "endGameTrigger": payload.get("endGameTrigger"),
        "destruction": payload.get("destruction"),
        "lastMandalaPlayerIndex": payload.get("lastMandalaPlayerIndex"),
        "turnNumber": int(payload.get("turnNumber", 1)),
    }

    # Re-list each player's nested arrays so we don't mutate caller objects.
    for i, p in enumerate(core["players"]):
        core["players"][i] = {
            "id": p.get("id", f"p{i}"),
            "hand": list(p.get("hand", [])),
            "cup": list(p.get("cup", [])),
            "river": list(p.get("river", [None] * 6)),
            "startingCupCount": int(p.get("startingCupCount", 0)),
        }

    # History reconstruction. Accept either:
    #   list of {templateIndex, actorIndex}        (camelCase, JS style)
    #   list of {template_index, actor_index}      (snake_case, Python style)
    history_in = payload.get("history") or []
    history = _HistoryList()
    history._meta = {"draws_by_player": [0, 0]}
    for entry in history_in:
        if "templateIndex" in entry:
            ti = int(entry["templateIndex"])
            ai = int(entry["actorIndex"])
        elif "template_index" in entry:
            ti = int(entry["template_index"])
            ai = int(entry["actor_index"])
        else:
            # Skip malformed entries silently (history is best-effort).
            continue
        history.append({"template_index": ti, "actor_index": ai})

    return MandalaState(core=core, history=history)


# -- Action template -> wire format ----------------------------------------


def action_to_wire(template_index: int, state: MandalaState) -> dict[str, Any]:
    """Encode a chosen template index as the engine action JSON the site expects.

    Mandala has 4 wire shapes (matching mandala-web/game.js performAction):
      {"type": "build_mountain", "cardId": str, "mandalaIndex": int}
      {"type": "grow_field",     "cardIds": [str], "mandalaIndex": int}
      {"type": "discard_redraw", "cardIds": [str]}
      {"type": "claim_color",    "color": str}

    Card-id resolution requires the bot's actual current hand contents,
    which is why this function takes the state as well as the template.
    """
    template: Template = index_to_template(template_index)
    return template_to_engine_action(template, state.core)
