"""Action encoding for Mandala.

The engine's action objects are variable-shape (build_mountain has a
cardId; grow_field has a list of cardIds; discard_redraw has a list of
cardIds; claim_color has a color). For the policy head we need a fixed-
size categorical. We use action **templates** keyed by the semantically
relevant fields only:

    (kind, mandala_index, color, count)
    where
      kind  in {0: build_mountain, 1: grow_field, 2: discard_redraw, 3: claim_color}
      mandala_index in {0, 1, None}  (None for discard_redraw and claim_color)
      color in {6 colors}
      count in {1..8}  (for discard/grow; 1 for build_mountain/claim_color)

Rationale: cards of the same colour in a hand are interchangeable for
purposes of legality and effect. The engine picks `cards.slice(0, count)`
to realize "play count cards of this color"; which specific card ids
those resolve to doesn't matter, so we don't need to expose the ids to
the policy.

Template index layout (fixed at 576 slots, of which a subset are legal
at any state):

    build_mountain:   kind=0, mandala in {0,1}, color in 6, count=1
                      -> 2 * 6 = 12 templates
    grow_field:       kind=1, mandala in {0,1}, color in 6, count in 1..7
                      -> 2 * 6 * 7 = 84 templates
                      (count max is hand_size-1 = 7, since you must keep >=1)
    discard_redraw:   kind=2, mandala=None, color in 6, count in 1..8
                      -> 6 * 8 = 48 templates
    claim_color:      kind=3, mandala=None, color in 6, count=1
                      -> 6 templates

Total: 12 + 84 + 48 + 6 = 150 distinct templates. Small enough for a
direct softmax head, big enough to preserve all relevant choice
granularity.
"""

from __future__ import annotations

from typing import NamedTuple

from mlfactory.games.mandala.rules import (
    COLORS,
    MAX_HAND_SIZE,
    get_valid_actions,
)


# -- Template index layout (see module docstring) ---------------------------

_N_COLORS = len(COLORS)
_BUILD_MANDALAS = 2
_GROW_MANDALAS = 2
_GROW_MAX_COUNT = MAX_HAND_SIZE - 1  # must keep >=1 in hand, so max play = hand-1 = 7
_DISCARD_MAX_COUNT = MAX_HAND_SIZE  # can discard entire hand

_BUILD_OFFSET = 0
_BUILD_N = _BUILD_MANDALAS * _N_COLORS  # 12

_GROW_OFFSET = _BUILD_OFFSET + _BUILD_N  # 12
_GROW_N = _GROW_MANDALAS * _N_COLORS * _GROW_MAX_COUNT  # 84

_DISCARD_OFFSET = _GROW_OFFSET + _GROW_N  # 96
_DISCARD_N = _N_COLORS * _DISCARD_MAX_COUNT  # 48

_CLAIM_OFFSET = _DISCARD_OFFSET + _DISCARD_N  # 144
_CLAIM_N = _N_COLORS  # 6

N_TEMPLATES = _CLAIM_OFFSET + _CLAIM_N  # 150

_COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}


class Template(NamedTuple):
    """Canonical description of an action choice from the policy head.

    `count` is 1 for build_mountain / claim_color (non-parametric on count)
    and in [1, max] for grow_field / discard_redraw. `mandala_index` is
    None for actions that don't target a mandala (discard / claim)."""

    kind: str  # 'build_mountain' | 'grow_field' | 'discard_redraw' | 'claim_color'
    mandala_index: int | None
    color: str
    count: int


# -- Index ↔ Template conversion --------------------------------------------


def template_to_index(t: Template) -> int:
    """Map a Template to its categorical slot in [0, N_TEMPLATES)."""
    c = _COLOR_TO_IDX[t.color]
    if t.kind == "build_mountain":
        assert t.mandala_index in (0, 1), t
        assert t.count == 1, t
        return _BUILD_OFFSET + t.mandala_index * _N_COLORS + c
    if t.kind == "grow_field":
        assert t.mandala_index in (0, 1), t
        assert 1 <= t.count <= _GROW_MAX_COUNT, t
        return (
            _GROW_OFFSET
            + t.mandala_index * _N_COLORS * _GROW_MAX_COUNT
            + c * _GROW_MAX_COUNT
            + (t.count - 1)
        )
    if t.kind == "discard_redraw":
        assert t.mandala_index is None, t
        assert 1 <= t.count <= _DISCARD_MAX_COUNT, t
        return _DISCARD_OFFSET + c * _DISCARD_MAX_COUNT + (t.count - 1)
    if t.kind == "claim_color":
        assert t.mandala_index is None, t
        assert t.count == 1, t
        return _CLAIM_OFFSET + c
    raise ValueError(f"unknown template kind: {t.kind}")


def index_to_template(i: int) -> Template:
    """Inverse of template_to_index."""
    if not 0 <= i < N_TEMPLATES:
        raise ValueError(f"index out of range: {i}")
    if i < _GROW_OFFSET:
        rel = i - _BUILD_OFFSET
        m, c = divmod(rel, _N_COLORS)
        return Template("build_mountain", m, COLORS[c], 1)
    if i < _DISCARD_OFFSET:
        rel = i - _GROW_OFFSET
        m, rel2 = divmod(rel, _N_COLORS * _GROW_MAX_COUNT)
        c, cnt = divmod(rel2, _GROW_MAX_COUNT)
        return Template("grow_field", m, COLORS[c], cnt + 1)
    if i < _CLAIM_OFFSET:
        rel = i - _DISCARD_OFFSET
        c, cnt = divmod(rel, _DISCARD_MAX_COUNT)
        return Template("discard_redraw", None, COLORS[c], cnt + 1)
    return Template("claim_color", None, COLORS[i - _CLAIM_OFFSET], 1)


# -- Template ↔ concrete engine action --------------------------------------


def template_from_engine_action(action: dict, state: dict) -> Template:
    """Map a concrete engine action back to its canonical Template.

    Used when recording self-play samples: we want to train the policy
    head on the template, not the raw engine action."""
    kind = action["type"]
    if kind == "build_mountain":
        player = state["players"][state["currentPlayerIndex"]]
        card = next(c for c in player["hand"] if c["id"] == action["cardId"])
        return Template("build_mountain", action["mandalaIndex"], card["color"], 1)
    if kind == "grow_field":
        player = state["players"][state["currentPlayerIndex"]]
        first_id = action["cardIds"][0]
        card = next(c for c in player["hand"] if c["id"] == first_id)
        return Template("grow_field", action["mandalaIndex"], card["color"], len(action["cardIds"]))
    if kind == "discard_redraw":
        player = state["players"][state["currentPlayerIndex"]]
        first_id = action["cardIds"][0]
        card = next(c for c in player["hand"] if c["id"] == first_id)
        return Template("discard_redraw", None, card["color"], len(action["cardIds"]))
    if kind == "claim_color":
        return Template("claim_color", None, action["color"], 1)
    raise ValueError(f"unknown action type: {kind}")


def template_to_engine_action(t: Template, state: dict) -> dict:
    """Materialise a Template into a concrete engine action using cards
    from the current state's hand.

    For grow_field / discard_redraw, picks the first `count` cards of the
    given colour from the hand (matching getValidActions' convention).
    For build_mountain, picks the first card of the given colour."""
    player = state["players"][state["currentPlayerIndex"]]
    if t.kind == "build_mountain":
        card = next(c for c in player["hand"] if c["color"] == t.color)
        return {"type": "build_mountain", "cardId": card["id"], "mandalaIndex": t.mandala_index}
    if t.kind == "grow_field":
        cards = [c for c in player["hand"] if c["color"] == t.color][: t.count]
        return {
            "type": "grow_field",
            "cardIds": [c["id"] for c in cards],
            "mandalaIndex": t.mandala_index,
        }
    if t.kind == "discard_redraw":
        cards = [c for c in player["hand"] if c["color"] == t.color][: t.count]
        return {"type": "discard_redraw", "cardIds": [c["id"] for c in cards]}
    if t.kind == "claim_color":
        return {"type": "claim_color", "color": t.color}
    raise ValueError(f"unknown template kind: {t.kind}")


# -- Legal-action mask over the template vocabulary -------------------------


def legal_template_indices(state: dict) -> list[int]:
    """Return the list of legal template indices at the current state."""
    if state["phase"] == "ended":
        return []
    valid = get_valid_actions(state)
    indices: set[int] = set()

    player = state["players"][state["currentPlayerIndex"]]
    # Quick color-in-hand lookup for inferring templates from the
    # id-based engine outputs.
    id_to_color = {c["id"]: c["color"] for c in player["hand"]}

    for a in valid["buildMountain"]:
        color = id_to_color[a["cardId"]]
        indices.add(template_to_index(Template("build_mountain", a["mandalaIndex"], color, 1)))
    for a in valid["growField"]:
        color = id_to_color[a["cardIds"][0]]
        indices.add(
            template_to_index(Template("grow_field", a["mandalaIndex"], color, len(a["cardIds"])))
        )
    for a in valid["discardRedraw"]:
        color = id_to_color[a["cardIds"][0]]
        indices.add(template_to_index(Template("discard_redraw", None, color, len(a["cardIds"]))))
    for color in valid["claimColor"]:
        indices.add(template_to_index(Template("claim_color", None, color, 1)))

    return sorted(indices)


def legal_mask(state: dict):
    """Return a boolean numpy mask of shape (N_TEMPLATES,), True where legal.
    Used by the evaluator to mask the net's raw policy logits."""
    import numpy as np

    mask = np.zeros(N_TEMPLATES, dtype=bool)
    for i in legal_template_indices(state):
        mask[i] = True
    return mask
