"""Python-side parity runner for Mandala. Mirrors parity_runner.js exactly.

Input/output format identical to the JS side so test harnesses can diff
them step-by-step. Both sides use the same mulberry32 seeded PRNG, so
shuffles and reshuffles produce identical deck orders.
"""

from __future__ import annotations

import json
import sys

from mlfactory.games.mandala._rng import Mulberry32
from mlfactory.games.mandala.rules import (
    create_game,
    get_valid_actions,
    get_winner,
    perform_action,
)


def action_objects_from_valid(valid: dict) -> list[dict]:
    """Flatten the getValidActions dict into a list of full action dicts.
    Order MUST match parity_runner.js exactly."""
    out: list[dict] = []
    for a in valid["buildMountain"]:
        out.append({"type": "build_mountain", **a})
    for a in valid["growField"]:
        out.append({"type": "grow_field", **a})
    for a in valid["discardRedraw"]:
        out.append({"type": "discard_redraw", **a})
    for color in valid["claimColor"]:
        out.append({"type": "claim_color", "color": color})
    return out


def main() -> None:
    payload = json.load(sys.stdin)
    seed = int(payload["seed"]) & 0xFFFFFFFF
    choices = list(payload["actionChoices"])

    rng = Mulberry32(seed)

    state = create_game("p0", "p1", rng=rng)

    steps = [{"turn": 0, "state": state}]

    for i, choice in enumerate(choices):
        if state["phase"] == "ended":
            break
        valid = get_valid_actions(state)
        flat = action_objects_from_valid(valid)
        if not flat:
            break
        idx = choice % len(flat)
        action = flat[idx]
        res = perform_action(state, action, rng=rng)
        if not res["success"]:
            steps.append({"turn": i + 1, "error": res["error"], "state": state})
            break
        state = res["newState"]
        steps.append({"turn": i + 1, "action": action, "state": state})

    winner = get_winner(state)
    sys.stdout.write(json.dumps({"steps": steps, "winner": winner}))


if __name__ == "__main__":
    main()
