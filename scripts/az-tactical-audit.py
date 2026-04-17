"""Adversarial tactical audit: set up positions where the correct move is
obvious, ask the live AZ, and see what it returns.

Specifically, 'AZ has a cat in pool and 3-in-a-row win available' —
the exact pattern you described. We construct a minimal position that
isolates that decision, send it to the live service, and verify.
"""

from __future__ import annotations

import json
import sys

import requests

URL = "https://mlfactory-az-production.up.railway.app/move"


def empty_state(to_play: str = "gray") -> dict:
    """Empty 6x6 board, both players have 8 kittens / 0 cats / 0 retired."""
    return {
        "board": [[None] * 6 for _ in range(6)],
        "players": {
            "orange": {
                "color": "orange",
                "kittensInPool": 8,
                "catsInPool": 0,
                "kittensRetired": 0,
                "socketId": "ox",
                "name": "Orange",
                "connected": True,
            },
            "gray": {
                "color": "gray",
                "kittensInPool": 8,
                "catsInPool": 0,
                "kittensRetired": 0,
                "socketId": "gx",
                "name": "Gray",
                "connected": True,
            },
        },
        "currentTurn": to_play,
        "phase": "playing",
        "winner": None,
        "lastMove": None,
        "boopedPieces": [],
        "graduatedPieces": [],
    }


def call_move(state: dict, color: str, sims: int = 200) -> dict:
    r = requests.post(URL, json={"state": state, "color": color, "sims": sims}, timeout=30)
    r.raise_for_status()
    return r.json()


def case_1_obvious_cat_horizontal_win() -> bool:
    """Gray has two cats at (3,1) and (3,2), a cat in pool, and the 3rd
    empty cell at (3,0) and (3,3) — either wins instantly. Expected: AZ
    places a cat at (3,0) or (3,3).
    """
    s = empty_state("gray")
    # Gray has 2 cats on the board, 1 in pool, 0 retired, 5 kittens (say 5)
    s["players"]["gray"]["catsInPool"] = 1
    s["players"]["gray"]["kittensInPool"] = 5  # doesn't matter exactly
    # Also need total "pieces on board + pool + retired" to add up sensibly;
    # game rules don't block us here but the AZ's net sees pool planes.
    s["board"][3][1] = {"color": "gray", "type": "cat"}
    s["board"][3][2] = {"color": "gray", "type": "cat"}
    # Put a few orange pieces so it isn't obviously move 3-of-game
    s["board"][1][1] = {"color": "orange", "type": "kitten"}
    s["board"][1][3] = {"color": "orange", "type": "kitten"}
    s["players"]["orange"]["kittensInPool"] = 6
    print("\n=== case 1: gray has 2 cats at (3,1),(3,2); cat in pool. ===")
    print_board(s)
    resp = call_move(s, "gray")
    print(f"  AZ played: {resp}")
    ok = (
        resp["kind"] == "place"
        and resp["pieceType"] == "cat"
        and resp["row"] == 3
        and resp["col"] in (0, 3)
    )
    print(f"  PASS" if ok else f"  FAIL: expected cat at (3,0) or (3,3)")
    return ok


def case_2_cat_diagonal_win() -> bool:
    """Gray has cats at (1,1) and (2,2); placing cat at (3,3) or (0,0) wins.
    Verifies diagonals work. Plus we use the cat-in-pool from 'retired/
    pool=1' to make sure the net sees a cat is placeable.
    """
    s = empty_state("gray")
    s["players"]["gray"]["catsInPool"] = 1
    s["players"]["gray"]["kittensInPool"] = 5
    s["board"][1][1] = {"color": "gray", "type": "cat"}
    s["board"][2][2] = {"color": "gray", "type": "cat"}
    s["board"][0][3] = {"color": "orange", "type": "kitten"}
    s["board"][4][2] = {"color": "orange", "type": "kitten"}
    s["players"]["orange"]["kittensInPool"] = 6
    print("\n=== case 2: gray cats at (1,1),(2,2); cat in pool. ===")
    print_board(s)
    resp = call_move(s, "gray")
    print(f"  AZ played: {resp}")
    ok = (
        resp["kind"] == "place"
        and resp["pieceType"] == "cat"
        and ((resp["row"] == 3 and resp["col"] == 3) or (resp["row"] == 0 and resp["col"] == 0))
    )
    print(f"  PASS" if ok else f"  FAIL: expected cat at (3,3) or (0,0)")
    return ok


def case_3_kitten_only_pool_cant_graduate_instantly() -> bool:
    """Just a sanity case: if gray has NO cats in pool, they physically
    can't play a cat. AZ should play a kitten. Not a test of the bug,
    but a control."""
    s = empty_state("gray")
    s["players"]["gray"]["catsInPool"] = 0
    s["players"]["gray"]["kittensInPool"] = 7
    s["board"][3][1] = {"color": "gray", "type": "kitten"}
    s["board"][3][2] = {"color": "gray", "type": "kitten"}
    print("\n=== case 3 (control): gray kittens, no cats in pool. ===")
    print_board(s)
    resp = call_move(s, "gray")
    print(f"  AZ played: {resp}")
    ok = resp["kind"] == "place" and resp["pieceType"] == "kitten"
    print(f"  PASS (kitten as expected)" if ok else f"  FAIL: net should have played kitten")
    return ok


def case_4_orange_cat_win_avail() -> bool:
    """Same as case 1 but with roles swapped: orange has a cat win.
    Verifies both colors work."""
    s = empty_state("orange")
    s["players"]["orange"]["catsInPool"] = 1
    s["players"]["orange"]["kittensInPool"] = 5
    s["board"][3][1] = {"color": "orange", "type": "cat"}
    s["board"][3][2] = {"color": "orange", "type": "cat"}
    s["board"][1][1] = {"color": "gray", "type": "kitten"}
    s["board"][1][3] = {"color": "gray", "type": "kitten"}
    s["players"]["gray"]["kittensInPool"] = 6
    print("\n=== case 4: orange has 2 cats at (3,1),(3,2); cat in pool. ===")
    print_board(s)
    resp = call_move(s, "orange")
    print(f"  AZ played: {resp}")
    ok = (
        resp["kind"] == "place"
        and resp["pieceType"] == "cat"
        and resp["row"] == 3
        and resp["col"] in (0, 3)
    )
    print(f"  PASS" if ok else f"  FAIL: expected cat at (3,0) or (3,3)")
    return ok


def print_board(state: dict) -> None:
    glyphs = {
        ("orange", "kitten"): "o",
        ("orange", "cat"): "O",
        ("gray", "kitten"): "g",
        ("gray", "cat"): "G",
    }
    print("  0 1 2 3 4 5")
    for r in range(6):
        cells = []
        for c in range(6):
            p = state["board"][r][c]
            cells.append(glyphs[(p["color"], p["type"])] if p else ".")
        print(f"{r} " + " ".join(cells))
    op = state["players"]["orange"]
    gp = state["players"]["gray"]
    print(
        f"pools: orange k{op['kittensInPool']} c{op['catsInPool']} | "
        f"gray k{gp['kittensInPool']} c{gp['catsInPool']}"
    )
    print(f"turn: {state['currentTurn']}")


if __name__ == "__main__":
    results = [
        case_1_obvious_cat_horizontal_win(),
        case_2_cat_diagonal_win(),
        case_3_kitten_only_pool_cant_graduate_instantly(),
        case_4_orange_cat_win_avail(),
    ]
    n_pass = sum(results)
    print(f"\n==== {n_pass}/{len(results)} tactical audits passed ====")
    sys.exit(0 if n_pass == len(results) else 1)
