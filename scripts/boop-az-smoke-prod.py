"""Production smoke: hit the deployed Boop server, play one full game as a
fake human against the deployed AZ service, confirm we get a winner.

No local services required — this talks to Railway.
"""

from __future__ import annotations

import asyncio
import random
import sys
from typing import Any

import socketio

BOOP_URL = "https://boop-production-6b80.up.railway.app"


async def main() -> int:
    sio = socketio.AsyncClient(logger=False, engineio_logger=False)
    rng = random.Random(42)
    game_state: dict[str, Any] = {}
    game_over = asyncio.Event()
    winner: str | None = None

    @sio.on("connect")
    async def on_connect() -> None:
        print(f"[smoke-prod] connected to {BOOP_URL}")

    @sio.on("game_update")
    async def on_game_update(payload: dict[str, Any]) -> None:
        nonlocal game_state
        game_state = payload["gameState"]

    @sio.on("game_over")
    async def on_game_over(payload: dict[str, Any]) -> None:
        nonlocal winner
        winner = payload["winner"]
        game_over.set()

    @sio.on("az_error")
    async def on_az_error(payload: dict[str, Any]) -> None:
        print(f"[smoke-prod] az_error: {payload.get('error')}")
        game_over.set()

    await sio.connect(BOOP_URL, transports=["websocket"])
    resp = await sio.call(
        "create_az_game", {"playerName": "SmokeProd", "humanColor": "orange"}, timeout=20
    )
    if not resp.get("success"):
        print(f"[smoke-prod] create_az_game failed: {resp}")
        await sio.disconnect()
        return 2
    my_color = resp["playerColor"]
    game_state = resp["gameState"]
    print(f"[smoke-prod] room={resp['roomCode']} me={my_color} az={resp['azColor']}")

    plies = 0
    while not game_over.is_set() and plies < 200:
        await asyncio.sleep(0.1)
        if game_state.get("phase") == "finished":
            break
        if game_state.get("phase") == "playing" and game_state.get("currentTurn") == my_color:
            move = pick_random(game_state, my_color, rng)
            if move is None:
                break
            ack = await sio.call("place_piece", move, timeout=20)
            if not ack.get("success"):
                print(f"[smoke-prod] place_piece rejected: {ack}")
                break
            plies += 1
            await asyncio.sleep(0.2)
            continue
        if (
            game_state.get("phase") == "selecting_graduation"
            and game_state.get("pendingGraduationPlayer") == my_color
        ):
            opts = game_state.get("pendingGraduationOptions") or []
            if not opts:
                break
            await sio.call(
                "select_graduation", {"optionIndex": rng.randrange(len(opts))}, timeout=20
            )
            await asyncio.sleep(0.2)
            continue
        await asyncio.sleep(0.1)

    try:
        await asyncio.wait_for(game_over.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        pass
    await sio.disconnect()

    if winner is None:
        print("[smoke-prod] FAIL: no winner")
        return 3
    print(f"[smoke-prod] OK: winner={winner} plies={plies}")
    return 0


def pick_random(state: dict[str, Any], color: str, rng: random.Random) -> dict[str, Any] | None:
    me = state.get("players", {}).get(color) or {}
    kittens = int(me.get("kittensInPool", 0))
    cats = int(me.get("catsInPool", 0))
    board = state["board"]
    empty = [(r, c) for r in range(6) for c in range(6) if board[r][c] is None]
    if not empty:
        return None
    rng.shuffle(empty)
    piece = "kitten" if kittens > 0 else ("cat" if cats > 0 else None)
    if piece is None:
        return None
    r, c = empty[0]
    return {"row": r, "col": c, "pieceType": piece}


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
