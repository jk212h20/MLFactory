"""End-to-end smoke: local Boop server talking to local AZ service.

Preconditions (both must be running locally before this script):
  1. FastAPI AZ service on http://127.0.0.1:8765
       AZ_CHECKPOINT=experiments/.../iter-0050.pt \\
       AZ_DEFAULT_SIMS=100 \\
       uv run uvicorn mlfactory.service.app:app --port 8765

  2. Boop server in dev mode on http://127.0.0.1:3001, with
       AZ_SERVICE_URL=http://127.0.0.1:8765 npm run dev:server

This script joins as a fake human, plays random legal moves, and
confirms the AZ responds and the game terminates with a winner.

Run:
    uv run python scripts/boop-az-smoke.py
"""

from __future__ import annotations

import asyncio
import random
import sys
from typing import Any

import socketio

BOOP_URL = "http://127.0.0.1:3001"


async def main() -> int:
    sio = socketio.AsyncClient(logger=False, engineio_logger=False)
    rng = random.Random(42)

    game_state: dict[str, Any] = {}
    my_color: str = ""
    az_color: str = ""
    game_over = asyncio.Event()
    winner: str | None = None

    @sio.on("connect")
    async def on_connect() -> None:
        print("[smoke] connected to boop server")

    @sio.on("game_update")
    async def on_game_update(payload: dict[str, Any]) -> None:
        nonlocal game_state
        game_state = payload["gameState"]
        # Concise summary for the log.
        p = game_state["phase"]
        t = game_state.get("currentTurn")
        last = payload.get("lastMove")
        print(f"[smoke] game_update: phase={p} turn={t} lastMove={last}")

    @sio.on("game_over")
    async def on_game_over(payload: dict[str, Any]) -> None:
        nonlocal winner
        winner = payload["winner"]
        print(f"[smoke] game_over: winner={winner}  cond={payload.get('winCondition')}")
        game_over.set()

    @sio.on("az_error")
    async def on_az_error(payload: dict[str, Any]) -> None:
        print(f"[smoke] !! az_error: {payload.get('error')}")
        game_over.set()

    await sio.connect(BOOP_URL, transports=["websocket"])

    # Create an AZ game with us as orange.
    resp = await sio.call(
        "create_az_game",
        {"playerName": "Smoke", "humanColor": "orange"},
        timeout=10,
    )
    if not resp.get("success"):
        print(f"[smoke] create_az_game failed: {resp}")
        await sio.disconnect()
        return 2
    my_color = resp["playerColor"]
    az_color = resp["azColor"]
    game_state = resp["gameState"]
    print(f"[smoke] room={resp['roomCode']} me={my_color} az={az_color}")

    ply = 0
    while not game_over.is_set() and ply < 200:
        # Let the server process prior events.
        await asyncio.sleep(0.05)
        if game_state.get("phase") == "finished":
            break

        # If it's our turn, play a random legal move.
        if game_state.get("phase") == "playing" and game_state.get("currentTurn") == my_color:
            action = pick_random_legal_move(game_state, my_color, rng)
            if action is None:
                print("[smoke] no legal move for human — draw by no-moves? aborting")
                break
            ack = await sio.call("place_piece", action, timeout=10)
            if not ack.get("success"):
                print(f"[smoke] place_piece rejected: {ack}")
                break
            ply += 1
            # Wait a tick to let game_update arrive + AZ pump run.
            await asyncio.sleep(0.1)
            continue

        if (
            game_state.get("phase") == "selecting_graduation"
            and game_state.get("pendingGraduationPlayer") == my_color
        ):
            options = game_state.get("pendingGraduationOptions") or []
            if not options:
                print("[smoke] selecting_graduation but no options; aborting")
                break
            idx = rng.randrange(len(options))
            ack = await sio.call("select_graduation", {"optionIndex": idx}, timeout=10)
            if not ack.get("success"):
                print(f"[smoke] select_graduation rejected: {ack}")
                break
            await asyncio.sleep(0.1)
            continue

        # Not our turn (AZ is moving). Wait.
        await asyncio.sleep(0.1)

    try:
        await asyncio.wait_for(game_over.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    await sio.disconnect()
    if winner is None:
        print("[smoke] FAIL: game did not finish")
        return 3
    print(f"[smoke] OK: winner={winner} plies={ply}")
    return 0


def pick_random_legal_move(
    state: dict[str, Any], color: str, rng: random.Random
) -> dict[str, Any] | None:
    """Pick any legal place move for `color`. Simple heuristic: any empty
    cell, preferring whichever piece type has inventory. We don't run the
    full boop-legality check here — the server will reject illegal moves."""
    players = state.get("players", {})
    me = players.get(color) or {}
    kittens = int(me.get("kittensInPool", 0))
    cats = int(me.get("catsInPool", 0))
    board = state["board"]
    empty: list[tuple[int, int]] = []
    for r in range(6):
        for c in range(6):
            if board[r][c] is None:
                empty.append((r, c))
    if not empty:
        return None
    rng.shuffle(empty)
    # Prefer kittens (always legal if kittens > 0).
    piece_type = "kitten" if kittens > 0 else ("cat" if cats > 0 else None)
    if piece_type is None:
        return None
    r, c = empty[0]
    return {"row": r, "col": c, "pieceType": piece_type}


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
