"""Tests for the FastAPI service using TestClient."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Point the service at a small test checkpoint before importing the app.
# We reuse the run50 iter-1 checkpoint which is already on disk and small.
_CKPT_CANDIDATES = [
    Path.cwd() / "experiments/boop/2026-04-16-222906-run50/checkpoints/iter-0001.pt",
]


def _find_checkpoint() -> Path | None:
    for p in _CKPT_CANDIDATES:
        if p.exists():
            return p
    # Fallback: any iter-*.pt under experiments/
    exp = Path.cwd() / "experiments"
    if exp.exists():
        cands = sorted(exp.rglob("iter-*.pt"))
        if cands:
            return cands[0]
    return None


@pytest.fixture(scope="module")
def client():
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint available for service tests")
    os.environ["AZ_CHECKPOINT"] = str(ckpt)
    os.environ["AZ_DEFAULT_SIMS"] = "8"  # fast for tests
    from mlfactory.service.app import app

    with TestClient(app) as c:
        yield c


def _empty_ts_state(current_turn: str = "orange") -> dict:
    return {
        "board": [[None] * 6 for _ in range(6)],
        "players": {
            "orange": {
                "color": "orange",
                "kittensInPool": 8,
                "catsInPool": 0,
                "kittensRetired": 0,
                "socketId": "x",
                "name": "P",
                "connected": True,
            },
            "gray": {
                "color": "gray",
                "kittensInPool": 8,
                "catsInPool": 0,
                "kittensRetired": 0,
                "socketId": "y",
                "name": "Q",
                "connected": True,
            },
        },
        "currentTurn": current_turn,
        "phase": "playing",
        "winner": None,
        "lastMove": None,
        "boopedPieces": [],
        "graduatedPieces": [],
    }


def test_health_reports_model_loaded(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["params"] > 0
    assert data["checkpoint"].endswith(".pt")


def test_move_initial_state_returns_placement(client: TestClient) -> None:
    r = client.post("/move", json={"state": _empty_ts_state(), "color": "orange"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["kind"] == "place"
    assert 0 <= data["row"] < 6
    assert 0 <= data["col"] < 6
    assert data["pieceType"] in ("kitten", "cat")
    assert data["sims"] > 0
    assert data["latency_ms"] >= 0


def test_move_rejects_wrong_color(client: TestClient) -> None:
    # State says orange to play, but we ask as gray.
    r = client.post("/move", json={"state": _empty_ts_state(), "color": "gray"})
    assert r.status_code == 400


def test_move_rejects_invalid_color(client: TestClient) -> None:
    r = client.post("/move", json={"state": _empty_ts_state(), "color": "purple"})
    assert r.status_code == 400


def test_move_rejects_finished_game(client: TestClient) -> None:
    state = _empty_ts_state()
    state["phase"] = "finished"
    state["winner"] = "orange"
    r = client.post("/move", json={"state": state, "color": "orange"})
    assert r.status_code == 400


def test_move_honours_per_request_sims(client: TestClient) -> None:
    # Small sims should return fast.
    r = client.post("/move", json={"state": _empty_ts_state(), "color": "orange", "sims": 4})
    assert r.status_code == 200
    assert r.json()["sims"] == 4
