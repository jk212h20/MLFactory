"""FastAPI app serving an AlphaZero checkpoint over HTTP.

Startup: loads the checkpoint path from the AZ_CHECKPOINT env var (or
falls back to a baked-in default). Applies the single-threaded torch
config that our wiki/insights/2026-04-16-mps-cpu-crossover-small-nets.md
showed is needed for batch-1 serving.

Endpoints:

    GET  /health              ->  liveness + model metadata
    POST /move                ->  choose a move for a given state
    POST /admin/reload        ->  swap in a new checkpoint at runtime
                                  (optional, useful for deploys without restart)

The `POST /move` request body:

    {
      "state": <boop GameState JSON from Boop TS server>,
      "color": "orange" | "gray"   # which side the AZ bot is playing
    }

The response:

    {"kind": "place", "row": 3, "col": 4, "pieceType": "kitten"}
    OR
    {"kind": "graduation", "optionIndex": 0}

Errors are returned as HTTP 4xx/5xx with JSON body `{"error": "..."}`.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

# These MUST be set before torch itself does any intra-op work, so we do
# it at import time.
import torch

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlfactory.agents.alphazero.agent import AlphaZeroAgent
from mlfactory.agents.alphazero.evaluator import NetEvaluator
from mlfactory.agents.alphazero.net import AlphaZeroNet
from mlfactory.agents.alphazero.puct import PUCTConfig
from mlfactory.games.boop import Boop
from mlfactory.games.boop.encode import encode_state, legal_mask
from mlfactory.service.boop_adapter import action_to_wire, parse_boop_state

log = logging.getLogger("mlfactory.service")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


# -- Request / response shapes ----------------------------------------------


class MoveRequest(BaseModel):
    """Inbound request: current game state + which colour the AZ is playing."""

    state: dict[str, Any] = Field(..., description="Boop TS GameState JSON")
    color: str = Field(..., description="Colour the AZ plays: 'orange' or 'gray'")
    # Optional per-request overrides:
    sims: int | None = Field(None, description="Override PUCT sims (default from config)")
    seed: int | None = Field(None, description="Override RNG seed")


class MoveResponse(BaseModel):
    kind: str
    # Fields populated depending on kind:
    row: int | None = None
    col: int | None = None
    pieceType: str | None = None
    optionIndex: int | None = None
    # Diagnostics:
    root_value: float | None = None
    sims: int
    latency_ms: int


# -- App ---------------------------------------------------------------------


app = FastAPI(title="MLFactory AZ service", version="1.0")


# Loaded lazily in startup event. Kept in module-level state for simplicity
# since this service is single-process single-model by design.
_state: dict[str, Any] = {
    "checkpoint_path": None,
    "net": None,
    "env": None,
    "default_sims": 200,
}


def _resolve_default_checkpoint() -> Path:
    """Locate a default checkpoint. Prefer AZ_CHECKPOINT env; otherwise
    pick the newest iter-*.pt file under experiments/."""
    env_path = os.environ.get("AZ_CHECKPOINT")
    if env_path:
        p = Path(env_path)
        if not p.exists():
            raise FileNotFoundError(f"AZ_CHECKPOINT does not exist: {p}")
        return p
    # Fall back to newest checkpoint found under experiments/.
    # This is useful for local dev but SHOULD NOT be relied on in deploy.
    exp_root = Path(__file__).resolve().parents[3] / "experiments"
    candidates = sorted(exp_root.rglob("iter-*.pt"), reverse=True)
    if not candidates:
        raise RuntimeError(
            "No AZ_CHECKPOINT env set and no iter-*.pt found under experiments/. "
            "Set AZ_CHECKPOINT=/path/to/iter-0050.pt"
        )
    log.warning("No AZ_CHECKPOINT set; using newest available checkpoint: %s", candidates[0])
    return candidates[0]


def _load_net(path: Path) -> tuple[AlphaZeroNet, dict]:
    net, extra = AlphaZeroNet.load(path)
    net = net.cpu().eval()
    return net, extra


@app.on_event("startup")
def _startup() -> None:
    path = _resolve_default_checkpoint()
    net, extra = _load_net(path)
    env = Boop()
    _state["checkpoint_path"] = str(path)
    _state["net"] = net
    _state["extra"] = extra
    _state["env"] = env
    _state["default_sims"] = int(os.environ.get("AZ_DEFAULT_SIMS", "200"))
    log.info(
        "Loaded checkpoint %s (params=%d, extra=%s, default_sims=%d)",
        path,
        net.param_count(),
        extra,
        _state["default_sims"],
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness + model metadata."""
    net: AlphaZeroNet | None = _state.get("net")
    return {
        "status": "ok" if net is not None else "starting",
        "checkpoint": _state.get("checkpoint_path"),
        "params": net.param_count() if net is not None else None,
        "extra": _state.get("extra"),
        "default_sims": _state.get("default_sims"),
    }


def _encoder(state):
    return encode_state(state), legal_mask(state)


@app.post("/move", response_model=MoveResponse)
def move(req: MoveRequest) -> MoveResponse:
    """Choose a move for the AZ playing `color` given the current state."""
    net: AlphaZeroNet | None = _state.get("net")
    env: Boop | None = _state.get("env")
    if net is None or env is None:
        raise HTTPException(503, "service not ready")

    # Validate colour
    if req.color not in ("orange", "gray"):
        raise HTTPException(400, "color must be 'orange' or 'gray'")

    # Parse state
    try:
        boop_state = parse_boop_state(req.state)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"failed to parse state: {e}") from e

    # Sanity check: it's actually the AZ's turn.
    requested_side = 0 if req.color == "orange" else 1
    if boop_state.to_play != requested_side:
        raise HTTPException(
            400,
            f"not {req.color}'s turn — state.currentTurn indicates opposite side "
            f"(to_play={boop_state.to_play})",
        )
    if boop_state.is_terminal:
        raise HTTPException(400, "game is already finished")

    sims = req.sims or _state["default_sims"]
    seed = req.seed if req.seed is not None else int(time.time() * 1000) & 0x7FFFFFFF

    # Build the agent (fresh per-request is fine; PUCT state is not reusable).
    evaluator = NetEvaluator(net, encoder=_encoder, device="cpu", name="az-service")
    agent = AlphaZeroAgent(
        evaluator,
        PUCTConfig(n_simulations=sims),
        mode="sample",
        temperature=1.0,
        temperature_moves=4,  # mild variety in openings; greedy afterwards
        add_root_noise=False,
        seed=seed,
        name="az-service",
    )

    t0 = time.monotonic()
    try:
        action = agent.act(env, boop_state)
    except Exception as e:  # noqa: BLE001
        log.exception("PUCT search failed")
        raise HTTPException(500, f"search failed: {e}") from e
    latency_ms = int((time.monotonic() - t0) * 1000)

    wire = action_to_wire(action)
    root_value = None
    if agent.last_search is not None:
        root_value = float(agent.last_search.root_value)

    log.info(
        "move color=%s phase=%s sims=%d latency=%dms action=%d wire=%s rv=%s",
        req.color,
        boop_state.phase,
        sims,
        latency_ms,
        action,
        wire,
        root_value,
    )

    return MoveResponse(
        kind=wire["kind"],
        row=wire.get("row"),
        col=wire.get("col"),
        pieceType=wire.get("pieceType"),
        optionIndex=wire.get("optionIndex"),
        root_value=root_value,
        sims=sims,
        latency_ms=latency_ms,
    )


class ReloadRequest(BaseModel):
    checkpoint: str


@app.post("/admin/reload")
def admin_reload(req: ReloadRequest) -> dict[str, Any]:
    """Swap the served checkpoint at runtime. No auth — only expose on
    internal network."""
    path = Path(req.checkpoint)
    if not path.exists():
        raise HTTPException(404, f"checkpoint not found: {path}")
    net, extra = _load_net(path)
    _state["net"] = net
    _state["checkpoint_path"] = str(path)
    _state["extra"] = extra
    log.info("reloaded checkpoint: %s", path)
    return {"status": "reloaded", "checkpoint": str(path)}
