"""FastAPI app serving a trained Mandala AlphaZero MLP checkpoint over HTTP.

Sibling to `service/app.py` (which serves the Boop AZ checkpoint). This
module mirrors that file's structure as closely as possible — the differences
are scoped to:
  - net class:    AlphaZeroMLP (mandala) vs AlphaZeroNet (boop convnet)
  - env / state:  MandalaEnv / MandalaState (dict-based) vs Boop / BoopState
  - encoder:      MandalaEncoderClosure (player-view + history)
  - actions:      150-template vocabulary, materialised against the bot's hand
  - wire format:  {type, cardId|cardIds|color, mandalaIndex} per game.js

Startup loads the checkpoint pointed to by AZ_CHECKPOINT (or the newest
mandala-*.pt under deploy/checkpoints as a fallback). Default agent mode is
the raw-net argmax-over-legal-actions policy: zero search, ~ms-scale
inference, deterministic. PUCT and PIMC modes are wired but disabled by
default (toggle via AZ_AGENT_MODE env var) so we can ship the simplest
thing first and tune from there.

Endpoints:

    GET  /health              ->  liveness + model metadata
    POST /move                ->  choose a move for a given state
    POST /admin/reload        ->  swap in a new checkpoint at runtime

The `POST /move` request body:

    {
      "state": <mandala-web getPlayerView JSON>,
      "playerIndex": 0 | 1,    # which side the bot is playing
      "history": [             # optional, for richer encoder features
        {"templateIndex": 12, "actorIndex": 0},
        ...
      ],
      "sims": 200,             # optional override (ignored in raw-net mode)
      "seed": 12345            # optional override
    }

The response:

    {
      "action": {"type": "build_mountain", "cardId": "red-3", "mandalaIndex": 0},
      "templateIndex": 4,
      "value": 0.31,           # net's value estimate (raw-net) or root value (puct)
      "latency_ms": 12,
      "mode": "raw_net"
    }

Errors are returned as HTTP 4xx/5xx with FastAPI's default `{"detail": "..."}`.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

# Single-thread torch BEFORE any forward pass — same rationale as
# service/app.py: batch-1 latency on small nets is dominated by thread
# overhead. See wiki/insights/2026-04-16-mps-cpu-crossover-small-nets.md.
import torch

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlfactory.agents.alphazero_mlp import AlphaZeroMLP
from mlfactory.games.mandala.actions import legal_mask
from mlfactory.games.mandala.encode import encode_view
from mlfactory.games.mandala.env import MandalaEnv
from mlfactory.games.mandala.rules import get_player_view
from mlfactory.service.mandala_adapter import action_to_wire, parse_mandala_state

log = logging.getLogger("mlfactory.service.mandala")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


# -- Request / response shapes ----------------------------------------------


class MoveRequest(BaseModel):
    """Inbound: full game state (from the bot's perspective) + which seat
    the bot occupies, plus optional history and overrides."""

    state: dict[str, Any] = Field(..., description="mandala-web getPlayerView JSON")
    playerIndex: int = Field(..., description="Which seat the bot plays: 0 or 1")
    history: list[dict[str, Any]] | None = Field(
        None,
        description="Optional action history for encoder features. List of "
        "{templateIndex, actorIndex} dicts in chronological order.",
    )
    sims: int | None = Field(None, description="Override PUCT sims (ignored in raw_net mode)")
    seed: int | None = Field(None, description="Override RNG seed")


class MoveResponse(BaseModel):
    action: dict[str, Any]
    templateIndex: int
    value: float | None = None
    latency_ms: int
    mode: str


# -- App ---------------------------------------------------------------------


app = FastAPI(title="MLFactory Mandala AZ service", version="1.0")


# Module-level state — single-process single-model by design (same as
# service/app.py).
_state: dict[str, Any] = {
    "checkpoint_path": None,
    "net": None,
    "env": None,
    "extra": None,
    "default_sims": 200,
    "agent_mode": "raw_net",
}


def _resolve_default_checkpoint() -> Path:
    """Locate a default checkpoint. Prefer AZ_CHECKPOINT env; otherwise
    pick the newest mandala-*.pt file under deploy/checkpoints/, then
    under experiments/."""
    env_path = os.environ.get("AZ_CHECKPOINT")
    if env_path:
        p = Path(env_path)
        if not p.exists():
            raise FileNotFoundError(f"AZ_CHECKPOINT does not exist: {p}")
        return p
    # Try deploy/checkpoints first (production-ish location).
    repo_root = Path(__file__).resolve().parents[3]
    for search_root, pattern in (
        (repo_root / "deploy" / "checkpoints", "mandala-*.pt"),
        (repo_root / "experiments", "mandala-*.pt"),
        (repo_root / "experiments", "iter-*.pt"),
    ):
        if not search_root.exists():
            continue
        candidates = sorted(search_root.rglob(pattern), reverse=True)
        if candidates:
            log.warning(
                "No AZ_CHECKPOINT set; using newest available checkpoint: %s",
                candidates[0],
            )
            return candidates[0]
    raise RuntimeError(
        "No AZ_CHECKPOINT env set and no mandala-*.pt found under deploy/checkpoints "
        "or experiments/. Set AZ_CHECKPOINT=/path/to/mandala-*.pt"
    )


def _load_net(path: Path) -> tuple[AlphaZeroMLP, dict]:
    net, extra = AlphaZeroMLP.load(path)
    net = net.cpu().eval()
    return net, extra


@app.on_event("startup")
def _startup() -> None:
    path = _resolve_default_checkpoint()
    net, extra = _load_net(path)
    env = MandalaEnv()
    _state["checkpoint_path"] = str(path)
    _state["net"] = net
    _state["extra"] = extra
    _state["env"] = env
    _state["default_sims"] = int(os.environ.get("AZ_DEFAULT_SIMS", "200"))
    _state["agent_mode"] = os.environ.get("AZ_AGENT_MODE", "raw_net")
    log.info(
        "Loaded mandala checkpoint %s (params=%d, extra=%s, default_sims=%d, mode=%s)",
        path,
        net.param_count(),
        extra,
        _state["default_sims"],
        _state["agent_mode"],
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness + model metadata."""
    net: AlphaZeroMLP | None = _state.get("net")
    return {
        "status": "ok" if net is not None else "starting",
        "checkpoint": _state.get("checkpoint_path"),
        "params": net.param_count() if net is not None else None,
        "extra": _state.get("extra"),
        "default_sims": _state.get("default_sims"),
        "agent_mode": _state.get("agent_mode"),
    }


# -- Inference --------------------------------------------------------------


def _select_action_raw_net(
    net: AlphaZeroMLP,
    state,
    bot_index: int,
) -> tuple[int, float]:
    """Run the trained net on the encoded view; return (template_index, value).

    Picks the argmax over legal templates of the policy logits. No search.
    Latency is dominated by feature encoding + a single forward pass —
    typically <10ms on CPU for our 1405-dim input / 256-wide / 4-block MLP.
    """
    view = get_player_view(state.core, bot_index)
    feats = encode_view(view, bot_index, state.history)
    mask = legal_mask(state.core)
    if not mask.any():
        raise ValueError("no legal actions at this state")

    x = torch.from_numpy(feats[None, :]).float()
    with torch.no_grad():
        logits, value = net(x)
    logits_np = logits[0].cpu().numpy()
    value_np = float(value[0].item())

    # Mask illegal actions to -inf, then argmax. Defensive: if no legal
    # action received a finite logit (numerically unlikely), fall back to
    # the first legal template.
    masked = np.where(mask.astype(bool), logits_np, -np.inf)
    if not np.isfinite(masked).any():
        legal_idx = np.where(mask)[0]
        return int(legal_idx[0]), value_np
    return int(masked.argmax()), value_np


@app.post("/move", response_model=MoveResponse)
def move(req: MoveRequest) -> MoveResponse:
    """Choose a move for the bot playing seat `playerIndex` given the state."""
    net: AlphaZeroMLP | None = _state.get("net")
    if net is None:
        raise HTTPException(503, "service not ready")

    if req.playerIndex not in (0, 1):
        raise HTTPException(400, "playerIndex must be 0 or 1")

    # Mash the optional history field into the state payload so the
    # adapter sees one combined dict.
    payload = dict(req.state)
    if req.history is not None:
        payload["history"] = req.history

    try:
        mandala_state = parse_mandala_state(payload)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"failed to parse state: {e}") from e

    # Sanity: it's actually the bot's turn (either normal play or claim phase).
    core = mandala_state.core
    is_play_turn = core["phase"] == "playing" and core["currentPlayerIndex"] == req.playerIndex
    is_claim_turn = (
        core["phase"] == "destroying"
        and core.get("destruction") is not None
        and core["destruction"].get("currentClaimerIndex") == req.playerIndex
    )
    if not (is_play_turn or is_claim_turn):
        raise HTTPException(
            400,
            f"not bot's turn — phase={core['phase']}, "
            f"currentPlayerIndex={core['currentPlayerIndex']}, "
            f"requestedPlayerIndex={req.playerIndex}",
        )
    if mandala_state.is_terminal:
        raise HTTPException(400, "game is already finished")

    mode = _state["agent_mode"]
    t0 = time.monotonic()
    try:
        if mode == "raw_net":
            template_index, value = _select_action_raw_net(net, mandala_state, req.playerIndex)
        else:
            # Future: "puct" or "pimc_net". Wired here so toggling the env
            # var doesn't require a service restart redesign.
            raise HTTPException(500, f"agent mode '{mode}' not implemented yet")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("inference failed")
        raise HTTPException(500, f"inference failed: {e}") from e
    latency_ms = int((time.monotonic() - t0) * 1000)

    try:
        wire = action_to_wire(template_index, mandala_state)
    except Exception as e:  # noqa: BLE001
        log.exception("action materialisation failed")
        raise HTTPException(500, f"action materialisation failed: {e}") from e

    log.info(
        "move seat=%d phase=%s turn=%d template=%d wire=%s value=%.3f "
        "latency=%dms mode=%s history_len=%d",
        req.playerIndex,
        core["phase"],
        core.get("turnNumber", -1),
        template_index,
        wire,
        value,
        latency_ms,
        mode,
        len(mandala_state.history),
    )

    return MoveResponse(
        action=wire,
        templateIndex=template_index,
        value=value,
        latency_ms=latency_ms,
        mode=mode,
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
    log.info("reloaded mandala checkpoint: %s", path)
    return {"status": "reloaded", "checkpoint": str(path)}
