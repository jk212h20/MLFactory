"""Evaluator: the oracle PUCT asks for priors + value at a state.

An evaluator encapsulates "whatever turns a game state into (prior, value)".
- `UniformEvaluator`: uniform priors over legal moves, value 0. PUCT with
  this evaluator should approximately reduce to MCTS-with-no-rollouts
  (pure exploration), useful as a sanity-test baseline.
- `NetEvaluator`: wraps an `AlphaZeroNet` and a game-specific encoder;
  masks illegal actions out of the policy before softmax.

Evaluators return **NumPy arrays** so PUCT stays pure Python + numpy and
doesn't need torch in its hot loop. Batch evaluation is via `evaluate_batch`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from mlfactory.core.env import Env, State


@dataclass(frozen=True)
class EvalResult:
    """Output of an evaluator for a single state.

    priors    : (n_actions,) float; zero on illegal actions, sums to 1.0 over
                legal actions. If there are zero legal actions (terminal),
                priors is an all-zero vector.
    value     : scalar in [-1, 1] — expected outcome from mover's perspective.
    """

    priors: np.ndarray
    value: float


class Evaluator(Protocol):
    """Turns a state into (priors, value)."""

    def evaluate(self, state: State) -> EvalResult: ...

    def evaluate_batch(self, states: list[State]) -> list[EvalResult]:
        """Batched version; default impl loops over `evaluate`."""
        ...


# ----------------------------------------------------------------------
# UniformEvaluator — no learning, just uniform priors over legal moves.
# ----------------------------------------------------------------------


class UniformEvaluator:
    """Uniform prior over legal actions, value = 0.

    PUCT with this evaluator is a pure exploration algorithm (no value signal,
    no policy signal). Useful as a baseline to sanity-check PUCT mechanics.
    """

    def __init__(self, env: Env) -> None:
        self.env = env
        self.n_actions = env.num_actions
        self.name = "uniform"

    def evaluate(self, state: State) -> EvalResult:
        priors = np.zeros(self.n_actions, dtype=np.float32)
        legal = self.env.legal_actions(state)
        if legal:
            priors[legal] = 1.0 / len(legal)
        return EvalResult(priors=priors, value=0.0)

    def evaluate_batch(self, states: list[State]) -> list[EvalResult]:
        return [self.evaluate(s) for s in states]


# ----------------------------------------------------------------------
# NetEvaluator — wraps an AlphaZeroNet.
# ----------------------------------------------------------------------


# Signature for game-specific encoders.  Takes (state) -> (planes, legal_mask).
#   planes:     np.ndarray of shape (C, H, W), float32
#   legal_mask: np.ndarray of shape (n_actions,), bool
EncoderFn = Callable[[State], tuple[np.ndarray, np.ndarray]]


class NetEvaluator:
    """Wraps an AlphaZeroNet + encoder + legal-action masker.

    The net is put in eval mode. Forward passes run in torch.no_grad(). The
    net and inputs are placed on `device` (`"cpu"`, `"mps"`, or `"cuda"`).
    """

    def __init__(
        self,
        net,  # AlphaZeroNet  (avoid import cycle at type-check time)
        encoder: EncoderFn,
        *,
        device: str = "cpu",
        name: str = "net",
    ) -> None:
        import torch

        self.net = net.to(device).eval()
        self.encoder = encoder
        self.device = device
        self.name = name
        self._torch = torch

    def evaluate(self, state: State) -> EvalResult:
        return self.evaluate_batch([state])[0]

    def evaluate_batch(self, states: list[State]) -> list[EvalResult]:
        torch = self._torch
        if not states:
            return []
        planes_list: list[np.ndarray] = []
        masks_list: list[np.ndarray] = []
        for s in states:
            planes, mask = self.encoder(s)
            planes_list.append(planes)
            masks_list.append(mask)

        batch = np.stack(planes_list, axis=0)
        x = torch.from_numpy(batch).to(self.device, non_blocking=True)
        with torch.no_grad():
            logits, value = self.net(x)
        logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        values_np = value.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)

        results: list[EvalResult] = []
        for i, mask in enumerate(masks_list):
            priors = _masked_softmax(logits_np[i], mask)
            results.append(EvalResult(priors=priors, value=float(values_np[i])))
        return results


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax restricted to legal actions.

    Zeros out illegal actions, renormalises. If there are no legal actions
    (terminal state), returns an all-zero vector.
    """
    legal = mask.astype(bool)
    if not legal.any():
        return np.zeros_like(logits, dtype=np.float32)
    masked = np.where(legal, logits, -np.inf)
    # Numerically stable softmax
    shifted = masked - masked[legal].max()
    exp = np.where(legal, np.exp(shifted), 0.0)
    total = exp.sum()
    if total <= 0.0:
        # All logits were -inf on legal actions -> uniform fallback
        out = np.zeros_like(logits, dtype=np.float32)
        out[legal] = 1.0 / legal.sum()
        return out
    return (exp / total).astype(np.float32)
