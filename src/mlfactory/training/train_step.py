"""One training step: forward, loss, backward, step. No side effects besides
the net's own weights and optimizer state.

Loss: MSE on value + cross-entropy on policy + L2 regularisation (via weight decay).

Policy loss: -sum(pi_target * log_softmax(pi_logits)). Only over legal moves
is implicit because policy_target has zero mass on illegal actions.

Value loss: MSE between tanh(v_head) and z (the terminal-result target).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mlfactory.agents.alphazero.net import AlphaZeroNet


@dataclass(frozen=True)
class TrainLosses:
    policy: float
    value: float
    total: float
    # Diagnostics (computed with no_grad; do not affect training):
    #   policy_entropy: mean entropy of softmax(policy_logits) in nats.
    #       For reference: uniform over K actions has entropy log(K);
    #       for Boop with ~20 legal moves that's ~3.0. As the net gets
    #       more confident, this should trend down.
    #   value_abs_mean: mean |tanh(value)| in [0, 1]. A net predicting 0
    #       everywhere has value_abs_mean = 0; a confident net
    #       approaches 1 on decided positions.
    #   value_std: std of value predictions across the batch. High spread
    #       = the net is differentiating positions; flat = all predictions
    #       are similar regardless of input.
    policy_entropy: float = 0.0
    value_abs_mean: float = 0.0
    value_std: float = 0.0


def train_step(
    net: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    planes: np.ndarray,
    policy_targets: np.ndarray,
    value_targets: np.ndarray,
    *,
    device: str,
) -> TrainLosses:
    """Run one minibatch: forward, loss, backward, step. Returns losses as floats."""
    net.train()
    planes_t = torch.from_numpy(planes).to(device, non_blocking=True)
    policy_t = torch.from_numpy(policy_targets).to(device, non_blocking=True)
    value_t = torch.from_numpy(value_targets).to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    policy_logits, value_pred = net(planes_t)

    # Policy loss: categorical cross-entropy against soft targets.
    # F.log_softmax for numerical stability; sum over actions, mean over batch.
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_t * log_probs).sum(dim=1).mean()

    # Value loss: MSE. value_pred has shape (B, 1); squeeze to match.
    value_loss = F.mse_loss(value_pred.squeeze(-1), value_t)

    total = policy_loss + value_loss
    total.backward()
    optimizer.step()

    # Diagnostics (outside grad; cheap).
    with torch.no_grad():
        probs = torch.exp(log_probs)
        # H = -sum(p * log p), clamped to avoid 0*(-inf).
        entropy = -(probs * torch.clamp(log_probs, min=-30.0)).sum(dim=1).mean()
        v_abs = value_pred.abs().mean()
        v_std = value_pred.detach().std(unbiased=False)

    return TrainLosses(
        policy=float(policy_loss.item()),
        value=float(value_loss.item()),
        total=float(total.item()),
        policy_entropy=float(entropy.item()),
        value_abs_mean=float(v_abs.item()),
        value_std=float(v_std.item()),
    )


def mean_losses(losses: list[TrainLosses]) -> TrainLosses:
    if not losses:
        return TrainLosses(0.0, 0.0, 0.0)
    n = len(losses)
    return TrainLosses(
        policy=sum(l.policy for l in losses) / n,
        value=sum(l.value for l in losses) / n,
        total=sum(l.total for l in losses) / n,
        policy_entropy=sum(l.policy_entropy for l in losses) / n,
        value_abs_mean=sum(l.value_abs_mean for l in losses) / n,
        value_std=sum(l.value_std for l in losses) / n,
    )
