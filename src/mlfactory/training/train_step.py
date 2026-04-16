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

    return TrainLosses(
        policy=float(policy_loss.item()),
        value=float(value_loss.item()),
        total=float(total.item()),
    )


def mean_losses(losses: list[TrainLosses]) -> TrainLosses:
    if not losses:
        return TrainLosses(0.0, 0.0, 0.0)
    return TrainLosses(
        policy=sum(l.policy for l in losses) / len(losses),
        value=sum(l.value for l in losses) / len(losses),
        total=sum(l.total for l in losses) / len(losses),
    )
