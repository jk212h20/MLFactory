"""Tests for the AlphaZero training step."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mlfactory.agents.alphazero.net import AlphaZeroNet, NetConfig
from mlfactory.training.train_step import mean_losses, train_step


def _tiny_net() -> AlphaZeroNet:
    cfg = NetConfig(
        in_channels=11,
        board_h=6,
        board_w=6,
        n_actions=104,
        num_blocks=1,
        channels=8,
    )
    return AlphaZeroNet(cfg)


def _synthetic_batch(batch_size: int, n_actions: int = 104):
    rng = np.random.default_rng(0)
    planes = rng.normal(size=(batch_size, 11, 6, 6)).astype(np.float32)
    # Random one-hot policy targets (simplest valid distribution)
    targets = rng.integers(0, n_actions, size=batch_size)
    policies = np.zeros((batch_size, n_actions), dtype=np.float32)
    for i, t in enumerate(targets):
        policies[i, t] = 1.0
    values = rng.choice([-1.0, 0.0, 1.0], size=batch_size).astype(np.float32)
    return planes, policies, values


def test_train_step_reduces_loss_on_repeating_batch() -> None:
    """Overfitting a single batch should drive total loss down."""
    net = _tiny_net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    planes, policies, values = _synthetic_batch(16)

    first = train_step(net, opt, planes, policies, values, device="cpu")
    for _ in range(30):
        train_step(net, opt, planes, policies, values, device="cpu")
    final = train_step(net, opt, planes, policies, values, device="cpu")

    # Policy + value losses should both decrease meaningfully.
    assert final.total < first.total * 0.6, (first, final)
    assert final.policy < first.policy * 0.8
    # (value loss can fluctuate more; just require it doesn't blow up)
    assert final.value < first.value * 1.5


def test_train_step_parameters_change() -> None:
    net = _tiny_net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    planes, policies, values = _synthetic_batch(8)
    before = [p.detach().clone() for p in net.parameters()]
    train_step(net, opt, planes, policies, values, device="cpu")
    after = [p.detach().clone() for p in net.parameters()]
    diffs = [((a - b).abs().sum().item()) for a, b in zip(before, after)]
    assert any(d > 0 for d in diffs), "no parameter changed — optimizer or grad issue"


def test_train_step_device_cpu_vs_mps_agree_shape() -> None:
    """Basic sanity that MPS execution works end to end (numerical match is not required)."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    net = _tiny_net().to("mps")
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    planes, policies, values = _synthetic_batch(8)
    losses = train_step(net, opt, planes, policies, values, device="mps")
    assert 0.0 <= losses.total < 1e6
    assert 0.0 <= losses.policy < 1e6
    assert 0.0 <= losses.value < 1e6


def test_mean_losses_handles_empty() -> None:
    m = mean_losses([])
    assert m.total == 0.0
    assert m.policy == 0.0
    assert m.value == 0.0


def test_train_step_reports_diagnostics() -> None:
    """policy_entropy, value_abs_mean, value_std should all be populated and sensible."""
    net = _tiny_net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    planes, policies, values = _synthetic_batch(16)
    losses = train_step(net, opt, planes, policies, values, device="cpu")

    # entropy is a real number between 0 and log(n_actions)
    import math

    assert 0.0 <= losses.policy_entropy <= math.log(104) + 1e-4
    # |tanh| in [0, 1]
    assert 0.0 <= losses.value_abs_mean <= 1.0
    # value_std is non-negative
    assert losses.value_std >= 0.0


def test_diagnostics_aggregate_via_mean_losses() -> None:
    """mean_losses averages the diagnostic fields too."""
    net = _tiny_net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    planes, policies, values = _synthetic_batch(8)
    a = train_step(net, opt, planes, policies, values, device="cpu")
    b = train_step(net, opt, planes, policies, values, device="cpu")
    m = mean_losses([a, b])
    assert abs(m.policy_entropy - (a.policy_entropy + b.policy_entropy) / 2) < 1e-6
    assert abs(m.value_abs_mean - (a.value_abs_mean + b.value_abs_mean) / 2) < 1e-6
    assert abs(m.value_std - (a.value_std + b.value_std) / 2) < 1e-6
