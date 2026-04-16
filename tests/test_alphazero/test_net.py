"""Tests for the AlphaZero residual net."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mlfactory.agents.alphazero.net import AlphaZeroNet, NetConfig


def _tiny_config() -> NetConfig:
    # Minimal Boop-shaped config (4 blocks × 16 channels; ~15k params).
    return NetConfig(
        in_channels=11,
        board_h=6,
        board_w=6,
        n_actions=104,
        num_blocks=2,
        channels=16,
    )


def test_net_forward_shapes_cpu() -> None:
    cfg = _tiny_config()
    net = AlphaZeroNet(cfg).cpu().eval()
    x = torch.zeros(3, cfg.in_channels, cfg.board_h, cfg.board_w)
    with torch.no_grad():
        policy_logits, value = net(x)
    assert policy_logits.shape == (3, cfg.n_actions)
    assert value.shape == (3, 1)
    assert torch.isfinite(policy_logits).all()
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_net_forward_shapes_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    cfg = _tiny_config()
    net = AlphaZeroNet(cfg).to("mps").eval()
    x = torch.zeros(2, cfg.in_channels, cfg.board_h, cfg.board_w, device="mps")
    with torch.no_grad():
        policy_logits, value = net(x)
    assert policy_logits.shape == (2, cfg.n_actions)
    assert value.shape == (2, 1)


def test_net_param_count_reasonable() -> None:
    """The default Boop config should land around 100k-500k params."""
    cfg = NetConfig(in_channels=11, board_h=6, board_w=6, n_actions=104)
    net = AlphaZeroNet(cfg)
    n = net.param_count()
    # 4 blocks x 64 channels on 6x6 -> ~220k params (rough expectation).
    assert 100_000 < n < 500_000, f"unexpected param count: {n}"


def test_net_handles_batch_size_1() -> None:
    cfg = _tiny_config()
    net = AlphaZeroNet(cfg).eval()
    x = torch.zeros(1, cfg.in_channels, cfg.board_h, cfg.board_w)
    with torch.no_grad():
        pl, v = net(x)
    assert pl.shape == (1, cfg.n_actions)
    assert v.shape == (1, 1)


def test_net_training_mode_batchnorm() -> None:
    """BatchNorm needs batch > 1 in training mode; verify eval mode works with B=1."""
    cfg = _tiny_config()
    net = AlphaZeroNet(cfg)
    net.eval()
    x = torch.zeros(1, cfg.in_channels, cfg.board_h, cfg.board_w)
    # Should not raise.
    with torch.no_grad():
        net(x)


def test_net_save_and_load_roundtrip(tmp_path: Path) -> None:
    cfg = _tiny_config()
    net = AlphaZeroNet(cfg).eval()
    # Randomize a layer's weights so we're not comparing zeros to zeros.
    with torch.no_grad():
        net.policy_fc.weight.normal_(0.0, 0.1)

    x = torch.randn(2, cfg.in_channels, cfg.board_h, cfg.board_w)
    with torch.no_grad():
        pl_before, v_before = net(x)

    ckpt = tmp_path / "ckpt.pt"
    net.save(ckpt, extra={"iter": 7, "note": "hello"})

    loaded, extra = AlphaZeroNet.load(ckpt)
    loaded.eval()
    with torch.no_grad():
        pl_after, v_after = loaded(x)

    assert torch.allclose(pl_before, pl_after, atol=1e-6)
    assert torch.allclose(v_before, v_after, atol=1e-6)
    assert extra["iter"] == 7
    assert extra["note"] == "hello"
    assert loaded.config == cfg


def test_net_policy_logits_unmasked() -> None:
    """The net must NOT apply any legal-action masking — that's the evaluator's job."""
    cfg = _tiny_config()
    net = AlphaZeroNet(cfg).eval()
    x = torch.zeros(1, cfg.in_channels, cfg.board_h, cfg.board_w)
    with torch.no_grad():
        pl, _ = net(x)
    # All actions have some logit, nothing is -inf.
    assert torch.isfinite(pl).all()
