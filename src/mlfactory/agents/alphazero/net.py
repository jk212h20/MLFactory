"""Residual CNN for AlphaZero-lite.

Design:
- Input:    (B, C_in, H, W) — C_in, H, W game-specific (Boop: 11, 6, 6)
- Trunk:    Conv3x3 -> BN -> ReLU, then N residual blocks (Conv-BN-ReLU-Conv-BN + skip + ReLU).
- Policy:   Conv1x1 -> BN -> ReLU -> Flatten -> Linear -> logits over n_actions.
- Value:    Conv1x1 -> BN -> ReLU -> Flatten -> Linear -> ReLU -> Linear(1) -> tanh.

Default config: 4 blocks × 64 channels, Boop input shape -> ~220k params.
This is a deliberately small net; Boop is 6x6 and we want fast iterations
on the M4 Max. We can scale up later.

The net does NOT apply a legal-action mask. That's the evaluator's job
(wraps the net with state-specific masking). The net's policy output is
raw logits; callers must mask + softmax before using them as priors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NetConfig:
    """Hyperparameters that define the network shape.

    Saved with every checkpoint so weights can be reloaded without external
    config files.
    """

    in_channels: int  # game-specific (Boop: 11)
    board_h: int  # game-specific (Boop: 6)
    board_w: int  # game-specific (Boop: 6)
    n_actions: int  # game-specific (Boop: 104)
    num_blocks: int = 4
    channels: int = 64
    policy_channels: int = 2  # conv-1x1 output channels for policy head
    value_channels: int = 1  # conv-1x1 output channels for value head
    value_hidden: int = 64  # FC hidden dim inside value head


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class AlphaZeroNet(nn.Module):
    """Residual CNN with policy + value heads."""

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config

        # Stem
        self.stem_conv = nn.Conv2d(
            config.in_channels, config.channels, kernel_size=3, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(config.channels)

        # Trunk
        self.blocks = nn.ModuleList(
            [_ResidualBlock(config.channels) for _ in range(config.num_blocks)]
        )

        # Policy head: 1x1 conv -> flatten -> linear
        self.policy_conv = nn.Conv2d(
            config.channels, config.policy_channels, kernel_size=1, bias=False
        )
        self.policy_bn = nn.BatchNorm2d(config.policy_channels)
        self.policy_fc = nn.Linear(
            config.policy_channels * config.board_h * config.board_w, config.n_actions
        )

        # Value head: 1x1 conv -> flatten -> fc -> fc -> tanh
        self.value_conv = nn.Conv2d(
            config.channels, config.value_channels, kernel_size=1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(config.value_channels)
        self.value_fc1 = nn.Linear(
            config.value_channels * config.board_h * config.board_w, config.value_hidden
        )
        self.value_fc2 = nn.Linear(config.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the net.

        Returns
        -------
        policy_logits : (B, n_actions)  — unmasked, un-softmaxed logits
        value         : (B, 1)          — in [-1, 1] via tanh
        """
        h = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            h = block(h)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    # ------------------------------------------------------------------
    # Convenience: parameter count and checkpoint io
    # ------------------------------------------------------------------
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path, *, extra: dict | None = None) -> None:  # noqa: ANN001
        """Save config + weights (+ optional extra metadata) to `path`."""
        payload: dict = {
            "config": {
                "in_channels": self.config.in_channels,
                "board_h": self.config.board_h,
                "board_w": self.config.board_w,
                "n_actions": self.config.n_actions,
                "num_blocks": self.config.num_blocks,
                "channels": self.config.channels,
                "policy_channels": self.config.policy_channels,
                "value_channels": self.config.value_channels,
                "value_hidden": self.config.value_hidden,
            },
            "state_dict": self.state_dict(),
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load(cls, path, *, map_location="cpu") -> tuple["AlphaZeroNet", dict]:  # noqa: ANN001
        """Load a net from `path`. Returns (net, extra)."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg = NetConfig(**payload["config"])
        net = cls(cfg)
        net.load_state_dict(payload["state_dict"])
        return net, payload.get("extra", {})
