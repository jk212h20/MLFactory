"""AlphaZero-style MLP: feature vector -> (policy logits, value).

Mirrors the API of mlfactory.agents.alphazero.net.AlphaZeroNet so the
existing evaluator, PUCT, and training code can use it unchanged:
- forward(x) -> (policy_logits, value)
- save(path, extra=...) / load(path) -> (net, extra)
- param_count() -> int
- config attribute carrying the hyperparameters

Net shape: feature_dim -> hidden -> hidden -> ... -> policy/value heads,
with residual connections (simple MLP residual blocks) to help training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MLPConfig:
    """Hyperparameters for AlphaZeroMLP.

    Saved with every checkpoint so weights can be reloaded without
    external config files.
    """

    feature_dim: int  # size of the flat input feature vector
    n_actions: int  # size of the policy head's output (template vocabulary)
    hidden: int = 256  # width of each trunk layer
    n_blocks: int = 4  # number of residual blocks in the trunk
    value_hidden: int = 128  # width of the value head's hidden layer


class _ResBlock(nn.Module):
    """MLP residual block: Linear -> LayerNorm -> ReLU -> Linear -> LN -> +skip -> ReLU."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.ln1 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.ln1(self.fc1(x)))
        h = self.ln2(self.fc2(h))
        return F.relu(h + x)


class AlphaZeroMLP(nn.Module):
    """MLP with policy + value heads."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        # Input projection: feature_dim -> hidden
        self.input_proj = nn.Linear(config.feature_dim, config.hidden)
        self.input_ln = nn.LayerNorm(config.hidden)

        # Residual trunk
        self.blocks = nn.ModuleList([_ResBlock(config.hidden) for _ in range(config.n_blocks)])

        # Policy head: hidden -> n_actions (raw logits; caller masks + softmaxes)
        self.policy_head = nn.Linear(config.hidden, config.n_actions)

        # Value head: hidden -> value_hidden -> 1 -> tanh
        self.value_fc1 = nn.Linear(config.hidden, config.value_hidden)
        self.value_ln = nn.LayerNorm(config.value_hidden)
        self.value_fc2 = nn.Linear(config.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the net.

        x : (B, feature_dim) float
        Returns policy_logits (B, n_actions), value (B, 1) in [-1, 1].
        """
        h = F.relu(self.input_ln(self.input_proj(x)))
        for block in self.blocks:
            h = block(h)
        policy_logits = self.policy_head(h)
        v = F.relu(self.value_ln(self.value_fc1(h)))
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value

    # ------------------------------------------------------------------
    # Convenience: parameter count + checkpoint io
    # ------------------------------------------------------------------
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path, *, extra: dict | None = None) -> None:  # noqa: ANN001
        """Save config + weights (+ optional extra metadata) to `path`."""
        payload: dict = {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "net_kind": "alphazero_mlp",
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load(cls, path, *, map_location="cpu") -> tuple["AlphaZeroMLP", dict]:  # noqa: ANN001
        """Load a net from `path`. Returns (net, extra)."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg = MLPConfig(**payload["config"])
        net = cls(cfg)
        net.load_state_dict(payload["state_dict"])
        return net, payload.get("extra", {})
