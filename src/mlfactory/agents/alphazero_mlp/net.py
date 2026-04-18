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
    # Optional auxiliary heads. These force the shared trunk to learn
    # representations that capture information beyond what the value
    # head extracts on its own. Output is exposed only via
    # forward_with_aux(); the standard forward() still returns
    # (policy_logits, value) so existing evaluator/PUCT code is
    # unaffected.
    aux_opp_hand: bool = False  # predict opp hand color histogram (6 outputs)
    aux_opp_hand_bins: int = 9  # bins per color (0..MAX_HAND_SIZE inclusive = 9)


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

        # Optional opp-hand auxiliary head: predict, for each of 6 colors,
        # the count of that color in the opponent's hand as a categorical
        # over `aux_opp_hand_bins` bins (0..MAX_HAND_SIZE). 6 colors ×
        # bins = output size. Trained with cross-entropy on the actual
        # count from the engine state during supervised pretraining.
        if config.aux_opp_hand:
            from mlfactory.games.mandala.rules import COLORS

            self._aux_n_colors = len(COLORS)
            self.aux_opp_hand_head = nn.Linear(
                config.hidden, self._aux_n_colors * config.aux_opp_hand_bins
            )
        else:
            self._aux_n_colors = 0
            self.aux_opp_hand_head = None

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared trunk; returns the (B, hidden) representation
        that feeds all heads."""
        h = F.relu(self.input_ln(self.input_proj(x)))
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the net (policy + value heads only).

        x : (B, feature_dim) float
        Returns policy_logits (B, n_actions), value (B, 1) in [-1, 1].

        This signature is unchanged from the pre-aux net so existing
        evaluator/PUCT code works without modification.
        """
        h = self._trunk(x)
        policy_logits = self.policy_head(h)
        v = F.relu(self.value_ln(self.value_fc1(h)))
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value

    def forward_with_aux(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run the net with all heads.

        Returns (policy_logits, value, aux_opp_hand_logits).
        aux_opp_hand_logits has shape (B, n_colors, aux_opp_hand_bins)
        if aux_opp_hand was enabled in config, else None.

        Used by training code that wants to compute the auxiliary loss.
        Inference paths use the plain forward() and ignore the aux head.
        """
        h = self._trunk(x)
        policy_logits = self.policy_head(h)
        v = F.relu(self.value_ln(self.value_fc1(h)))
        value = torch.tanh(self.value_fc2(v))
        aux = None
        if self.aux_opp_hand_head is not None:
            aux_flat = self.aux_opp_hand_head(h)
            aux = aux_flat.view(-1, self._aux_n_colors, self.config.aux_opp_hand_bins)
        return policy_logits, value, aux

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
