"""AlphaZero variant with an MLP trunk instead of a residual CNN.

Used for games with non-spatial state (like Mandala). The MLP is a
drop-in replacement for mlfactory.agents.alphazero.net.AlphaZeroNet:
same API (save, load, forward returning (policy_logits, value)), so the
same evaluator, PUCT search, agent, and training code can wrap it.
"""

from mlfactory.agents.alphazero_mlp.net import AlphaZeroMLP, MLPConfig

__all__ = ["AlphaZeroMLP", "MLPConfig"]
