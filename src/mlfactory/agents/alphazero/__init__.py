"""AlphaZero-lite: small residual CNN + PUCT search for MLFactory games.

Modules:
- net.py:       the residual network (policy + value heads)
- evaluator.py: wraps a net + game encoder as a prior/value oracle for PUCT
- puct.py:      PUCT Monte Carlo tree search (AlphaZero-style, single-threaded)
- agent.py:     AlphaZeroAgent: Agent protocol implementation using PUCT + net
"""

from mlfactory.agents.alphazero.agent import AlphaZeroAgent
from mlfactory.agents.alphazero.evaluator import NetEvaluator, UniformEvaluator
from mlfactory.agents.alphazero.net import AlphaZeroNet, NetConfig
from mlfactory.agents.alphazero.puct import PUCTSearch, PUCTConfig, SearchResult

__all__ = [
    "AlphaZeroAgent",
    "AlphaZeroNet",
    "NetConfig",
    "NetEvaluator",
    "PUCTConfig",
    "PUCTSearch",
    "SearchResult",
    "UniformEvaluator",
]
