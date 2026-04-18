"""Game analysis: classify a game's structural dimensions and recommend
an ML pipeline. See game_classifier.py for the dimensions we measure
and the framework decision tree.
"""

from mlfactory.analysis.game_classifier import (
    GameProfile,
    GameProbe,
    classify,
    pretty_print,
)

__all__ = ["GameProfile", "GameProbe", "classify", "pretty_print"]
