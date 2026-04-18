"""Heuristic-prior evaluator for PUCT — no neural net, no training.

Tests whether wrapping the heuristic in PUCT improves over the heuristic
alone. If it does, we know the bottleneck in the AZ pipeline is the net
(the policy prior the net produces is worse than the heuristic's).

Two variants:
- HeuristicSoftEvaluator: priors = softmax of per-action heuristic
  scores. Value from a rollout.
- HeuristicSharpEvaluator: priors place most mass on the heuristic's
  top action, small mass on the rest. Closer to "imitate the
  heuristic".

Value comes from a single random-rollout to terminal (classic MCTS
flavor) or, optionally, from a heuristic-rollout (Bridge/Splendor
flavor — better signal but slower).
"""

from __future__ import annotations

import random

import numpy as np

from mlfactory.agents.alphazero.evaluator import EvalResult
from mlfactory.games.mandala.actions import (
    N_TEMPLATES,
    index_to_template,
    legal_template_indices,
)
from mlfactory.games.mandala.env import MandalaEnv, MandalaState
from mlfactory.games.mandala.heuristic_agent import HeuristicMandalaAgent


class HeuristicPriorEvaluator:
    """Evaluator that uses the rule-based heuristic for both prior and value.

    Prior: softmax over heuristic scores for legal actions (so the
    heuristic's preferences shape PUCT's exploration). A `temperature`
    knob sharpens or flattens — temperature=1 is "as scored", lower is
    sharper.

    Value: one rollout from the leaf using either random or heuristic
    policy until terminal, returning ±1/0 from the leaf-mover's
    perspective.
    """

    def __init__(
        self,
        env: MandalaEnv,
        *,
        prior_temperature: float = 1.0,
        rollout_policy: str = "heuristic",  # "heuristic" or "random"
        rollout_max_moves: int = 200,
        rng_seed: int | None = None,
        name: str = "heur",
    ) -> None:
        if rollout_policy not in ("heuristic", "random"):
            raise ValueError(
                f"rollout_policy must be 'heuristic' or 'random', got {rollout_policy}"
            )
        self.env = env
        self.prior_temperature = prior_temperature
        self.rollout_policy = rollout_policy
        self.rollout_max_moves = rollout_max_moves
        self.name = name
        # A scoring heuristic instance, reused across calls. The heuristic
        # is stateless beyond its rng (used only for tie-breaking).
        self._scorer = HeuristicMandalaAgent(seed=rng_seed if rng_seed is not None else 0)
        # Separate rng for rollouts so it's independent of the scorer.
        self._rng = random.Random(rng_seed)

    # PUCT calls evaluate(state) one at a time inside _simulate.
    def evaluate(self, state: MandalaState) -> EvalResult:
        legal = legal_template_indices(state.core)
        priors = np.zeros(N_TEMPLATES, dtype=np.float32)

        if not legal:
            # Terminal-ish (zero-legal edge case). PUCT will treat this
            # branch as terminal anyway.
            return EvalResult(priors=priors, value=0.0)

        # Score every legal template via the heuristic's internal scorer.
        # We reach into the heuristic's per-template scoring helper rather
        # than calling .act() (which would just pick one); we want all the
        # scores so we can softmax them into a prior distribution.
        scores = np.full(len(legal), -1e9, dtype=np.float32)
        for i, t_idx in enumerate(legal):
            template = index_to_template(t_idx)
            scores[i] = self._scorer._score_template(state, template)

        # Numerically-stable softmax over scored legal actions.
        s = scores - scores.max()
        # Apply temperature: lower = sharper.
        s = s / max(self.prior_temperature, 1e-6)
        e = np.exp(s)
        e_sum = e.sum()
        if e_sum > 0:
            probs = e / e_sum
        else:
            probs = np.full(len(legal), 1.0 / len(legal), dtype=np.float32)
        for i, t_idx in enumerate(legal):
            priors[t_idx] = probs[i]

        # Value: one rollout from `state` from the perspective of the
        # current mover.
        value = self._rollout_value(state)
        return EvalResult(priors=priors, value=value)

    def evaluate_batch(self, states: list[MandalaState]) -> list[EvalResult]:
        return [self.evaluate(s) for s in states]

    # ------------------------------------------------------------------

    def _rollout_value(self, state: MandalaState) -> float:
        """Play out from `state` until terminal; return ±1/0 for the
        mover at `state`."""
        mover = state.to_play
        cur = state
        n = 0
        # Use heuristic agents for the rollout if requested. Cheap to
        # build because the heuristic is stateless beyond rng.
        h0 = HeuristicMandalaAgent(seed=self._rng.randint(0, 2**31 - 1))
        h1 = HeuristicMandalaAgent(seed=self._rng.randint(0, 2**31 - 1))
        while not cur.is_terminal and n < self.rollout_max_moves:
            legal = self.env.legal_actions(cur)
            if not legal:
                break
            if self.rollout_policy == "heuristic":
                action = (h0 if cur.to_play == 0 else h1).act(self.env, cur)
            else:
                action = self._rng.choice(legal)
            cur = self.env.step(cur, action)
            n += 1
        if not cur.is_terminal or cur.winner is None:
            return 0.0
        return 1.0 if cur.winner == mover else -1.0
