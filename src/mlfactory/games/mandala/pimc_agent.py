"""PIMC (Perfect Information Monte Carlo) agent for Mandala.

Standard technique for imperfect-information games. At each move:

  1. Sample N consistent completions of the hidden state. For Mandala
     this means: opponent's hand, opponent's starting cup cards, and
     deck order — each sampled from the residual color pool.
  2. For each completion, run a normal (perfect-info) PUCT search.
  3. Aggregate visit counts across all completions.
  4. Pick the most-visited action.

This is "root PIMC": determinization happens only at the root of each
search, not at every tree node. Simpler than full PIMC, known to work
well for moderate hidden-state sizes (Bridge etc).

Why this should help on Mandala:
- Our previous PUCT searched the actual current state (which has
  hidden_color "?" cards in the deck). The evaluator + PUCT had no
  way to reason about what those cards are. The value backups were
  noisy averages.
- PIMC turns each search into a perfect-info problem, so the value
  signal is clean per-determinization. The aggregation picks moves
  that are good across many possible opponent hands.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from mlfactory.agents.alphazero.agent import AlphaZeroAgent
from mlfactory.agents.alphazero.evaluator import EvalResult
from mlfactory.agents.alphazero.puct import PUCTConfig, PUCTSearch, SearchResult
from mlfactory.games.mandala.actions import N_TEMPLATES
from mlfactory.games.mandala.env import MandalaEnv, MandalaState
from mlfactory.games.mandala.heuristic_evaluator import HeuristicPriorEvaluator
from mlfactory.training.mandala_value_smooth import _resample_hidden_state


class PIMCMandalaAgent:
    """Mandala agent that performs root-PIMC over an inner search.

    Per move:
      - Sample n_determinizations consistent hidden states from the
        public-info-implied residual pool.
      - For each, instantiate a fresh inner PUCT search and run sims_per_det
        simulations.
      - Aggregate visit counts across determinizations (sum, then normalize).
      - Pick argmax (greedy mode) or sample (sample mode).

    The inner search uses HeuristicPriorEvaluator by default — same prior
    we use for HP-PUCT. PIMC corrects the value signal by running each
    sim under a known hidden state.
    """

    def __init__(
        self,
        n_determinizations: int = 8,
        sims_per_det: int = 25,
        prior_temperature: float = 1.0,
        rollout_policy: str = "random",
        mode: str = "greedy",
        sample_temperature: float = 1.0,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        if mode not in ("greedy", "sample"):
            raise ValueError(f"mode must be 'greedy' or 'sample', got {mode}")
        self.n_determinizations = n_determinizations
        self.sims_per_det = sims_per_det
        self.prior_temperature = prior_temperature
        self.rollout_policy = rollout_policy
        self.mode = mode
        self.sample_temperature = sample_temperature
        self._rng = random.Random(seed)
        # Total search budget = n_dets * sims_per_det. The name reflects this
        # for clear comparison vs HP-PUCT(n_dets * sims_per_det).
        self.name = name or f"pimc{n_determinizations}x{sims_per_det}"
        self.last_search: SearchResult | None = None

    def reset(self) -> None:
        self.last_search = None

    def act(self, env: MandalaEnv, state: MandalaState) -> int:
        """Run root PIMC, return chosen action template index."""
        if state.is_terminal:
            raise ValueError("cannot act on terminal state")

        mover = state.to_play
        # Aggregate visits across determinizations.
        visits_total: dict[int, int] = defaultdict(int)
        # Track avg root_value across dets for the SearchResult we expose.
        root_values: list[float] = []

        for det_i in range(self.n_determinizations):
            # Sample a hidden completion for THIS determinization.
            sampled_core = _resample_hidden_state(state.core, mover, self._rng)
            sampled_state = MandalaState(core=sampled_core, history=list(state.history))

            # Build a fresh inner search for the determinized state.
            ev_seed = self._rng.randint(0, 2**31 - 1)
            inner_evaluator = HeuristicPriorEvaluator(
                env,
                prior_temperature=self.prior_temperature,
                rollout_policy=self.rollout_policy,
                rng_seed=ev_seed,
            )
            inner_cfg = PUCTConfig(n_simulations=self.sims_per_det)
            inner_rng = np.random.default_rng(ev_seed)
            inner_search = PUCTSearch(env, inner_evaluator, inner_cfg, rng=inner_rng)

            try:
                result = inner_search.search(sampled_state, add_root_noise=False)
            except ValueError:
                # Could happen if the sampled state turns out terminal —
                # very unlikely (we just sampled hidden cards) but be defensive.
                continue

            for action, n_visits in result.root_visits.items():
                visits_total[action] += n_visits
            root_values.append(result.root_value)

        if not visits_total:
            # Fallback: pick uniformly from legal actions.
            legal = env.legal_actions(state)
            if not legal:
                raise ValueError("no legal actions and no PIMC visits")
            return self._rng.choice(legal)

        # Build a SearchResult so downstream code (sample-game logging) works.
        legal_mask = np.zeros(N_TEMPLATES, dtype=bool)
        visits_arr = np.zeros(N_TEMPLATES, dtype=np.float32)
        root_q = {}
        for a, v in visits_total.items():
            visits_arr[a] = v
            legal_mask[a] = True
            # We don't track per-action Q across dets; expose 0 as a
            # placeholder. Could be improved by tracking value_sum/visits
            # per action across dets.
            root_q[a] = 0.0
        total_visits = visits_arr.sum()
        policy_target = (visits_arr / total_visits) if total_visits > 0 else visits_arr
        avg_root_value = float(np.mean(root_values)) if root_values else 0.0
        self.last_search = SearchResult(
            root_visits=dict(visits_total),
            root_q=root_q,
            root_value=avg_root_value,
            policy_target=policy_target,
            legal_mask=legal_mask,
            total_sims=self.n_determinizations * self.sims_per_det,
        )

        # Choose action.
        if self.mode == "greedy":
            return max(visits_total.items(), key=lambda kv: kv[1])[0]
        # Sample mode: visits^(1/temperature)
        actions = list(visits_total.keys())
        counts = np.array([visits_total[a] for a in actions], dtype=np.float64)
        if self.sample_temperature <= 1e-6:
            return max(visits_total.items(), key=lambda kv: kv[1])[0]
        if self.sample_temperature == 1.0:
            probs = counts / counts.sum()
        else:
            probs = np.power(counts, 1.0 / self.sample_temperature)
            probs /= probs.sum()
        idx = int(self._rng.choices(range(len(actions)), weights=probs.tolist())[0])
        return actions[idx]
