"""Automatic game classifier for MLFactory.

Given a game's engine + a small probe interface, measure the dimensions
that determine which ML technique fits, and recommend a training pipeline.

This is the first stage of the "describe a game, get a bot" pipeline. A
human would manually inspect a game and reason about its properties; this
module does the same checks programmatically and produces a structured
report.

Inputs (the GameProbe protocol below):
- create_initial_state(seed) -> state
- legal_actions(state) -> list of action ids (or templates)
- step(state, action, rng) -> next_state
- is_terminal(state) -> bool
- winner(state) -> int | None
- player_view(state, player_index) -> visible state (for hidden-info games)
- a small heuristic agent function (optional but very useful)

Outputs:
- GameProfile dataclass with measured dimensions
- pretty-print to stdout for human inspection
- recommended pipeline (string identifier the framework chooser uses)

The dimensions and their downstream implications are documented inline.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol


# --- Game probe interface --------------------------------------------------


class GameProbe(Protocol):
    """The minimal interface a game must implement to be analyzed.

    A game should provide a probe that wraps its engine. The probe
    abstracts away game-specific types so the classifier can work
    uniformly across games.
    """

    name: str

    def create_initial(self, seed: int) -> Any:
        """Create a fresh initial state, with the RNG seeded by `seed`.

        The state is opaque to the classifier; only the methods below
        are called on it."""
        ...

    def legal_actions(self, state: Any) -> list:
        """Return the list of legal actions at `state`. Used to measure
        branching factor + action-space size."""
        ...

    def step(self, state: Any, action: Any, seed: int) -> Any:
        """Apply `action` to `state` and return next state. RNG seeded by
        `seed` for stochastic transitions."""
        ...

    def is_terminal(self, state: Any) -> bool: ...

    def winner(self, state: Any) -> int | None: ...

    def to_play(self, state: Any) -> int: ...

    def player_view(self, state: Any, player_index: int) -> Any:
        """Return the version of `state` visible to `player_index`. For
        perfect-info games returns the same state."""
        ...

    def state_size_bytes(self, state: Any) -> int:
        """Cheap estimate of the state representation size, used as a
        heuristic for whether to use spatial/structured nets."""
        ...

    def num_actions_total(self) -> int:
        """If actions are drawn from a fixed vocabulary, return the size.
        Otherwise return -1 (variable-length action space)."""
        ...

    # Optional — None if not provided.
    heuristic_agent: Callable[[Any], Any] | None


# --- Profile dataclass -----------------------------------------------------


@dataclass
class GameProfile:
    """Measured dimensions of a game, plus a recommended pipeline."""

    name: str
    # Discreteness / determinism
    is_perfect_info: bool
    is_deterministic: bool
    has_chance_nodes: bool

    # Action space
    action_space_size: int  # -1 if variable
    branching_factor_mean: float
    branching_factor_max: int
    branching_factor_std: float

    # Game length
    game_length_mean: float
    game_length_std: float
    game_length_min: int
    game_length_max: int

    # State
    state_bytes_estimate: int

    # Reward signal
    reward_density: str  # 'sparse_terminal' | 'dense_intermediate'

    # Hidden info (if imperfect_info)
    hidden_state_size_estimate: int

    # Heuristic gap
    heuristic_winrate_vs_random: float | None

    # Recommended pipeline + reasoning
    pipeline_recommendation: str
    pipeline_reasoning: list[str] = field(default_factory=list)

    # Raw measurement notes
    notes: dict = field(default_factory=dict)


# --- Classifier ------------------------------------------------------------


def classify(
    probe: GameProbe,
    n_random_games: int = 50,
    n_heuristic_games: int = 30,
    seed: int = 0,
) -> GameProfile:
    """Run the game through the classification probes and return a profile.

    All measurements use only the GameProbe interface; the classifier is
    game-agnostic.
    """
    rng = random.Random(seed)
    notes: dict[str, Any] = {}

    # --- 1. Perfect vs imperfect info -----------------------------------
    # Compare full state to player view at a fresh state. If they differ
    # in a way that drops information, it's imperfect info.
    s = probe.create_initial(seed)
    view0_state = probe.player_view(s, 0)
    full_norm = _normalize_for_compare(s)
    view0_norm = _normalize_for_compare(view0_state)
    is_perfect = full_norm == view0_norm
    notes["perfect_info_check"] = "full == view0" if is_perfect else "differs"

    # --- 2. Determinism + chance nodes ----------------------------------
    # Step from the same state with the same action twice using different
    # seeds. If results differ, the transition is stochastic.
    legal = probe.legal_actions(s)
    if legal:
        a = legal[0]
        s_a = probe.step(s, a, seed=1)
        s_b = probe.step(s, a, seed=2)
        is_deterministic = _normalize_for_compare(s_a) == _normalize_for_compare(s_b)
    else:
        is_deterministic = True
    has_chance = not is_deterministic
    notes["determinism_check"] = (
        "same outcome on identical action with different seeds"
        if is_deterministic
        else "differs across seeds — stochastic"
    )

    # --- 3. Action space + branching factor -----------------------------
    branchings = []
    game_lengths = []
    state_bytes = []
    for game_i in range(n_random_games):
        s = probe.create_initial(seed + game_i + 1)
        n = 0
        max_steps = 500
        while not probe.is_terminal(s) and n < max_steps:
            la = probe.legal_actions(s)
            if not la:
                break
            branchings.append(len(la))
            state_bytes.append(probe.state_size_bytes(s))
            a = rng.choice(la)
            s = probe.step(s, a, seed=rng.randint(0, 2**31 - 1))
            n += 1
        game_lengths.append(n)

    bf_mean = statistics.fmean(branchings) if branchings else 0
    bf_max = max(branchings) if branchings else 0
    bf_std = statistics.stdev(branchings) if len(branchings) > 1 else 0
    gl_mean = statistics.fmean(game_lengths) if game_lengths else 0
    gl_std = statistics.stdev(game_lengths) if len(game_lengths) > 1 else 0
    state_bytes_est = int(statistics.median(state_bytes)) if state_bytes else 0

    # --- 4. Reward density ----------------------------------------------
    # We assume sparse-terminal unless the engine exposes per-step rewards.
    # The probe interface doesn't have a per-step reward method (yet),
    # so we mark all games as sparse_terminal here. Future: probe could
    # expose per_step_reward(state, action, next_state).
    reward_density = "sparse_terminal"

    # --- 5. Hidden state size --------------------------------------------
    if is_perfect:
        hidden_state_size = 0
    else:
        # Cheap estimate: byte difference between full and view0 at the
        # initial state. Real games may have hidden state that grows over
        # turns (deck depleting); this is an underestimate for those.
        hidden_state_size = max(0, state_bytes_est - probe.state_size_bytes(view0_state))
    notes["hidden_state_estimate_method"] = "bytes(full) - bytes(view0) at initial state"

    # --- 6. Heuristic gap ------------------------------------------------
    heuristic_wr = None
    if probe.heuristic_agent is not None:
        wins = losses = draws = 0
        for game_i in range(n_heuristic_games):
            s = probe.create_initial(seed + 1000 + game_i)
            heuristic_first = game_i % 2 == 0
            n = 0
            while not probe.is_terminal(s) and n < 500:
                la = probe.legal_actions(s)
                if not la:
                    break
                if (probe.to_play(s) == 0) == heuristic_first:
                    a = probe.heuristic_agent(s)
                else:
                    a = rng.choice(la)
                s = probe.step(s, a, seed=rng.randint(0, 2**31 - 1))
                n += 1
            if not probe.is_terminal(s) or probe.winner(s) is None:
                draws += 1
            else:
                w = probe.winner(s)
                heur_won = (w == 0) == heuristic_first
                if heur_won:
                    wins += 1
                else:
                    losses += 1
        total = wins + losses + draws
        if total > 0:
            heuristic_wr = wins / total

    # --- 7. Recommended pipeline ----------------------------------------
    rec, reasoning = _recommend_pipeline(
        is_perfect=is_perfect,
        is_deterministic=is_deterministic,
        bf_mean=bf_mean,
        gl_mean=gl_mean,
        action_space_size=probe.num_actions_total(),
        hidden_state_size=hidden_state_size,
        state_bytes_est=state_bytes_est,
        heuristic_wr=heuristic_wr,
    )

    return GameProfile(
        name=probe.name,
        is_perfect_info=is_perfect,
        is_deterministic=is_deterministic,
        has_chance_nodes=has_chance,
        action_space_size=probe.num_actions_total(),
        branching_factor_mean=bf_mean,
        branching_factor_max=bf_max,
        branching_factor_std=bf_std,
        game_length_mean=gl_mean,
        game_length_std=gl_std,
        game_length_min=min(game_lengths) if game_lengths else 0,
        game_length_max=max(game_lengths) if game_lengths else 0,
        state_bytes_estimate=state_bytes_est,
        reward_density=reward_density,
        hidden_state_size_estimate=hidden_state_size,
        heuristic_winrate_vs_random=heuristic_wr,
        pipeline_recommendation=rec,
        pipeline_reasoning=reasoning,
        notes=notes,
    )


def _normalize_for_compare(s: Any) -> str:
    """Stringify a state for equality comparison; uses JSON for portability."""
    try:
        return json.dumps(s, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return repr(s)


def _recommend_pipeline(
    *,
    is_perfect: bool,
    is_deterministic: bool,
    bf_mean: float,
    gl_mean: float,
    action_space_size: int,
    hidden_state_size: int,
    state_bytes_est: int,
    heuristic_wr: float | None,
) -> tuple[str, list[str]]:
    """Apply the framework decision tree.

    Returns (pipeline_id, list_of_human_readable_reasons).
    """
    reasons: list[str] = []

    # Tier 1: perfect vs imperfect info.
    if is_perfect and is_deterministic:
        reasons.append("perfect-info + deterministic → standard AlphaZero applies")
        if state_bytes_est < 200:
            reasons.append("small non-spatial state → MLP-trunk net")
            arch = "az_mlp"
        else:
            reasons.append("larger state, possibly spatial → ResNet-trunk net")
            arch = "az_resnet"
    elif is_perfect and not is_deterministic:
        reasons.append("perfect-info + stochastic → AlphaZero with chance-node handling")
        arch = "az_mlp_chance"
    elif not is_perfect:
        reasons.append("imperfect-info → vanilla AlphaZero will struggle")
        if hidden_state_size < 50:
            reasons.append(
                f"low hidden-state size ({hidden_state_size} bytes) → "
                "PIMC (sample consistent completions per MCTS sim) is feasible"
            )
            arch = "az_pimc"
        else:
            reasons.append(
                f"high hidden-state size ({hidden_state_size} bytes) → "
                "consider counterfactual regret or deep-CFR; AlphaZero+PIMC may be too noisy"
            )
            arch = "deep_cfr_or_pimc"

    # Tier 2: bootstrap strategy.
    if heuristic_wr is None:
        reasons.append("no heuristic available → cold-start from random")
    elif heuristic_wr > 0.7:
        reasons.append(
            f"strong heuristic ({heuristic_wr:.0%} vs random) → "
            "bootstrap from heuristic-distillation"
        )
        arch += "+heuristic_bootstrap"
    elif heuristic_wr > 0.55:
        reasons.append(
            f"weak heuristic ({heuristic_wr:.0%} vs random) → "
            "may bootstrap or skip; not strongly differentiated from random"
        )
        arch += "+optional_bootstrap"
    else:
        reasons.append(
            f"heuristic ≈ random ({heuristic_wr:.0%}) → "
            "no benefit from heuristic bootstrap; train from scratch"
        )

    # Tier 3: branching factor / search budget.
    if bf_mean > 100:
        reasons.append(
            f"high branching factor (~{bf_mean:.0f}) → "
            "use top-k action filtering or progressive widening in PUCT"
        )
    if gl_mean > 100:
        reasons.append(
            f"long games (~{gl_mean:.0f} moves) → "
            "value targets are noisier; consider value-target smoothing or weighting"
        )

    return arch, reasons


def pretty_print(profile: GameProfile) -> str:
    """Human-readable formatted report."""
    lines = []
    lines.append(f"=== Game profile: {profile.name} ===")
    lines.append("")
    lines.append("Information structure:")
    lines.append(f"  perfect_info       : {profile.is_perfect_info}")
    lines.append(f"  deterministic      : {profile.is_deterministic}")
    if not profile.is_perfect_info:
        lines.append(f"  hidden_state_size  : ~{profile.hidden_state_size_estimate} bytes")
    lines.append("")
    lines.append("Action space:")
    if profile.action_space_size > 0:
        lines.append(f"  vocabulary size    : {profile.action_space_size}")
    else:
        lines.append("  vocabulary size    : variable / unbounded")
    lines.append(
        f"  branching factor   : mean={profile.branching_factor_mean:.1f} "
        f"max={profile.branching_factor_max} std={profile.branching_factor_std:.1f}"
    )
    lines.append("")
    lines.append("Game shape:")
    lines.append(
        f"  length             : mean={profile.game_length_mean:.0f} "
        f"std={profile.game_length_std:.0f} "
        f"range=[{profile.game_length_min}, {profile.game_length_max}]"
    )
    lines.append(f"  state size         : ~{profile.state_bytes_estimate} bytes")
    lines.append(f"  reward density     : {profile.reward_density}")
    lines.append("")
    lines.append("Heuristic baseline:")
    if profile.heuristic_winrate_vs_random is None:
        lines.append("  not provided")
    else:
        lines.append(f"  vs random          : {profile.heuristic_winrate_vs_random:.0%}")
    lines.append("")
    lines.append(f"Recommended pipeline: {profile.pipeline_recommendation}")
    lines.append("Reasoning:")
    for r in profile.pipeline_reasoning:
        lines.append(f"  - {r}")
    return "\n".join(lines)
