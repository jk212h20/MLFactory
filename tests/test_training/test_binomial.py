"""Tests for the one-sided binomial p-value helper used in early-stop.

Pinning the math here is important because the whole point of the rule
is statistical honesty: false-positive rate must match advertised alpha.
"""

from mlfactory.training.trainer import _binomial_p_value_one_sided as pvalue


def test_zero_games_returns_one():
    assert pvalue(0, 0, 0) == 1.0


def test_equal_split_returns_one():
    """5-5-0 is exactly the null. p should be 1.0 (no evidence)."""
    assert pvalue(5, 0, 5) == 1.0
    assert pvalue(10, 0, 10) == 1.0
    assert pvalue(20, 0, 20) == 1.0


def test_at_or_below_half_returns_one():
    """Score <= 0.5 should never trigger a one-sided 'better than parity'
    rejection."""
    assert pvalue(3, 0, 7) == 1.0
    # 9-2-9 -> 10/20 score, exactly half -> p=1
    assert pvalue(9, 2, 9) == 1.0


def test_pure_sweep_gives_tiny_pvalue():
    """40-0-0 should give astronomical evidence."""
    p = pvalue(40, 0, 0)
    assert p < 1e-10, p


def test_known_examples():
    """Reference values from a manual calculation; if these change something
    is wrong with the helper."""
    # 15/20 wins -> P(X>=15 | n=20, p=0.5) = (sum C(20,15..20)) * 2^-20
    # = (15504+5760+1140+190+20+1)/1048576 ≈ 0.0207
    p = pvalue(15, 0, 5)
    assert 0.020 < p < 0.022, p

    # 27/40 -> ~0.019
    p = pvalue(27, 0, 13)
    assert 0.018 < p < 0.020, p


def test_run51_iter15_result_does_not_trigger():
    """The actual outcome that misled run51's old fixed-threshold rule:
    12 wins, 5 losses, 3 draws. Score = 13.5/20 = 67.5%. Looks impressive
    in win-rate terms but is well within parity-noise.

    We round half-points DOWN (conservatively), so 13.5 -> 13.
    P(X >= 13 | n=20, p=0.5) ≈ 0.13. That MUST be > 0.05 alpha or our
    'evidence-based stop' is no better than the old rule."""
    p = pvalue(12, 3, 5)
    assert p > 0.05, p
    assert p < 0.20, p  # but not absurdly high either


def test_draws_count_as_half_points():
    """0 wins / 20 draws / 0 losses -> 10/20 score -> exactly null."""
    assert pvalue(0, 20, 0) == 1.0


def test_all_draws_with_some_wins():
    """8-12-0: score=8/20=0.4. Should be 1.0 (below null).
    Then 8 wins + 12 draws + 0 losses: score=14/20=0.7 -> should reject.
    """
    assert pvalue(8, 0, 12) == 1.0
    p = pvalue(8, 12, 0)
    # score=14, P(X>=14 | 20, .5) ≈ 0.058
    assert 0.05 < p < 0.06, p


def test_threshold_calibration_for_typical_run():
    """For a realistic stop config (--baseline-ckpt-games 40, alpha 0.05),
    the win threshold below which we don't stop should be 26 (since 26
    gives p=0.040 and 25 gives p=0.077)."""
    assert pvalue(26, 0, 14) <= 0.05
    assert pvalue(25, 0, 15) > 0.05
