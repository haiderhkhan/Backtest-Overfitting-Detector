"""DSR sanity checks on synthetic series with known properties."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from overfitting_detector.deflated_sharpe import EULER_MASCHERONI, deflated_sharpe_ratio


def _normal(n=520, mean=0.003, sd=0.02, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mean, sd, n))


def test_sigma_sr_reduces_to_normal_closed_form():
    """For normal returns (skew 0, kurtosis 3) the paper formula collapses
    to sqrt((1 + SR^2/2) / (T-1)). Check we land close to that."""
    r = _normal()
    out = deflated_sharpe_ratio(r, num_trials=1, trial_sharpes=[])
    sr = out["raw_sharpe"]
    expected = math.sqrt((1 + sr**2 / 2) / (len(r) - 1))
    assert out["sharpe_std_error"] == pytest.approx(expected, rel=0.15)


def test_n_equals_one_is_plain_significance_with_warning():
    out = deflated_sharpe_ratio(_normal(), num_trials=1, trial_sharpes=[])
    assert out["expected_max_sr"] == 0.0
    assert any("N=1" in w for w in out["warnings"])
    plain = stats.norm.cdf(out["raw_sharpe"] / out["sharpe_std_error"])
    assert out["dsr"] == pytest.approx(plain)


def test_sr0_matches_hand_formula():
    trial_sharpes = [0.05, 0.10, 0.02, -0.03, 0.08, 0.12, 0.00, 0.04, 0.07, 0.09]
    N = len(trial_sharpes)
    out = deflated_sharpe_ratio(_normal(), num_trials=N, trial_sharpes=trial_sharpes)
    V = np.var(trial_sharpes, ddof=1)
    sr0 = math.sqrt(V) * (
        (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / N)
        + EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (N * math.e))
    )
    assert out["expected_max_sr"] == pytest.approx(sr0)
    assert out["overfitting_gap"] == pytest.approx(out["raw_sharpe"] - sr0)


def test_more_trials_means_lower_dsr():
    r = _normal()
    rng = np.random.default_rng(1)
    few = list(rng.normal(0, 0.1, 5))
    many = list(rng.normal(0, 0.1, 200))
    assert deflated_sharpe_ratio(r, 200, many)["dsr"] < deflated_sharpe_ratio(r, 5, few)["dsr"]


def test_short_series_warns():
    out = deflated_sharpe_ratio(_normal(n=20), num_trials=3, trial_sharpes=[0.1, 0.2, 0.3])
    assert any("T=20" in w for w in out["warnings"])


def test_kurtosis_is_non_excess():
    """Fat tails must WIDEN sigma_SR. With excess kurtosis (normal=0) the
    (g4-1)/4 term would go negative for normal data -- a classic bug."""
    fat = pd.Series(np.random.default_rng(2).standard_t(3, 520) * 0.012 + 0.003)
    assert stats.kurtosis(fat, fisher=False) > 3.5
    sig_n = deflated_sharpe_ratio(_normal(), 1, [])["sharpe_std_error"]
    sig_f = deflated_sharpe_ratio(fat, 1, [])["sharpe_std_error"]
    assert sig_n >= math.sqrt(1 / 519) * 0.95
    assert sig_f > sig_n * 0.9


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(pd.Series([0.1, 0.2]), 1, [])
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(_normal(), 0, [])


def test_annualized_inputs_give_same_dsr_as_per_period():
    r = _normal()
    weekly = [0.05, 0.10, 0.02, -0.03, 0.08, 0.12, 0.00, 0.04]
    annual = [s * math.sqrt(52) for s in weekly]
    a = deflated_sharpe_ratio(r, 8, weekly, sharpe_basis="per_period")
    b = deflated_sharpe_ratio(r, 8, annual, sharpe_basis="annualized")
    assert b["dsr"] == pytest.approx(a["dsr"])
    assert b["expected_max_sr"] == pytest.approx(a["expected_max_sr"])
    assert b["raw_sharpe"] == pytest.approx(a["raw_sharpe"])
    assert a["raw_sharpe_annualized"] == pytest.approx(a["raw_sharpe"] * math.sqrt(52))


def test_mismatched_units_are_caught():
    r = _normal()
    annual_looking = [1.5, 2.1, 0.8, 1.9, 2.4]
    with pytest.raises(ValueError, match="look annualized"):
        deflated_sharpe_ratio(r, 5, annual_looking)  # default basis = per_period
    with pytest.raises(ValueError, match="sharpe_basis"):
        deflated_sharpe_ratio(r, 5, [0.1] * 5, sharpe_basis="weekly")
    # the same numbers are fine once the unit is declared
    deflated_sharpe_ratio(r, 5, annual_looking, sharpe_basis="annualized")


from overfitting_detector.deflated_sharpe import effective_num_trials


def _matrix(n_cols, seed=0, identical=False, n=300):
    rng = np.random.default_rng(seed)
    if identical:
        base = rng.normal(0, 0.02, n)
        return pd.DataFrame({f"t{i}": base for i in range(n_cols)})
    return pd.DataFrame(rng.normal(0, 0.02, (n, n_cols)), columns=[f"t{i}" for i in range(n_cols)])


def test_effective_n_identical_series_is_one():
    assert effective_num_trials(_matrix(10, identical=True)) == 1


def test_effective_n_independent_series_is_about_ten():
    assert effective_num_trials(_matrix(10, seed=1)) >= 9


def test_effective_n_near_duplicates_collapse():
    M = _matrix(5, seed=2)
    noisy = M + np.random.default_rng(3).normal(0, 0.002, M.shape)  # 5 near-copies
    both = pd.concat([M, noisy.add_prefix("dup_")], axis=1)
    assert effective_num_trials(both) == 5


def test_dsr_reports_both_counts_and_mode_is_explicit():
    r = _normal()
    M = _matrix(10, identical=True)
    sharpes = [0.01 * i for i in range(10)]  # spread out, so V > 0 and N matters
    raw = deflated_sharpe_ratio(r, 10, sharpes, trial_returns_matrix=M)
    assert raw["num_trials_mode"] == "raw" and raw["num_trials"] == 10
    assert raw["num_trials_effective"] == 1
    assert raw["dsr"] == pytest.approx(raw["dsr_raw_n"])
    assert raw["dsr_effective_n"] > raw["dsr_raw_n"]  # fewer trials -> less deflation
    eff = deflated_sharpe_ratio(r, 10, sharpes, trial_returns_matrix=M, num_trials_mode="effective")
    assert eff["num_trials"] == 1 and eff["dsr"] == pytest.approx(eff["dsr_effective_n"])
    assert eff["dsr_raw_n"] == pytest.approx(raw["dsr_raw_n"])  # both always present
    no_matrix = deflated_sharpe_ratio(r, 10, sharpes)
    assert no_matrix["dsr_effective_n"] is None and no_matrix["num_trials_effective"] is None


def test_effective_mode_requires_matrix_and_valid_mode():
    with pytest.raises(ValueError, match="requires trial_returns_matrix"):
        deflated_sharpe_ratio(_normal(), 5, [0.1] * 5, num_trials_mode="effective")
    with pytest.raises(ValueError, match="num_trials_mode"):
        deflated_sharpe_ratio(_normal(), 5, [0.1] * 5, num_trials_mode="clustered")
