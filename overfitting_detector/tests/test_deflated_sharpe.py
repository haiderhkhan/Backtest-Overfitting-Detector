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
