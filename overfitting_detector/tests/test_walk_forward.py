"""Walk-forward on a series with a deliberate performance break."""

import numpy as np
import pandas as pd
import pytest

from overfitting_detector.walk_forward import walk_forward_validate

FOLD_COLS = [
    "fold", "is_start", "is_end", "oos_start", "oos_end",
    "is_sharpe", "oos_sharpe", "is_max_drawdown", "oos_max_drawdown",
]


def _series(n_weeks, drift_first, drift_second, seed=0, sd=0.01):
    rng = np.random.default_rng(seed)
    half = n_weeks // 2
    r = np.concatenate([
        rng.normal(drift_first, sd, half),
        rng.normal(drift_second, sd, n_weeks - half),
    ])
    return pd.Series(r, index=pd.date_range("2015-01-02", periods=n_weeks, freq="W-FRI"))


def test_fold_schema_and_no_overlap():
    out = walk_forward_validate(_series(520, 0.004, 0.004), "2Y", "6M", "6M")
    f = out["folds"]
    assert list(f.columns) == FOLD_COLS
    assert out["num_folds"] == len(f) >= 4
    assert (f["oos_start"] > f["is_end"]).all()
    assert (f["is_start"].diff().dropna() > pd.Timedelta(0)).all()


def test_stable_series_has_decay_near_one():
    out = walk_forward_validate(_series(520, 0.004, 0.004), "2Y", "6M", "6M")
    assert 0.6 < out["decay_ratio"] < 1.4
    assert out["warnings"] == []


def test_performance_break_is_flagged_as_decay():
    # strong edge for 5 years, then it vanishes: OOS Sharpe collapses
    out = walk_forward_validate(_series(520, 0.006, -0.002), "2Y", "6M", "6M")
    assert out["decay_ratio"] < 0.4


def test_insufficient_history_warns():
    out = walk_forward_validate(_series(120, 0.004, 0.004), "1Y", "6M", "6M")
    assert out["num_folds"] < 4
    assert any("insufficient history" in w for w in out["warnings"])


def test_zero_is_sharpe_gives_none():
    # +1% / -1% alternating: mean is exactly zero in every window
    idx = pd.date_range("2015-01-02", periods=520, freq="W-FRI")
    flat = pd.Series(np.tile([0.01, -0.01], 260), index=idx)
    out = walk_forward_validate(flat, "2Y", "6M", "6M")
    assert out["decay_ratio"] is None
    assert any("undefined" in w for w in out["warnings"])


def test_requires_datetime_index():
    with pytest.raises(ValueError):
        walk_forward_validate(pd.Series([0.1, 0.2, 0.3]), "1Y", "6M", "6M")
