import numpy as np
import pandas as pd
import pytest

from engine.momentum import run_momentum
from engine.sweep import build_grid, run_sweep
from overfitting_detector import get_all_trials


def _weekly(n_weeks=400, n_names=20, seed=0, trend_strength=0.0):
    """Synthetic weekly log returns. trend_strength>0 gives each name a
    persistent drift: past winners keep winning (a momentum pattern)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-08", periods=n_weeks, freq="W-FRI")
    drifts = rng.normal(0, trend_strength, n_names)
    R = rng.normal(0, 0.03, (n_weeks, n_names)) + drifts
    return pd.DataFrame(R, index=idx, columns=[f"S{i}" for i in range(n_names)])


def test_output_satisfies_engine_contract():
    s = run_momentum(_weekly(), lookback_weeks=8, skip_weeks=1, holding_weeks=1, n_quantiles=5)
    assert isinstance(s, pd.Series) and isinstance(s.index, pd.DatetimeIndex)
    assert s.index.is_monotonic_increasing and s.notna().all()
    assert s.index[0] > pd.Timestamp("2016-01-08") + pd.Timedelta(weeks=8)
    assert len(s) > 350


def test_captures_known_momentum_pattern():
    s = run_momentum(_weekly(trend_strength=0.01, seed=1), 13, 1, 1, 5)
    assert s.mean() > 0
    assert s.mean() / s.std() * np.sqrt(52) > 1.5


def test_pure_noise_has_no_edge():
    sharpes = [
        run_momentum(_weekly(seed=k), 13, 1, 1, 5).pipe(lambda s: s.mean() / s.std() * np.sqrt(52))
        for k in range(5)
    ]
    assert abs(np.mean(sharpes)) < 0.6


def test_no_lookahead_positions_earn_next_week():
    R = _weekly(n_weeks=60, n_names=6, seed=3)
    R.iloc[:10, 0] = 0.05     # S0 is the obvious winner over weeks 0..9
    R.iloc[10, 0] = -0.5      # then crashes in week 10
    s = run_momentum(R, lookback_weeks=5, skip_weeks=0, holding_weeks=1, n_quantiles=3)
    assert s.loc[R.index[10]] < 0   # the long formed at week 9 eats the crash


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        run_momentum(_weekly(), 0, 1, 1, 5)
    with pytest.raises(ValueError):
        run_momentum(_weekly(), 8, 1, 1, 1)


def test_sweep_logs_exactly_grid_size(tmp_path):
    grid = {"lookback_weeks": [4, 8], "skip_weeks": [0, 1], "holding_weeks": [1], "n_quantiles": [3, 5]}
    assert len(build_grid(grid)) == 8
    db = tmp_path / "t.db"
    out = run_sweep(_weekly(), tag="g", db_path=str(db), grid=grid)
    assert len(out["trial_ids"]) == 8 and out["skipped"] == []
    df = get_all_trials(tag="g", db_path=str(db))
    assert len(df) == 8
    assert (df["sharpe_basis"] == "annualized").all()
    assert sorted(p["lookback_weeks"] for p in df["params"]) == [4, 4, 4, 4, 8, 8, 8, 8]


def test_default_grid_is_in_target_range():
    assert 40 <= len(build_grid()) <= 400


def test_reversal_is_exact_negation_of_long_short_momentum():
    R = _weekly(seed=5)
    mom = run_momentum(R, 13, 1, 1, 5, direction="momentum")
    rev = run_momentum(R, 13, 1, 1, 5, direction="reversal")
    # same names with legs swapped: simple returns negate exactly
    np.testing.assert_allclose(np.expm1(rev.values), -np.expm1(mom.values), atol=1e-12)


def test_costs_reduce_returns_and_scale_with_turnover():
    R = _weekly(seed=6)
    free = run_momentum(R, 8, 1, 1, 5, cost_bps=0)
    paid = run_momentum(R, 8, 1, 1, 5, cost_bps=25)
    assert (paid <= free + 1e-12).all() and paid.mean() < free.mean()
    drag_weekly = (run_momentum(R, 8, 1, 1, 5) - run_momentum(R, 8, 1, 1, 5, cost_bps=25)).mean()
    drag_monthly = (run_momentum(R, 8, 1, 4, 5) - run_momentum(R, 8, 1, 4, 5, cost_bps=25)).mean()
    assert drag_weekly > drag_monthly > 0


def test_long_only_is_benchmark_relative_and_captures_pattern():
    R = _weekly(trend_strength=0.01, seed=7)
    lo = run_momentum(R, 13, 1, 1, 5, long_only=True)
    assert lo.mean() > 0
    # every name identical: long leg == benchmark -> exactly zero excess
    one = np.random.default_rng(8).normal(0, 0.03, (400, 1))
    flat = pd.DataFrame(np.tile(one, (1, 10)), index=R.index, columns=[f"S{i}" for i in range(10)])
    np.testing.assert_allclose(run_momentum(flat, 8, 0, 1, 5, long_only=True).values, 0.0, atol=1e-12)


def test_rejects_bad_direction_and_cost():
    with pytest.raises(ValueError):
        run_momentum(_weekly(), 8, 1, 1, 5, direction="sideways")
    with pytest.raises(ValueError):
        run_momentum(_weekly(), 8, 1, 1, 5, cost_bps=-1)


def test_grid_covers_new_axes_and_tags_are_versioned():
    from engine.sweep import DEFAULT_GRID, GRID_VERSION, GRIDS, sweep_tag
    g = build_grid()
    assert {"long_only", "cost_bps", "direction"} <= set(g[0])
    assert len(g) == int(np.prod([len(v) for v in DEFAULT_GRID.values()]))
    assert sweep_tag("u", "2026-01-01") == f"psx_momentum_{GRID_VERSION}_u_2026-01-01"
    assert DEFAULT_GRID is GRIDS[GRID_VERSION]
    assert [len(build_grid(GRIDS[v])) for v in ("v1", "v2")] == [90, 320]


def test_v3_grid_is_executable_only():
    from engine.sweep import GRIDS
    for cfg in build_grid(GRIDS["v3"]):
        assert cfg["long_only"] is True and cfg["cost_bps"] > 0
