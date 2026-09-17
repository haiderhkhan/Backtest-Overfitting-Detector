"""Crude cross-sectional momentum on a weekly log-return matrix.

Each rebalance: score every ticker by its cumulative log return over the
`lookback_weeks` window ending `skip_weeks` ago (the skip avoids the
well-known short-term reversal). Rank, go long the top 1/n_quantiles,
short the bottom 1/n_quantiles, equal weight, hold for `holding_weeks`,
repeat.

Output: ONE pd.Series of weekly log returns for the long-short portfolio,
DatetimeIndex, exactly the shape dashboard/mock_engine.py promises the
detector.

KNOWN OMISSION: no transaction costs, no slippage, no borrow cost. See
docs/LIMITATIONS.md. Every Sharpe from this engine is an upper bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_NAMES_PER_LEG = 2  # need at least this many valid tickers in each leg


def momentum_signal(returns: pd.DataFrame, lookback_weeks: int, skip_weeks: int) -> pd.DataFrame:
    """Score at week t = sum of log returns over (t-skip-lookback, t-skip]."""
    if lookback_weeks < 1 or skip_weeks < 0:
        raise ValueError("lookback_weeks >= 1 and skip_weeks >= 0 required")
    window = returns.rolling(lookback_weeks, min_periods=lookback_weeks).sum()
    return window.shift(skip_weeks)


def run_momentum(
    weekly_returns: pd.DataFrame,
    lookback_weeks: int = 26,
    skip_weeks: int = 1,
    holding_weeks: int = 1,
    n_quantiles: int = 5,
) -> pd.Series:
    """Weekly log returns of the equal-weight long-short momentum portfolio.

    Positions chosen at the end of week t earn week t+1's returns (no
    look-ahead). Weeks where either leg has < MIN_NAMES_PER_LEG valid
    names are dropped, so the series starts after lookback+skip weeks.
    """
    if n_quantiles < 2 or holding_weeks < 1:
        raise ValueError("n_quantiles >= 2 and holding_weeks >= 1 required")
    R = weekly_returns.sort_index()
    signal = momentum_signal(R, lookback_weeks, skip_weeks)
    simple = np.expm1(R)  # average simple returns, then back to log

    out = {}
    longs = shorts = None
    for i in range(len(R) - 1):
        t = R.index[i]
        if (i % holding_weeks) == 0:
            row = signal.loc[t].dropna()
            row = row[R.loc[t].notna()[row.index]]  # must be trading now
            k = len(row) // n_quantiles
            if k >= MIN_NAMES_PER_LEG:
                ranked = row.sort_values()
                shorts, longs = list(ranked.index[:k]), list(ranked.index[-k:])
            else:
                longs = shorts = None
        if not longs:
            continue
        nxt = simple.iloc[i + 1]
        lr, sr = nxt[longs].dropna(), nxt[shorts].dropna()
        if len(lr) < MIN_NAMES_PER_LEG or len(sr) < MIN_NAMES_PER_LEG:
            continue
        port_simple = lr.mean() - sr.mean()
        out[R.index[i + 1]] = float(np.log1p(port_simple))

    s = pd.Series(out, dtype=float)
    s.index = pd.DatetimeIndex(s.index, name="period_end_date")
    s.name = f"mom_lb{lookback_weeks}_sk{skip_weeks}_h{holding_weeks}_q{n_quantiles}"
    return s
