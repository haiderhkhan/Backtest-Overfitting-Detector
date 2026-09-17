"""Crude cross-sectional momentum on a weekly log-return matrix.

Each rebalance: score every ticker by its cumulative log return over the
`lookback_weeks` window ending `skip_weeks` ago (the skip avoids the
well-known short-term reversal). Rank, go long the top 1/n_quantiles,
short the bottom 1/n_quantiles, equal weight, hold for `holding_weeks`,
repeat.

Output: ONE pd.Series of weekly log returns for the long-short portfolio,
DatetimeIndex, exactly the shape dashboard/mock_engine.py promises the
detector.

Options:
  long_only   drop the short leg; the return is then BENCHMARK-RELATIVE
              (long leg minus the equal-weight universe), so it is still
              "did picking winners beat not picking".
  cost_bps    flat cost per unit of weight traded (bought OR sold), charged
              on turnover at each rebalance. 25 = 0.25% per side.
  direction   "momentum" ranks high past return as long; "reversal" inverts
              the ranking (long the losers).

Still omitted: slippage, borrow cost, market impact. See docs/LIMITATIONS.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_NAMES_PER_LEG = 2  # need at least this many valid tickers in each leg
DIRECTIONS = ("momentum", "reversal")


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
    long_only: bool = False,
    cost_bps: float = 0.0,
    direction: str = "momentum",
) -> pd.Series:
    """Weekly log returns of the equal-weight momentum (or reversal) portfolio.

    Positions chosen at the end of week t earn week t+1's returns (no
    look-ahead). Weeks where a leg has < MIN_NAMES_PER_LEG valid names are
    dropped, so the series starts after lookback+skip weeks.

    Weights: +1/k on each long, -1/k on each short (long-short) or +1/k on
    each long vs an equal-weight universe benchmark (long_only). Cost at a
    rebalance = cost_bps/1e4 * sum(|w_new - w_old|), taken out of the
    following week's simple return.
    """
    if n_quantiles < 2 or holding_weeks < 1:
        raise ValueError("n_quantiles >= 2 and holding_weeks >= 1 required")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if cost_bps < 0:
        raise ValueError("cost_bps must be >= 0")
    R = weekly_returns.sort_index()
    signal = momentum_signal(R, lookback_weeks, skip_weeks)
    if direction == "reversal":
        signal = -signal
    simple = np.expm1(R)  # average simple returns, then back to log
    cost_rate = cost_bps / 1e4

    out = {}
    weights = pd.Series(0.0, index=R.columns)  # current book
    have_position = False
    pending_cost = 0.0
    for i in range(len(R) - 1):
        t = R.index[i]
        if (i % holding_weeks) == 0:
            row = signal.loc[t].dropna()
            row = row[R.loc[t].notna()[row.index]]  # must be trading now
            k = len(row) // n_quantiles
            new_w = pd.Series(0.0, index=R.columns)
            if k >= MIN_NAMES_PER_LEG:
                ranked = row.sort_values()
                new_w[ranked.index[-k:]] = 1.0 / k
                if not long_only:
                    new_w[ranked.index[:k]] = -1.0 / k
                have_position = True
            else:
                have_position = False
            pending_cost = cost_rate * float((new_w - weights).abs().sum())
            weights = new_w
        if not have_position:
            continue
        nxt = simple.iloc[i + 1]
        held = weights[weights != 0]
        got = nxt[held.index]
        if got.notna().sum() < MIN_NAMES_PER_LEG:
            continue
        port_simple = float((held * got.fillna(0.0)).sum())
        if long_only:
            bench = nxt.dropna()
            port_simple -= float(bench.mean()) if len(bench) else 0.0
        port_simple -= pending_cost
        pending_cost = 0.0
        out[R.index[i + 1]] = float(np.log1p(port_simple))

    s = pd.Series(out, dtype=float)
    s.index = pd.DatetimeIndex(s.index, name="period_end_date")
    leg = "lo" if long_only else "ls"
    s.name = f"{direction[:3]}_lb{lookback_weeks}_sk{skip_weeks}_h{holding_weeks}_q{n_quantiles}_{leg}_c{int(cost_bps)}"
    return s
