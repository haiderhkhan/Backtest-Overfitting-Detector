"""Daily prices -> weekly log returns matrix (dates x tickers).

PSX realities are handled by WARNING, never crashing:
  - missing trading days for a ticker (gaps vs the union calendar)
  - stale prices (runs of identical closes = no trades)
  - listings that start after the universe start (mid-history)
  - delistings / data ending early
  - holiday-shortened weeks
The result dict carries a "warnings": list[str], same convention as the
detector modules, so report/dashboard code can surface them unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_DAY_WARN_FRACTION = 0.05   # >5% of union trading days missing
STALE_RUN_DAYS = 5                  # >=5 identical closes in a row
LATE_START_DAYS = 30                # first price >30 days after universe start
EARLY_END_DAYS = 30                 # last price >30 days before universe end
MIN_DAYS_IN_WEEK = 3                # fewer -> holiday-shortened week
MIN_WEEKS = 60                      # below this, nothing downstream is meaningful


def build_close_matrix(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{symbol: daily df} -> wide DataFrame of closes (dates x symbols)."""
    cols = {}
    for sym, df in prices.items():
        if df is None or df.empty:
            continue
        s = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
        cols[sym] = s[~s.index.duplicated()].sort_index()
    if not cols:
        raise ValueError("no price data at all")
    return pd.DataFrame(cols).sort_index()


def _quality_warnings(close: pd.DataFrame, start, end) -> list[str]:
    w: list[str] = []
    calendar = close.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for sym in close.columns:
        s = close[sym]
        valid = s.dropna()
        if valid.empty:
            w.append(f"{sym}: no prices at all — dropped")
            continue
        first, last = valid.index[0], valid.index[-1]
        span_days = int(((calendar >= first) & (calendar <= last)).sum())
        gaps = span_days - len(valid)
        if span_days and gaps / span_days > MISSING_DAY_WARN_FRACTION:
            w.append(f"{sym}: {gaps}/{span_days} trading days missing inside its own history")
        if (first - start).days > LATE_START_DAYS:
            w.append(f"{sym}: listing/data starts {first.date()}, {(first - start).days} days after universe start")
        if (end - last).days > EARLY_END_DAYS:
            w.append(f"{sym}: data ends {last.date()}, {(end - last).days} days before universe end (delisted?)")
        runs = (valid.diff() == 0).astype(int)
        run_len = runs.groupby((runs == 0).cumsum()).cumsum()
        longest = int(run_len.max()) + 1 if len(run_len) else 0
        if longest >= STALE_RUN_DAYS:
            w.append(f"{sym}: stale price run of {longest} days (illiquid / not trading)")
    weeks = calendar.to_series().groupby(calendar.to_period("W-FRI")).count()
    short_weeks = int((weeks < MIN_DAYS_IN_WEEK).sum())
    if short_weeks:
        w.append(f"{short_weeks} holiday-shortened weeks (< {MIN_DAYS_IN_WEEK} trading days)")
    return w


def weekly_log_returns(
    prices: dict[str, pd.DataFrame],
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Daily prices -> weekly (W-FRI) log returns.

    Weekly close = last available close in the week. Log return =
    ln(close_t / close_{t-1}). A ticker's returns are NaN before it lists
    and after it stops trading.

    Returns {"returns": DataFrame, "close": DataFrame, "warnings": [...]}.
    """
    close = build_close_matrix(prices)
    start = start or close.index[0]
    end = end or close.index[-1]
    warnings = _quality_warnings(close, start, end)
    close = close.dropna(axis=1, how="all")

    weekly_close = close.resample("W-FRI").last()
    # ffill only bridges a missing week INSIDE a ticker's history; before
    # listing / after delisting stays NaN so the engine ignores it there.
    weekly_close = weekly_close.ffill().where(weekly_close.bfill().notna())
    rets = np.log(weekly_close / weekly_close.shift(1))
    rets = rets.iloc[1:]
    if len(rets) < MIN_WEEKS:
        warnings.append(f"only {len(rets)} weeks of returns (< {MIN_WEEKS}): far too thin for validation")
    return {"returns": rets, "close": weekly_close, "warnings": warnings}
