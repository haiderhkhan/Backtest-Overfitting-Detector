"""Walk-forward validation: does performance survive moving through time?

Plain-English idea: pretend you're living through history. Fit/measure on
a `train_window` of past data (in-sample, IS), then see what happened in
the next `test_window` (out-of-sample, OOS). Roll forward by `step` and
repeat. If OOS Sharpe is consistently much lower than IS Sharpe, the
strategy "decays" — it looked good in hindsight but not going forward.

decay_ratio = mean(OOS Sharpe) / mean(IS Sharpe). 1.0 = no decay.
"""

from __future__ import annotations

import re

import pandas as pd

from .metrics import max_drawdown, sharpe_annualized

MIN_FOLDS = 4
NEAR_ZERO_SHARPE = 0.05  # below this, dividing by mean IS Sharpe is meaningless

_UNIT = {"Y": "years", "M": "months", "W": "weeks", "D": "days"}


def _parse_window(text: str) -> pd.DateOffset:
    """'3Y' -> 3 years, '6M' -> 6 months, '26W' -> 26 weeks, '90D' -> 90 days.

    Hand-rolled because pandas 3 removed the bare 'Y'/'M' offset aliases.
    """
    m = re.fullmatch(r"\s*(\d+)\s*([YMWD])\s*", str(text).upper())
    if not m:
        raise ValueError(f"window {text!r} must look like '3Y', '6M', '26W' or '90D'")
    return pd.DateOffset(**{_UNIT[m.group(2)]: int(m.group(1))})


def walk_forward_validate(
    returns: pd.Series,
    train_window: str = "3Y",
    test_window: str = "6M",
    step: str = "6M",
    risk_free_rate: float = 0.0,
) -> dict:
    """Roll train/test windows over a date-indexed weekly return series.

    Windows are strings like "3Y", "6M", "26W", "90D". Each fold is
    [train_start, train_end) then [train_end, test_end), so IS and OOS
    never overlap. Rolling stops when a full test window no longer fits.

    Returns {"folds": DataFrame, "decay_ratio": float | None,
             "num_folds": int, "warnings": list[str]}.
    """
    r = pd.Series(returns).dropna().astype(float).sort_index()
    if not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError("returns must have a DatetimeIndex")
    if r.empty:
        raise ValueError("returns is empty")

    train_off = _parse_window(train_window)
    test_off = _parse_window(test_window)
    step_off = _parse_window(step)

    rows = []
    start = r.index[0]
    last = r.index[-1]
    fold = 1
    while True:
        train_end = start + train_off
        test_end = train_end + test_off
        if test_end > last + pd.Timedelta(days=1):
            break
        is_r = r[(r.index >= start) & (r.index < train_end)]
        oos_r = r[(r.index >= train_end) & (r.index < test_end)]
        if len(is_r) < 2 or len(oos_r) < 2:
            break
        rows.append(
            {
                "fold": fold,
                "is_start": is_r.index[0],
                "is_end": is_r.index[-1],
                "oos_start": oos_r.index[0],
                "oos_end": oos_r.index[-1],
                "is_sharpe": sharpe_annualized(is_r, risk_free_rate),
                "oos_sharpe": sharpe_annualized(oos_r, risk_free_rate),
                "is_max_drawdown": max_drawdown(is_r),
                "oos_max_drawdown": max_drawdown(oos_r),
            }
        )
        fold += 1
        start = start + step_off

    folds = pd.DataFrame(rows)
    warnings: list[str] = []
    if len(folds) < MIN_FOLDS:
        warnings.append(
            f"only {len(folds)} folds (< {MIN_FOLDS}): insufficient history "
            "for a reliable decay ratio"
        )

    decay_ratio = None
    if len(folds):
        mean_is = folds["is_sharpe"].mean()
        mean_oos = folds["oos_sharpe"].mean()
        if abs(mean_is) < NEAR_ZERO_SHARPE:
            warnings.append(
                f"mean IS Sharpe {mean_is:.3f} is ~0; decay ratio undefined"
            )
        else:
            decay_ratio = float(mean_oos / mean_is)

    return {
        "folds": folds,
        "decay_ratio": decay_ratio,
        "num_folds": int(len(folds)),
        "warnings": warnings,
    }
