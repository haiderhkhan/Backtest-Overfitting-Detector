"""Small shared statistics used by every other module.

All inputs are WEEKLY LOG RETURNS (see ARCHITECTURE.md §2). Keeping these
in one place means the trial log, PBO, and walk-forward all agree on what
"Sharpe" means — a mismatch there would silently corrupt every comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 52  # weekly data


# Two Sharpe functions, named by UNIT, on purpose. A per-period Sharpe and
# an annualized Sharpe differ by sqrt(52) — mixing them up inside DSR gives
# a confident, wrong answer with no error. Never add an "ambiguous" one.


def sharpe_per_period(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Sharpe ratio in PER-PERIOD units (weekly): mean excess / std dev.

    This is the unit the Deflated Sharpe formula expects.
    `risk_free_rate` is ANNUAL and is converted to a per-period rate here.
    Returns 0.0 if the series has no spread (avoids divide-by-zero).
    """
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd)


def sharpe_annualized(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Sharpe ratio in ANNUALIZED units: per-period Sharpe * sqrt(52).

    This is the unit people quote ("Sharpe of 1.5") and the unit stored in
    the trial log. Convert back with sharpe_to_per_period() before DSR.
    """
    return sharpe_per_period(returns, risk_free_rate, periods_per_year) * np.sqrt(
        periods_per_year
    )


def sharpe_to_per_period(
    annualized: float, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Annualized Sharpe -> per-period Sharpe (divide by sqrt(52))."""
    return float(annualized) / np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall in cumulative return, as a NEGATIVE fraction.

    Because inputs are log returns, cumulative wealth is exp(cumsum).
    A result of -0.25 means "at the worst point you were 25% below your
    previous high".
    """
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return 0.0
    wealth = np.exp(r.cumsum())
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def annualized_return(returns: pd.Series) -> float:
    """Mean weekly log return times 52 (log returns add, so this is exact)."""
    r = pd.Series(returns).dropna().astype(float)
    return float(r.mean() * PERIODS_PER_YEAR) if len(r) else 0.0


def annualized_volatility(returns: pd.Series) -> float:
    """Weekly standard deviation scaled to a yearly figure."""
    r = pd.Series(returns).dropna().astype(float)
    return float(r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(r) > 1 else 0.0
