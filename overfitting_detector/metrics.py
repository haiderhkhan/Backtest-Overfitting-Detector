"""Small shared statistics used by every other module.

All inputs are WEEKLY LOG RETURNS (see ARCHITECTURE.md §2). Keeping these
in one place means the trial log, PBO, and walk-forward all agree on what
"Sharpe" means — a mismatch there would silently corrupt every comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 52  # weekly data


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio of a weekly log-return series.

    Sharpe = mean excess return / standard deviation, scaled by sqrt(52) so
    a weekly number reads like the yearly figures people quote.
    `risk_free_rate` is ANNUAL and is converted to a weekly rate here.
    Returns 0.0 if the series has no spread (avoids divide-by-zero).
    """
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free_rate / PERIODS_PER_YEAR
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(PERIODS_PER_YEAR))


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
