"""Cross-sectional "value" on PSX — with a PRICE-BASED PROXY, flagged loudly.

What we wanted: earnings yield (E/P) or book-to-price (B/P) per stock per
week. What psxdata exposes: /fundamentals returns only filing document
listings (no numbers); /quote and /screener give TODAY's P/E only, no
history. Using today's P/E across ten years of history would be pure
look-ahead. So there is no honest fundamental value signal here yet.

PROXY USED: long-horizon price reversal. Score = MINUS the cumulative log
return over `lookback_weeks` (default 156 = 3 years) ending `skip_weeks`
ago (default 52, so the last year — the momentum window — is excluded).
"Cheap" = fell the most over the prior multi-year window. This is the
De Bondt-Thaler (1985) long-term reversal effect, which Fama-French show
loads on the value factor, but it is NOT book-to-price. Treat results as
"long-horizon reversal", and see docs/LIMITATIONS.md item 9.

Because the proxy is price-based, it shares raw material with momentum;
a "value vs momentum" comparison here is weaker than it would be with
real fundamentals. The horizons do not overlap (skip >= 52 vs momentum
lookback <= 52), which is the only thing keeping the two signals distinct.

Same contract as momentum.py: long_only, cost_bps, direction, weekly log
returns out.
"""

from __future__ import annotations

import pandas as pd

from engine.momentum import momentum_signal, run_ranked_portfolio

DIRECTIONS = ("value", "glamour")  # glamour = long the multi-year winners
VALUE_PROXY = "long_horizon_reversal"  # recorded in every trial's params


def value_signal(returns: pd.DataFrame, lookback_weeks: int = 156, skip_weeks: int = 52) -> pd.DataFrame:
    """Cheapness proxy: minus the cumulative log return over the window."""
    return -momentum_signal(returns, lookback_weeks, skip_weeks)


def run_value(
    weekly_returns: pd.DataFrame,
    lookback_weeks: int = 156,
    skip_weeks: int = 52,
    holding_weeks: int = 4,
    n_quantiles: int = 5,
    long_only: bool = False,
    cost_bps: float = 0.0,
    direction: str = "value",
) -> pd.Series:
    """Weekly log returns of the value-proxy portfolio (see module doc)."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    signal = value_signal(weekly_returns.sort_index(), lookback_weeks, skip_weeks)
    if direction == "glamour":
        signal = -signal
    leg = "lo" if long_only else "ls"
    name = f"{direction[:3]}_lb{lookback_weeks}_sk{skip_weeks}_h{holding_weeks}_q{n_quantiles}_{leg}_c{int(cost_bps)}"
    return run_ranked_portfolio(weekly_returns, signal, holding_weeks, n_quantiles, long_only, cost_bps, name)
