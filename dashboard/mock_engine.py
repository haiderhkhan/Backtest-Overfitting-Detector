"""A fake backtest engine. Defines the contract the REAL engine must meet.

The real PSX factor-model engine does not exist yet. Until it does, this
file is the spec: an engine hands the detector one `pd.Series` of WEEKLY
LOG RETURNS with a DatetimeIndex per trial, plus a metadata dict with
`factors`, `params`, `risk_free_rate` (annual) and an optional `tag`.
That is the whole interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from overfitting_detector import log_trial


def run_mock_backtest(params: dict, dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    """Pretend to backtest one parameter set. Returns weekly log returns.

    `edge` is the only knob: a small positive weekly drift = real skill,
    zero = pure noise. Real engines obviously do more than this.
    """
    edge = params.get("edge", 0.0)
    return pd.Series(rng.normal(edge, 0.02, len(dates)), index=dates)


def populate_mock_trials(db_path: str, tag: str, n_trials: int = 12, years: int = 8, seed: int = 42) -> list[str]:
    """Log n_trials synthetic runs; one of them has a genuine edge."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-08", periods=years * 52, freq="W-FRI")
    ids = []
    for i in range(n_trials):
        params = {"lookback_weeks": 4 + 4 * i, "edge": 0.003 if i == n_trials // 2 else 0.0}
        returns = run_mock_backtest(params, dates, rng)
        ids.append(
            log_trial(
                {"factors": ["momentum"], "params": params, "risk_free_rate": 0.0, "tag": tag},
                returns,
                db_path=db_path,
            )
        )
    return ids
