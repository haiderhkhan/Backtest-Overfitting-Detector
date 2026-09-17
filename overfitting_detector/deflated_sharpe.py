"""Deflated Sharpe Ratio — Bailey & López de Prado (2014).

Plain-English idea: if you try N strategies with zero real skill, the BEST
one will still show a decent Sharpe just by luck. DSR asks: "is the
winner's Sharpe high enough that luck alone is an unlikely explanation?"
It returns a probability (0 to 1) that the true Sharpe is above zero.

Three steps:
  1. sigma_SR  — how noisy the observed Sharpe is (wider if returns are
                 skewed or fat-tailed).
  2. SR_0      — the Sharpe you'd EXPECT the luckiest of N no-skill trials
                 to show, given how spread out the trial Sharpes were.
  3. DSR       — Phi((SR - SR_0) / sigma_SR): how many noise-units the
                 winner sits above the luck benchmark, turned into a
                 probability.

UNITS. The paper's sigma_SR formula assumes SR and T are on the same time
scale, so all math inside runs on PER-PERIOD (weekly) Sharpes. Callers may
pass annualized trial Sharpes by saying so via `sharpe_basis="annualized"`;
the conversion then happens in exactly one place below.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import PERIODS_PER_YEAR, sharpe_per_period

EULER_MASCHERONI = 0.5772156649015329
MIN_OBS_FOR_MOMENTS = 30  # below this, skew/kurtosis estimates are noise
# A weekly per-period Sharpe above 1.0 is an annualized ~7.2 — not a real
# strategy, almost certainly an annualized number passed as per-period.
MAX_PLAUSIBLE_PER_PERIOD_SHARPE = 1.0
SHARPE_BASES = ("per_period", "annualized")


def deflated_sharpe_ratio(
    returns: pd.Series,
    num_trials: int,
    trial_sharpes: list[float],
    sharpe_basis: str = "per_period",
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict:
    """Compute the DSR for one candidate strategy.

    returns:          the candidate's weekly log returns.
    num_trials:       N — how many strategies were tried in total.
    trial_sharpes:    the Sharpe of EVERY trial (from the trial log). Its
                      variance is V in the SR_0 formula.
    sharpe_basis:     the unit of `trial_sharpes`: "per_period" (weekly) or
                      "annualized". Say which; there is no safe guess.
    periods_per_year: 52 for weekly data.

    All returned Sharpe numbers (raw_sharpe, expected_max_sr, ...) are
    PER-PERIOD; raw_sharpe_annualized is added for human reading.
    """
    r = pd.Series(returns).dropna().astype(float)
    T = len(r)
    if T < 3:
        raise ValueError("need at least 3 return observations")
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")
    if sharpe_basis not in SHARPE_BASES:
        raise ValueError(f"sharpe_basis must be one of {SHARPE_BASES}, got {sharpe_basis!r}")

    # --- Unit normalization: the ONE place units are touched. ------------
    # The observed SR is always computed per-period from the raw returns.
    # trial_sharpes are converted to match if the caller says they are
    # annualized. Both then live on the same time scale as T.
    sr = sharpe_per_period(r, 0.0, periods_per_year)
    scale = math.sqrt(periods_per_year)
    if sharpe_basis == "annualized":
        trial_sharpes = [float(s) / scale for s in trial_sharpes]
    elif trial_sharpes and max(abs(float(s)) for s in trial_sharpes) > MAX_PLAUSIBLE_PER_PERIOD_SHARPE:
        raise ValueError(
            "trial_sharpes contain values > "
            f"{MAX_PLAUSIBLE_PER_PERIOD_SHARPE} but sharpe_basis='per_period'; "
            "these look annualized — pass sharpe_basis='annualized'"
        )

    warnings: list[str] = []

    # Step 1: standard error of the Sharpe estimate, corrected for
    # non-normal returns. KURTOSIS CONVENTION: the formula wants
    # NON-excess kurtosis (normal = 3). scipy's default is excess (normal
    # = 0), so we pass fisher=False. Getting this wrong is the single most
    # common DSR bug — it silently shrinks sigma_SR and inflates DSR.
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))
    if T < MIN_OBS_FOR_MOMENTS:
        warnings.append(
            f"T={T} < {MIN_OBS_FOR_MOMENTS}: skew/kurtosis estimates are unreliable"
        )
    var_sr = (1 - g3 * sr + (g4 - 1) / 4 * sr**2) / (T - 1)
    sigma_sr = math.sqrt(max(var_sr, 1e-12))

    # Step 2: expected max Sharpe under the null of zero skill.
    N = int(num_trials)
    if N == 1:
        warnings.append(
            "N=1: no multiple-testing correction possible; SR_0 set to 0 so "
            "DSR is just the plain significance of SR"
        )
        sr0 = 0.0
    else:
        V = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
        if len(trial_sharpes) != N:
            warnings.append(
                f"len(trial_sharpes)={len(trial_sharpes)} != num_trials={N}"
            )
        sr0 = math.sqrt(V) * (
            (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / N)
            + EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (N * math.e))
        )

    # Step 3: the deflated Sharpe itself.
    dsr = float(stats.norm.cdf((sr - sr0) / sigma_sr))

    return {
        "sharpe_basis": "per_period",
        "raw_sharpe": sr,
        "raw_sharpe_annualized": sr * scale,
        "expected_max_sr": sr0,
        "sharpe_std_error": sigma_sr,
        "dsr": dsr,
        "overfitting_gap": sr - sr0,
        "num_trials": N,
        "num_observations": T,
        "warnings": warnings,
    }
