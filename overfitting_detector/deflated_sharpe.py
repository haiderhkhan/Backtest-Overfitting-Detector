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

Sharpe ratios here are per-period (weekly), NOT annualized — the paper's
sigma_SR formula assumes SR and T are on the same time scale.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329
MIN_OBS_FOR_MOMENTS = 30  # below this, skew/kurtosis estimates are noise


def deflated_sharpe_ratio(
    returns: pd.Series,
    num_trials: int,
    trial_sharpes: list[float],
) -> dict:
    """Compute the DSR for one candidate strategy.

    returns:        the candidate's weekly log returns.
    num_trials:     N — how many strategies were tried in total.
    trial_sharpes:  the (weekly, per-period) Sharpe of EVERY trial, from
                    the trial log. Its variance is V in the SR_0 formula.

    Returns a dict with raw_sharpe, expected_max_sr, sharpe_std_error,
    dsr, overfitting_gap, and a list of warnings.
    """
    r = pd.Series(returns).dropna().astype(float)
    T = len(r)
    if T < 3:
        raise ValueError("need at least 3 return observations")
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")

    warnings: list[str] = []
    sd = r.std(ddof=1)
    sr = float(r.mean() / sd) if sd > 0 else 0.0

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
        "raw_sharpe": sr,
        "expected_max_sr": sr0,
        "sharpe_std_error": sigma_sr,
        "dsr": dsr,
        "overfitting_gap": sr - sr0,
        "num_trials": N,
        "num_observations": T,
        "warnings": warnings,
    }
