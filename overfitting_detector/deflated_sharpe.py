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

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .metrics import PERIODS_PER_YEAR, sharpe_per_period

EULER_MASCHERONI = 0.5772156649015329
MIN_OBS_FOR_MOMENTS = 30  # below this, skew/kurtosis estimates are noise
# A weekly per-period Sharpe above 1.0 is an annualized ~7.2 — not a real
# strategy, almost certainly an annualized number passed as per-period.
MAX_PLAUSIBLE_PER_PERIOD_SHARPE = 1.0
SHARPE_BASES = ("per_period", "annualized")
NUM_TRIALS_MODES = ("raw", "effective")
# Two trials whose return series have |corr| above this are "the same bet"
# for counting purposes. 0.5 is a convention; the paper (López de Prado &
# Lewis 2019, "Detection of false investment strategies using unsupervised
# learning") clusters more carefully. Tune, but never silently.
DEFAULT_CLUSTER_MAX_DISTANCE = 0.5


def effective_num_trials(
    trial_returns_matrix: pd.DataFrame,
    max_distance: float = DEFAULT_CLUSTER_MAX_DISTANCE,
) -> int:
    """How many INDEPENDENT bets a set of trials really represents.

    A parameter sweep produces many near-identical return series (lookback
    26 vs 27 weeks). Counting each as a separate trial over-deflates the
    Sharpe. Cluster the columns by distance = 1 - |correlation| with
    average-linkage hierarchical clustering; clusters closer than
    `max_distance` merge. The number of clusters is the effective N.
    """
    M = pd.DataFrame(trial_returns_matrix).dropna(how="any")
    n = M.shape[1]
    if n <= 1:
        return n
    corr = M.corr().abs().fillna(0.0).to_numpy().copy()  # pandas 3 returns read-only
    np.fill_diagonal(corr, 1.0)
    dist = np.clip(1.0 - corr, 0.0, 1.0)
    dist = (dist + dist.T) / 2  # exact symmetry for squareform
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=max_distance, criterion="distance")
    return int(len(np.unique(labels)))


def _expected_max_sharpe(num_trials: int, variance: float) -> float:
    """SR_0: the best Sharpe you'd expect from N skill-less trials."""
    if num_trials <= 1:
        return 0.0
    return math.sqrt(variance) * (
        (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / num_trials)
        + EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (num_trials * math.e))
    )


def deflated_sharpe_ratio(
    returns: pd.Series,
    num_trials: int,
    trial_sharpes: list[float],
    sharpe_basis: str = "per_period",
    periods_per_year: int = PERIODS_PER_YEAR,
    num_trials_mode: str = "raw",
    trial_returns_matrix: pd.DataFrame | None = None,
) -> dict:
    """Compute the DSR for one candidate strategy.

    num_trials_mode:      "raw" uses num_trials as given (default, never
                          switched silently). "effective" uses the cluster
                          count from effective_num_trials() and REQUIRES
                          trial_returns_matrix.
    trial_returns_matrix: T x N returns of every trial. When given, BOTH
                          dsr_raw_n and dsr_effective_n are reported
                          whatever the mode; neither replaces the other.

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
    if num_trials_mode not in NUM_TRIALS_MODES:
        raise ValueError(f"num_trials_mode must be one of {NUM_TRIALS_MODES}, got {num_trials_mode!r}")
    if num_trials_mode == "effective" and trial_returns_matrix is None:
        raise ValueError("num_trials_mode='effective' requires trial_returns_matrix")

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

    # Step 2: expected max Sharpe under the null of zero skill, for the RAW
    # trial count and (if we can) the EFFECTIVE one. Both are reported.
    N_raw = int(num_trials)
    V = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    if N_raw == 1:
        warnings.append(
            "N=1: no multiple-testing correction possible; SR_0 set to 0 so "
            "DSR is just the plain significance of SR"
        )
    elif len(trial_sharpes) != N_raw:
        warnings.append(f"len(trial_sharpes)={len(trial_sharpes)} != num_trials={N_raw}")

    N_eff = None
    if trial_returns_matrix is not None:
        N_eff = effective_num_trials(trial_returns_matrix)
        if N_eff > N_raw:
            warnings.append(f"effective N ({N_eff}) > raw N ({N_raw}); check inputs")

    def _dsr_for(n: int) -> tuple[float, float]:
        sr0_n = _expected_max_sharpe(n, V)
        return sr0_n, float(stats.norm.cdf((sr - sr0_n) / sigma_sr))

    sr0_raw, dsr_raw = _dsr_for(N_raw)
    sr0_eff, dsr_eff = _dsr_for(N_eff) if N_eff is not None else (None, None)

    # Step 3: the headline DSR is whichever mode the caller asked for.
    if num_trials_mode == "effective":
        N, sr0, dsr = N_eff, sr0_eff, dsr_eff
    else:
        N, sr0, dsr = N_raw, sr0_raw, dsr_raw

    return {
        "sharpe_basis": "per_period",
        "raw_sharpe": sr,
        "raw_sharpe_annualized": sr * scale,
        "expected_max_sr": sr0,
        "sharpe_std_error": sigma_sr,
        "dsr": dsr,
        "overfitting_gap": sr - sr0,
        "num_trials": N,
        "num_trials_mode": num_trials_mode,
        "num_trials_raw": N_raw,
        "num_trials_effective": N_eff,
        "dsr_raw_n": dsr_raw,
        "dsr_effective_n": dsr_eff,
        "num_observations": T,
        "warnings": warnings,
    }
