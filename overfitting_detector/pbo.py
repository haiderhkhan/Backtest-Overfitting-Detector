"""Probability of Backtest Overfitting via CSCV.

Source: Bailey, Borwein, López de Prado, Zhu (2017), "The Probability of
Backtest Overfitting".

Plain-English idea: chop history into S equal chunks. Pick half the chunks
as "in-sample" (IS), the other half as "out-of-sample" (OOS). Find the
strategy that wins IS. Ask: how did that winner rank OOS? Repeat for
EVERY way of choosing the half. If the IS winner usually lands in the
bottom half OOS, your selection process is picking luck, not skill.

PBO = fraction of splits where the IS winner is below the OOS median.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from .metrics import sharpe_annualized

MIN_STRATEGIES = 5
MIN_ROWS_PER_CHUNK = 15


def compute_pbo(
    strategy_returns_matrix: pd.DataFrame,
    num_partitions: int = 16,
) -> dict:
    """CSCV probability of backtest overfitting.

    strategy_returns_matrix: T periods (rows) x N strategies (columns).
                             Usually one tagged group from the trial log.
    num_partitions:          S, must be even. 16 -> C(16, 8) = 12870 splits.

    Returns pbo, the per-split logits (lambdas) and relative ranks (omegas)
    for plotting, num_combinations, and warnings.
    """
    if num_partitions < 2 or num_partitions % 2:
        raise ValueError("num_partitions must be an even integer >= 2")
    M = strategy_returns_matrix.dropna(how="any")
    T, N = M.shape
    if N < 2:
        raise ValueError("need at least 2 strategies to compare")
    if T < num_partitions:
        raise ValueError(f"only {T} rows for {num_partitions} partitions")

    warnings: list[str] = []
    if N < MIN_STRATEGIES:
        warnings.append(f"only {N} strategies (< {MIN_STRATEGIES}); PBO is coarse")
    rows_per_chunk = T // num_partitions
    if rows_per_chunk < MIN_ROWS_PER_CHUNK:
        warnings.append(
            f"{rows_per_chunk} rows per chunk (< {MIN_ROWS_PER_CHUNK}); "
            "per-chunk Sharpes are noisy"
        )

    # Equal-length contiguous chunks; drop the leftover tail rows.
    usable = rows_per_chunk * num_partitions
    chunks = np.array_split(M.iloc[:usable].to_numpy(), num_partitions)
    half = num_partitions // 2
    all_idx = set(range(num_partitions))

    lambdas: list[float] = []
    omegas: list[float] = []
    for is_idx in combinations(range(num_partitions), half):
        oos_idx = sorted(all_idx - set(is_idx))
        is_mat = np.vstack([chunks[i] for i in is_idx])
        oos_mat = np.vstack([chunks[i] for i in oos_idx])

        is_perf = np.array([sharpe_annualized(is_mat[:, j]) for j in range(N)])
        oos_perf = np.array([sharpe_annualized(oos_mat[:, j]) for j in range(N)])

        best = int(np.argmax(is_perf))
        # rank 1 = worst OOS, N = best OOS
        rank = int((oos_perf < oos_perf[best]).sum() + 1)
        omega = rank / (N + 1)
        lam = math.log(omega / (1 - omega))
        omegas.append(omega)
        lambdas.append(lam)

    num_combos = len(lambdas)
    pbo = sum(1 for lam in lambdas if lam < 0) / num_combos
    return {
        "pbo": float(pbo),
        "lambdas": lambdas,
        "omegas": omegas,
        "num_combinations": num_combos,
        "num_strategies": N,
        "num_partitions": num_partitions,
        "warnings": warnings,
    }
