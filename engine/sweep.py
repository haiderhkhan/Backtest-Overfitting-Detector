"""Parameter sweep: run EVERY config and log EVERY one. No cherry-picking.

The honest trial count is what makes DSR and PBO mean anything. If you
ever filter here, the detector will lie to you politely.
"""

from __future__ import annotations

from itertools import product

import pandas as pd

from engine.momentum import run_momentum
from overfitting_detector import log_trial

GRID_VERSION = "v2"  # bump whenever DEFAULT_GRID changes so tags never collide

DEFAULT_GRID = {
    "lookback_weeks": [4, 8, 13, 26, 52],
    "skip_weeks": [0, 1],
    "holding_weeks": [1, 4],
    "n_quantiles": [3, 5],
    "long_only": [False, True],
    "cost_bps": [0.0, 25.0],
    "direction": ["momentum", "reversal"],
}  # 5 * 2 * 2 * 2 * 2 * 2 * 2 = 320 configs

# v1 (tag psx_momentum_kse100_liquid_starter_2026-09-18): 90 configs,
# lookback x skip x holding[1,2,4] x quantiles[3,4,5], long-short, no cost.

MIN_WEEKS_PER_TRIAL = 52  # skip (and warn about) configs that leave < 1y of returns


def build_grid(grid: dict | None = None) -> list[dict]:
    g = grid or DEFAULT_GRID
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in product(*(g[k] for k in keys))]


def sweep_tag(universe_name: str, run_date: str) -> str:
    """One tag per (grid version, universe, day). Old sweeps stay untouched."""
    return f"psx_momentum_{GRID_VERSION}_{universe_name}_{run_date}"


def run_sweep(
    weekly_returns: pd.DataFrame,
    tag: str,
    db_path: str,
    grid: dict | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """Run every config, log every config under one tag.

    Returns {"trial_ids": [...], "params": [...], "skipped": [...],
             "warnings": [...]}. A config is skipped ONLY if it produces
             too few return weeks to log at all; that is recorded, not
             hidden, because it still counts as something you tried.
    """
    configs = build_grid(grid)
    ids, used, skipped, warnings = [], [], [], []
    for p in configs:
        series = run_momentum(weekly_returns, **p)
        if len(series) < MIN_WEEKS_PER_TRIAL:
            skipped.append(p)
            warnings.append(f"skipped {p}: only {len(series)} weeks of returns")
            continue
        tid = log_trial(
            {"factors": ["momentum"], "params": p, "risk_free_rate": risk_free_rate, "tag": tag},
            series,
            db_path=db_path,
        )
        ids.append(tid)
        used.append(p)
    if skipped:
        warnings.append(
            f"{len(skipped)} of {len(configs)} configs skipped — the logged trial count "
            "understates what was tried; treat DSR/PBO as slightly optimistic"
        )
    return {"trial_ids": ids, "params": used, "skipped": skipped, "warnings": warnings}
