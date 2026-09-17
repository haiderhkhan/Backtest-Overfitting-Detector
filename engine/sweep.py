"""Parameter sweep: run EVERY config and log EVERY one. No cherry-picking.

The honest trial count is what makes DSR and PBO mean anything. If you
ever filter here, the detector will lie to you politely.
"""

from __future__ import annotations

from itertools import product

import pandas as pd

from engine.momentum import run_momentum
from engine.value import VALUE_PROXY, run_value
from overfitting_detector import log_trial

# "factor" key in a config picks the engine; missing key = momentum (v1-v3).
ENGINES = {"momentum": run_momentum, "value": run_value}

GRID_VERSION = "v5_combo"  # bump whenever the default grid changes so tags never collide

# Every grid ever run stays here so old tags remain reproducible.
GRIDS = {
    # v1 tag: psx_momentum_kse100_liquid_starter_2026-09-18 (no version in tag)
    "v1": {
        "lookback_weeks": [4, 8, 13, 26, 52],
        "skip_weeks": [0, 1],
        "holding_weeks": [1, 2, 4],
        "n_quantiles": [3, 4, 5],
    },  # 90, long-short, no cost
    "v2": {
        "lookback_weeks": [4, 8, 13, 26, 52],
        "skip_weeks": [0, 1],
        "holding_weeks": [1, 4],
        "n_quantiles": [3, 5],
        "long_only": [False, True],
        "cost_bps": [0.0, 25.0],
        "direction": ["momentum", "reversal"],
    },  # 320
    "v3": {
        # only what a PSX retail account could actually execute:
        # long-only, and paying costs. 5*2*2*2*1*2*2 = 160
        "lookback_weeks": [4, 8, 13, 26, 52],
        "skip_weeks": [0, 1],
        "holding_weeks": [1, 4],
        "n_quantiles": [3, 5],
        "long_only": [True],
        "cost_bps": [25.0, 50.0],
        "direction": ["momentum", "reversal"],
    },
    "v4_value": {
        # value PROXY (long-horizon reversal, see engine/value.py), executable
        # only. 3*2*2*2*1*2*2 = 96
        "factor": ["value"],
        "lookback_weeks": [52, 104, 156],
        "skip_weeks": [0, 52],
        "holding_weeks": [1, 4],
        "n_quantiles": [3, 5],
        "long_only": [True],
        "cost_bps": [25.0, 50.0],
        "direction": ["value", "glamour"],
    },
}


def _expand(axes: dict) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in product(*(axes[k] for k in keys))]


# v5_combo: momentum (v3) and value (v4) configs in ONE tag, so PBO ranks
# genuinely different strategies against each other. 160 + 96 = 256.
GRIDS["v5_combo"] = [{"factor": "momentum", **c} for c in _expand(GRIDS["v3"])] + _expand(GRIDS["v4_value"])
DEFAULT_GRID = GRIDS[GRID_VERSION]

MIN_WEEKS_PER_TRIAL = 52  # skip (and warn about) configs that leave < 1y of returns


def build_grid(grid: dict | list | None = None) -> list[dict]:
    """A dict of axes -> full cartesian product; a list is already configs."""
    g = DEFAULT_GRID if grid is None else grid
    if isinstance(g, list):
        return [dict(c) for c in g]
    return _expand(g)


def sweep_tag(universe_name: str, run_date: str, version: str = GRID_VERSION) -> str:
    """One tag per (grid version, universe, day). Old sweeps stay untouched."""
    return f"psx_momentum_{version}_{universe_name}_{run_date}"


def run_sweep(
    weekly_returns: pd.DataFrame,
    tag: str,
    db_path: str,
    grid: dict | list | None = None,
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
        factor = p.get("factor", "momentum")
        engine_kwargs = {k: v for k, v in p.items() if k != "factor"}
        series = ENGINES[factor](weekly_returns, **engine_kwargs)
        if len(series) < MIN_WEEKS_PER_TRIAL:
            skipped.append(p)
            warnings.append(f"skipped {p}: only {len(series)} weeks of returns")
            continue
        logged = {**p, "factor": factor}
        if factor == "value":
            logged["value_proxy"] = VALUE_PROXY  # never let a proxy look like real fundamentals
        tid = log_trial(
            {"factors": [factor], "params": logged, "risk_free_rate": risk_free_rate, "tag": tag},
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
