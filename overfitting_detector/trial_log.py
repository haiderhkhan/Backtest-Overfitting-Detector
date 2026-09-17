"""SQLite-backed log of EVERY backtest trial, not just the winner.

Why this exists: the Deflated Sharpe Ratio needs to know how many things
you tried (N) and how spread out their Sharpes were (V). If you only keep
the winner, you cannot compute either honestly. So every run gets logged.

Two tables (schema in ARCHITECTURE.md §2):
  trials         one row per run: metadata + summary stats
  trial_returns  one row per week per run: the raw weekly log return

Dates are stored as ISO strings (YYYY-MM-DD). Numbers as REAL.
A fresh database is created on first use; nothing is reused from elsewhere.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy import stats

from .metrics import (
    PERIODS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_annualized,
    sharpe_to_per_period,
)

DEFAULT_DB_PATH = "trials.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id          TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    factors           TEXT NOT NULL,      -- JSON list
    params            TEXT NOT NULL,      -- JSON dict
    start_date        TEXT NOT NULL,
    end_date          TEXT NOT NULL,
    frequency         TEXT NOT NULL DEFAULT 'weekly',
    risk_free_rate    REAL NOT NULL,      -- explicit, never assumed
    sharpe_ratio      REAL NOT NULL,      -- unit given by sharpe_basis
    sharpe_basis      TEXT NOT NULL,      -- always 'annualized' (explicit, never guessed)
    periods_per_year  INTEGER NOT NULL,   -- 52 for weekly
    annualized_return REAL NOT NULL,
    volatility        REAL NOT NULL,
    max_drawdown      REAL NOT NULL,
    skewness          REAL NOT NULL,
    kurtosis          REAL NOT NULL,      -- NON-excess: normal = 3
    num_observations  INTEGER NOT NULL,
    tag               TEXT                -- nullable; groups trials for PBO
);
CREATE TABLE IF NOT EXISTS trial_returns (
    trial_id        TEXT NOT NULL REFERENCES trials(trial_id),
    period_end_date TEXT NOT NULL,
    return_value    REAL NOT NULL,
    PRIMARY KEY (trial_id, period_end_date)
);
CREATE INDEX IF NOT EXISTS idx_trials_tag ON trials(tag);
"""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    return conn


def _iso(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def log_trial(
    trial_metadata: dict,
    returns: pd.Series,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    """Store one backtest run and return its trial_id.

    trial_metadata keys:
      factors (list[str], required), params (dict, required),
      risk_free_rate (float, default 0.0, ANNUAL), tag (str, optional),
      trial_id (str, optional — generated if missing).
    returns: weekly log returns indexed by week-ending date.

    Summary stats (Sharpe, drawdown, skew, kurtosis...) are computed HERE
    from the returns, so every trial is graded by the same code path.
    """
    if not isinstance(returns, pd.Series) or returns.empty:
        raise ValueError("returns must be a non-empty pandas Series")
    if "factors" not in trial_metadata or "params" not in trial_metadata:
        raise ValueError("trial_metadata needs 'factors' and 'params'")

    r = returns.dropna().astype(float).sort_index()
    rf = float(trial_metadata.get("risk_free_rate", 0.0))
    trial_id = str(trial_metadata.get("trial_id") or uuid.uuid4())

    row = (
        trial_id,
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        json.dumps(list(trial_metadata["factors"])),
        json.dumps(trial_metadata["params"]),
        _iso(r.index[0]),
        _iso(r.index[-1]),
        "weekly",
        rf,
        sharpe_annualized(r, rf, PERIODS_PER_YEAR),
        "annualized",
        PERIODS_PER_YEAR,
        annualized_return(r),
        annualized_volatility(r),
        max_drawdown(r),
        float(stats.skew(r)),
        float(stats.kurtosis(r, fisher=False)),  # fisher=False -> normal = 3
        int(len(r)),
        trial_metadata.get("tag"),
    )
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row
        )
        conn.executemany(
            "INSERT INTO trial_returns VALUES (?,?,?)",
            [(trial_id, _iso(d), float(v)) for d, v in r.items()],
        )
    return trial_id


def get_all_trials(
    tag: str | None = None, db_path: str | Path = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """All logged trials (optionally only those with a given tag)."""
    with _connect(db_path) as conn:
        if tag is None:
            df = pd.read_sql("SELECT * FROM trials ORDER BY created_at", conn)
        else:
            df = pd.read_sql(
                "SELECT * FROM trials WHERE tag = ? ORDER BY created_at",
                conn,
                params=(tag,),
            )
    if not df.empty:
        df["factors"] = df["factors"].map(json.loads)
        df["params"] = df["params"].map(json.loads)
    return df


def get_dsr_inputs(
    tag: str | None = None, db_path: str | Path = DEFAULT_DB_PATH
) -> dict:
    """The two multiple-testing inputs DSR needs, already in the right unit.

    Returns {"num_trials": N, "trial_sharpes": [per-period Sharpe, ...]}.
    Use this instead of pulling `sharpe_ratio` off get_all_trials() and
    dividing by hand — the conversion happens in exactly one place.
    """
    df = get_all_trials(tag, db_path)
    if df.empty:
        raise ValueError(f"no trials logged for tag={tag!r}")
    bad = df.loc[df["sharpe_basis"] != "annualized", "trial_id"].tolist()
    if bad:
        raise ValueError(f"unexpected sharpe_basis on trials {bad}")
    per_period = [
        sharpe_to_per_period(s, int(p))
        for s, p in zip(df["sharpe_ratio"], df["periods_per_year"])
    ]
    return {"num_trials": int(len(df)), "trial_sharpes": per_period}


def get_trial_returns(
    trial_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> pd.Series:
    """Weekly log returns for one trial, indexed by week-ending date."""
    with _connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT period_end_date, return_value FROM trial_returns "
            "WHERE trial_id = ? ORDER BY period_end_date",
            conn,
            params=(trial_id,),
        )
    if df.empty:
        raise KeyError(f"no returns logged for trial_id={trial_id!r}")
    s = pd.Series(
        df["return_value"].values, index=pd.to_datetime(df["period_end_date"])
    )
    s.index.name = "period_end_date"
    s.name = trial_id
    return s


def get_returns_matrix(
    trial_ids: list[str], db_path: str | Path = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Wide matrix (dates x trials) — the input shape compute_pbo() wants.

    Rows are aligned on date; a trial missing a date gets NaN there.
    """
    if not trial_ids:
        raise ValueError("trial_ids is empty")
    cols = [get_trial_returns(t, db_path) for t in trial_ids]
    return pd.concat(cols, axis=1).sort_index()
