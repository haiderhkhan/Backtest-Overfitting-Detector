# Backtest Overfitting Detector

A standalone validation layer for a Pakistan Stock Exchange (PSX) multi-factor
alpha model (value, momentum, size, quality). It sits between the backtest
engine and the dashboard, and answers one question:

> **How much should I actually trust this backtest result?**

## Why this exists

When you test many factor combinations, lookback windows, and parameter
choices, the "best" result you find is partly — sometimes mostly — luck. This
is called **backtest overfitting**, and it's one of the most common ways
quant strategies look great in a backtest and then fail in production.

This module doesn't build strategies. It grades them, honestly, using
peer-reviewed statistical methods instead of gut feeling.

## What it does

1. **Logs every backtest trial** — not just the winner — so the honesty of
   later checks depends on a real trial count, not a guess.
2. **Deflated Sharpe Ratio (DSR)** — corrects the observed Sharpe ratio for
   the number of trials run and the shape of the return distribution.
3. **Probability of Backtest Overfitting (PBO)** — via Combinatorially
   Symmetric Cross-Validation (CSCV), estimates the odds that the "best"
   in-sample strategy is an overfit artifact.
4. **Walk-forward validation** — rolling train/test windows across the full
   history, to check whether performance decays over time.
5. **Backtest Health Report** — one JSON-serializable object combining all
   of the above, with pass/warn/fail thresholds, for the Streamlit dashboard.

## Project philosophy

This is a from-scratch, learning-first build. A few ground rules:

- Nothing gets built without explicit sign-off — design first, code second.
- Every method is implemented from its original paper, not a black box.
- The data contract (below) is defined *before* the backtest engine exists,
  so the engine has to conform to it — not the other way around.

## Module structure

```
overfitting_detector/
├── __init__.py
├── trial_log.py        # SQLite-backed logger for every backtest run
├── deflated_sharpe.py   # DSR calculation (Bailey & López de Prado)
├── pbo.py               # CSCV-based PBO calculation
├── walk_forward.py      # Rolling window validator + decay metric
├── report.py             # Aggregates everything into a Health Report
└── tests/
    ├── test_deflated_sharpe.py
    ├── test_pbo.py
    ├── test_walk_forward.py
    └── test_trial_log.py
```

Design notes for each module live in [`docs/`](./docs), and the full system
design lives in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Data contract

Two SQLite tables are the source of truth for everything downstream:

- **`trials`** — one row per backtest run (params, date range, summary stats)
- **`trial_returns`** — one row per week per trial (the raw log-return series)

Full schema: see [`ARCHITECTURE.md#data-contract`](./ARCHITECTURE.md#data-contract).

## Tech stack

- Python 3.11+
- `numpy`, `pandas`, `scipy.stats`, `itertools`
- SQLite for the trial log (migration path to Postgres kept open)
- No external paid APIs — operates purely on backtest output already
  produced by the factor-model pipeline
- PSX market data (for the factor model side, not this module) via
  [psxdata](https://psxdata.mintlify.app)

## Status

🚧 Design phase — data contract and architecture defined, no code written yet.

## Author

Haider Hasan Khan ([@haiderhkhan](https://github.com/haiderhkhan)) — BS
Accounting & Finance, FAST-NUCES Lahore.
