# `trial_log.py`

## Purpose

This is the honesty layer. Every backtest you run — a full factor
combination, a single parameter tweak, a rebalancing frequency change —
gets a permanent row here, whether it looked good or not.

DSR and PBO both need to know "how many times did you actually try?" This
module is what makes that number real instead of a guess.

## Storage

SQLite, two tables — see [`ARCHITECTURE.md#data-contract`](./ARCHITECTURE.md#2-data-contract)
for the full schema:

- `trials` — one row per backtest run
- `trial_returns` — one row per week per trial

## Interface

```python
log_trial(trial_metadata: dict, returns: pd.Series) -> str
```
Inserts one row into `trials` and N rows into `trial_returns` (one per week
in the `returns` series). Returns the generated `trial_id`.

`trial_metadata` is expected to contain: `factors`, `params`, `start_date`,
`end_date`, `frequency`, `risk_free_rate` (defaults to `0.0` if omitted),
and optionally `tag`. Summary stats (`sharpe_ratio`, `annualized_return`,
`volatility`, `max_drawdown`, `skewness`, `kurtosis`, `num_observations`)
are computed from `returns` inside this function — you don't calculate
them yourself before calling it.

```python
get_all_trials(tag: str | None = None) -> pd.DataFrame
```
Returns the `trials` table as a DataFrame. Pass `tag` to filter to one
comparison group (e.g. `"round1_finalists"`).

```python
get_trial_returns(trial_id: str) -> pd.Series
```
Returns the weekly log-return series for one trial, indexed by
`period_end_date`. Used by `deflated_sharpe.py`.

```python
get_returns_matrix(trial_ids: list[str]) -> pd.DataFrame
```
Returns a wide DataFrame (dates as rows, one column per `trial_id`),
aligned on `period_end_date`. This is exactly the shape `pbo.py` needs.

## Why SQLite

File-based, no server to run, trivial to back up or commit alongside the
repo. The schema is simple enough that moving to Postgres later — if the
PSX War Room stack needs it — is a straight table copy, not a redesign.

## What "logging a trial" should feel like in practice

Every time the (future) backtest engine finishes a run, it hands its
output straight to `log_trial()` before anything else happens with it —
before you look at the Sharpe ratio, before you decide if it's worth
keeping. The log has to come first, or the honest-trial-count guarantee
breaks.
