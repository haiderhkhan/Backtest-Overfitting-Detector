# Product Requirements Document — Backtest Overfitting Detector

## 1. Problem statement

Testing many factor/parameter combinations and reporting only the best result
inflates the apparent skill of a strategy — the "best" Sharpe ratio is partly
a function of how many times you looked, not just how good the strategy is.
There is currently no systematic check for this in the PSX factor-model
pipeline. This module is that check.

## 2. Goals

- Make "how many trials were run" an honest, logged fact — never an estimate.
- Grade a selected strategy on three independent statistical dimensions:
  selection bias (DSR), search-process overfitting (PBO), and time-decay
  (walk-forward).
- Produce one dashboard-ready verdict combining all three.

## 3. Non-goals

- This module does not build or select strategies itself.
- This module does not decide *for* Haider whether to deploy a strategy —
  it reports risk, it doesn't gate deployment.

## 4. Target user

Haider, building and validating a PSX long-short multi-factor model
(value, momentum, size, quality) — and, secondarily, anyone who later
reads the Streamlit "Backtest Health" panel he builds on top of this.

## 5. Functional requirements

| ID | Requirement | Acceptance criteria |
|---|---|---|
| FR1 | Log every backtest trial | Every call to `log_trial()` persists full metadata + weekly return series; nothing is lost on restart |
| FR2 | Compute Deflated Sharpe Ratio | `deflated_sharpe_ratio()` returns raw Sharpe, deflated Sharpe, and P(skill genuine), matching the Bailey & López de Prado (2014) formula |
| FR3 | Compute PBO via CSCV | `compute_pbo()` returns a PBO score and rank distribution, matching Bailey et al. (2017) |
| FR4 | Run walk-forward validation | `walk_forward_validate()` returns a fold-by-fold table and an aggregate decay ratio |
| FR5 | Produce a Health Report | `generate_health_report()` combines FR2–FR4 into one JSON-serializable object with pass/warn/fail verdicts |

## 6. Non-functional requirements

- No external paid APIs — operates purely on backtest output already produced
  by the factor-model pipeline.
- Every statistical method implemented from its source paper — not a wrapped
  black-box library — so the build is a learning exercise, not just a result.
- SQLite for now; schema simple enough to migrate to Postgres without a
  redesign if the PSX War Room stack needs it later.
- `report.py`'s output must be `json.dumps()`-safe with zero custom
  serialization logic required downstream.
- Every module ships with unit tests against synthetic data with a *known*
  expected result — not just "it runs."

## 7. Data requirements

Full schema: [`ARCHITECTURE.md#data-contract`](./ARCHITECTURE.md#2-data-contract).
Summary: two SQLite tables, `trials` (one row per run) and `trial_returns`
(one row per week per trial, log returns).

## 8. Success criteria / definition of done

- All five modules implemented and passing their unit tests.
- `generate_health_report()` output matches the documented JSON shape exactly.
- An end-to-end example (log → grade → report) runs on synthetic data with
  no manual intervention.
- The README Quick Start snippet runs as written, unmodified.

## 9. Out of scope

- Factor construction logic (value/momentum/size/quality) — separate,
  already-planned module.
- The Streamlit dashboard's visual design — only `report.py`'s data contract
  matters here.
- Live/paper trading execution — this module is backtest-validation only.

## 10. Risks & open questions

- The real backtest engine doesn't exist yet, so end-to-end integration
  (Phase 6) is speculative until it's built — see `phases.md`.
- Trial count is currently zero. DSR and PBO become meaningful once a real
  batch of trials accumulates, not before.
- Green/yellow/red thresholds are defaults from the literature and may need
  recalibrating once real PSX trial data exists.
