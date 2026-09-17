# STATUS — read this first when you come back

Last updated: 2026-09-18 (end of Claude Code session 6, Phase 8). Written for Haider
returning after ~3 weeks. Everything below is verified by `python -m pytest`
on that date.

## What is built and working

| File | Does | State |
|---|---|---|
| `overfitting_detector/metrics.py` | One shared Sharpe (per-period AND annualized, named by unit), drawdown, vol | done |
| `overfitting_detector/trial_log.py` | SQLite log of every trial; `get_dsr_inputs()` hands DSR per-period Sharpes | done |
| `overfitting_detector/deflated_sharpe.py` | DSR per Bailey & López de Prado; non-excess kurtosis; unit-mismatch guard | done |
| `overfitting_detector/pbo.py` | CSCV / PBO with lambdas + omegas for plotting | done |
| `overfitting_detector/walk_forward.py` | Rolling IS/OOS folds + decay ratio; "insufficient history" warnings | done |
| `overfitting_detector/report.py` | Green/yellow/red grading, plain-English verdicts, worst-of overall | done |
| `example.py` | End-to-end on synthetic data | done |
| `dashboard/formatting.py` | Pure colour/number helpers (unit-tested) | done |
| `dashboard/health_panel.py` | Streamlit panel, presentation only, zero stats | done |
| `dashboard/mock_engine.py` | Fake engine — **this file IS the engine contract for now** | done |
| `dashboard/app.py` | Standalone demo, no real data | done |
| `data/psx_client.py` | psxdata REST client; key from `.env`; rate-limit + retry/backoff | done |
| `data/cache.py` | Parquet cache; `PSX_OFFLINE=1` forbids network; second run is fully offline | done |
| `data/universe.yaml` + `universe.py` | 45 liquid KSE-100 names, hand-picked (current constituents — survivorship!) | done |
| `data/returns.py` | Daily closes -> weekly log returns; warns on gaps/stale/late-listing/delisting/holidays | done |
| `engine/momentum.py` | Crude cross-sectional momentum, long-short equal weight, no costs | done |
| `engine/sweep.py` | 90-config grid, EVERY config logged under one tag | done |
| `examples/real_psx_run.py` | Real PSX data -> sweep -> DSR + PBO + walk-forward -> report JSON | done |
| `docs/LIMITATIONS.md` | Why every real-data number is an upper bound | done |

Tests: 66 (44 core + 6 dashboard + 9 data layer + 7 engine). Data tests use a
JSON fixture only — zero network. Streamlit rendering is not tested on purpose.

## How to run

```bash
pip install -r requirements.txt          # core: numpy, pandas, scipy, pytest
pip install -r requirements-dev.txt      # + streamlit (dashboard only)

python -m pytest                          # all tests
python example.py                         # end-to-end demo, prints a report
streamlit run dashboard/app.py            # dashboard demo, sidebar knobs

cp .env.example .env                      # then put PSXDATA_API_KEY in it
python examples/real_psx_run.py           # real PSX: fetch+cache 45 tickers, 90 momentum configs, report
PSX_OFFLINE=1 python examples/real_psx_run.py   # re-run from data/cache/ with no network
```

Read `docs/LIMITATIONS.md` before believing the real-data report.

## First real result (2026-09-18, for calibration, not for trading)

45 tickers, 2016-09 to 2026-09, 90 configs. In-sample winner: lookback 26w,
skip 1w, hold 4w, quartiles, annualized Sharpe 0.17. Report: **FAIL** —
DSR 0.03 (red), PBO 0.24 (yellow), decay ratio -0.51 (red). The detector
did its job: a crude momentum sort on survivors with no costs still has no
edge worth the name, and the tools say so.

The core package imports without streamlit installed. Only `dashboard/`
needs it.

## Key conventions you will forget

- Returns are WEEKLY LOG RETURNS with a `DatetimeIndex`.
- The trial log stores ANNUALIZED Sharpe and says so in the `sharpe_basis`
  column. DSR math runs on PER-PERIOD Sharpe. `get_dsr_inputs()` is the only
  sanctioned bridge; never divide by sqrt(52) by hand.
- Kurtosis is NON-excess everywhere (normal = 3).
- All modules return dicts. Walk-forward is a dict, not a tuple.
- Thresholds live in `report.DEFAULT_THRESHOLDS` and are overridable per call.

## Open items and deferred decisions

1. **The real engine is crude.** `engine/momentum.py` satisfies the
   `dashboard/mock_engine.py` contract but has no transaction costs, no
   slippage, no borrow cost, and shorts names that cannot realistically be
   shorted on PSX. Next engine steps, in order: long-only variant, a flat
   per-trade cost, volume-based liquidity filter. `mock_engine.py` stays as
   the contract doc and the dashboard's demo source.
2. **Survivorship bias is baked into the universe.** `data/universe.yaml`
   is today's constituents. The psxdata API has no point-in-time index
   history. Either accept it (and say so) or source historical membership
   elsewhere.
3. **Corporate-action adjustment of API prices is undocumented.** Spot
   check a few known bonus/rights events before trusting long lookbacks.
4. **Threshold recalibration.** The green/yellow/red bands (PBO 0.20/0.40,
   DSR 0.95/0.50, decay 0.70/0.40) are conventions. Revisit once a few
   dozen real PSX trials are logged and you can see where honest strategies
   actually land.
5. **Reverse-direction unit mismatch is deliberately NOT guarded.** DSR
   raises if per-period inputs look annualized (any |Sharpe| > 1.0). It does
   not raise if annualized inputs look per-period (all tiny). Reason: tiny
   annualized Sharpes are legitimate for bad strategies, so there is no
   threshold that catches the bug without false alarms. `get_dsr_inputs()`
   makes the mistake hard to make in the first place.
6. **README Quick Start** was executed against real signatures on
   2026-09-18. If any function signature changes, re-run that block.
7. **Coverage is not measured.** Add `pytest-cov` if you want a number.
8. **Postgres migration** not started; schema is plain enough to port.
9. **graphify-out/** and **data/cache/** are git-ignored. A fresh clone needs one online run (45 API calls) before `PSX_OFFLINE=1` works.
10. **`.env` has a UTF-8 BOM** (saved by a Windows editor). The loader handles it; do not "fix" the file and break something else.
11. **graphify-out/** is git-ignored and regenerated by `graphify update .`.

## Pick this up first

Add a `long_only: bool` flag to `engine/momentum.py` and a flat
`cost_bps` per rebalance, re-run `examples/real_psx_run.py`, and compare
the two reports. That single change turns the current result from "a
theoretical long-short with free trading" into something a PSX account
could actually have done, and it is the smallest honest step forward.
