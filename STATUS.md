# STATUS — read this first when you come back

Last updated: 2026-09-18 (end of Claude Code session 8, Phase 10). Written for Haider
returning after ~3 weeks. Everything below is verified by `python -m pytest`
on that date.

## What is built and working

| File | Does | State |
|---|---|---|
| `overfitting_detector/metrics.py` | One shared Sharpe (per-period AND annualized, named by unit), drawdown, vol | done |
| `overfitting_detector/trial_log.py` | SQLite log of every trial; `get_dsr_inputs()` hands DSR per-period Sharpes | done |
| `overfitting_detector/deflated_sharpe.py` | DSR per Bailey & López de Prado; non-excess kurtosis; unit guard; `effective_num_trials()` by correlation clustering; reports `dsr_raw_n` AND `dsr_effective_n`, headline picked by explicit `num_trials_mode` (default raw) | done |
| `overfitting_detector/pbo.py` | CSCV / PBO with lambdas + omegas for plotting | done |
| `overfitting_detector/walk_forward.py` | Rolling IS/OOS folds + decay ratio; `n_folds` in result; warns decay_ratio unreliable below 5 folds | done |
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
| `engine/momentum.py` | Cross-sectional momentum/reversal; long-short or long-only (benchmark-relative); flat `cost_bps` on turnover | done |
| `engine/sweep.py` | `GRIDS` dict keeps every grid ever run (v1 90, v2 320, v3 160); `GRID_VERSION` picks the default; EVERY config logged; tag = version+universe+date | done |
| `examples/compare_runs.py` | One health report per tag: Sharpe, effective_n, dsr_raw_n, dsr_effective_n, PBO, decay, n_folds; sorted by decay | done |
| `examples/real_psx_run.py` | Real PSX data -> sweep -> DSR + PBO + walk-forward -> report JSON | done |
| `docs/LIMITATIONS.md` | Why every real-data number is an upper bound | done |

Tests: 77 (49 core + 6 dashboard + 9 data layer + 13 engine). Data tests use a
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
PSX_OFFLINE=1 python examples/real_psx_run.py   # re-run from data/cache/ with no network (~6 min for 320 configs)
python examples/compare_runs.py           # every sweep tag side by side
```

Read `docs/LIMITATIONS.md` before believing the real-data report.

## Real results so far (calibration, not trading)

45 tickers, 2016-09 to 2026-09. Both sweeps live in `data/psx_trials.db`
under their own tags; `examples/compare_runs.py` prints this table.

| sweep | trials | eff. N | in-sample winner | Sharpe | DSR raw N | DSR eff. N | PBO | decay | folds | grade |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 long-short momentum, no cost | 90 | 2 | lb 26w, skip 1, hold 4w, quartiles | 0.17 | 0.03 | 0.51 | 0.24 | -0.51 | 12 | FAIL |
| v2 + long_only x cost x direction | 320 | 3 | **reversal**, lb 4w, hold 1w, terciles, long-short, **0 cost** | 0.87 | 0.05 | 0.93 | 0.14 | 1.15 | 13 | FAIL |
| v3 executable only (long-only, cost > 0) | 160 | 4 | momentum, lb 26w, skip 1, hold 4w, quintiles, 25 bps | 0.34 | 0.00 | 0.22 | 0.53 | -0.71 | 12 | FAIL |

Reading the three together:
- **v3 is the honest one** and it fails everything: DSR 0.00, PBO 0.53
  (worse than a coin flip), decay -0.71. Once you can only go long and must
  pay costs, this universe has no momentum or reversal edge to find.
- **v2's reversal winner** looks better on "effective N" DSR (0.93) but is
  a zero-cost weekly long-short, i.e. not executable. Its cost-paying twin
  is in the v2 tag if you want to look.
- **Effective N is tiny (2-4) for every sweep.** All configs trade the
  same 45 names, and momentum vs reversal are near-exact negatives, so
  |corr| clustering at 0.5 merges almost everything. That is arguably
  correct for multiple-testing (a strategy and its mirror are one bet you
  would have flipped) but it makes `dsr_effective_n` generous. The
  headline stays `dsr_raw_n`; `num_trials_mode="effective"` must be asked
  for explicitly.

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

1. **The engine is still crude.** Flat `cost_bps` is a stand-in for real
   PSX costs (brokerage + CVT + slippage vary by broker and name); 25 bps
   per side is a guess. No borrow cost on shorts, no market impact, no
   volume-based liquidity filter. `mock_engine.py` stays as the contract doc
   and the dashboard's demo source.
1b. **Effective-N clustering threshold (0.5 on 1 - |corr|) is a convention.**
   It collapses momentum and its mirror reversal into one cluster. If you
   think a strategy and its inverse are two bets, use signed correlation
   (distance = 1 - corr) instead; that is a one-line change in
   `effective_num_trials()` and should be a deliberate, documented choice.
1c. **v3 says there is nothing here.** The next engine question is not
   "more parameters" but "different signal or different universe": e.g.
   sector-neutral ranking, a volume/liquidity filter, or a broader,
   point-in-time universe. Any of those is a new `GRID_VERSION`.
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

Decide the effective-N convention (|corr| vs signed corr, threshold) and
write the decision into `deflated_sharpe.py`'s comment and this file. Then
stop tuning momentum: the v3 result is a clean negative. The next real
step is a second factor (value or size, from psxdata fundamentals) run
through the same sweep -> log -> detector pipeline, so the project has
one honest comparison across factors rather than three sweeps of one.
