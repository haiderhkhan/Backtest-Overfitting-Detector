# STATUS — read this first when you come back

Last updated: 2026-09-18 (end of Claude Code session 9, Phase 11). Written for Haider
returning after ~3 weeks. Everything below is verified by `python -m pytest`
on that date.

## What is built and working

| File | Does | State |
|---|---|---|
| `overfitting_detector/metrics.py` | One shared Sharpe (per-period AND annualized, named by unit), drawdown, vol | done |
| `overfitting_detector/trial_log.py` | SQLite log of every trial; `get_dsr_inputs()` hands DSR per-period Sharpes | done |
| `overfitting_detector/deflated_sharpe.py` | DSR per Bailey & López de Prado; non-excess kurtosis; unit guard; `effective_num_trials()` by SIGNED correlation clustering (mirrors stay separate; `signed=False` for |corr|); reports `dsr_raw_n` AND `dsr_effective_n`, headline picked by explicit `num_trials_mode` (default raw) | done |
| `overfitting_detector/pbo.py` | CSCV / PBO with lambdas + omegas for plotting | done |
| `overfitting_detector/walk_forward.py` | Rolling IS/OOS folds + decay ratio; `n_folds` in result; warns decay_ratio unreliable below 5 folds | done |
| `overfitting_detector/report.py` | Green/yellow/red grading, plain-English verdicts, worst-of overall; PBO in 0.45-0.55 gets a separate `no_differentiation` diagnosis (IS ranking is noise), not an overfitting verdict | done |
| `example.py` | End-to-end on synthetic data | done |
| `dashboard/formatting.py` | Pure colour/number helpers (unit-tested) | done |
| `dashboard/health_panel.py` | Streamlit panel, presentation only, zero stats | done |
| `dashboard/mock_engine.py` | Fake engine — **this file IS the engine contract for now** | done |
| `dashboard/app.py` | Standalone demo, no real data | done |
| `data/psx_client.py` | psxdata REST client; key from `.env`; rate-limit + retry/backoff | done |
| `data/cache.py` | Parquet cache; `PSX_OFFLINE=1` forbids network; second run is fully offline | done |
| `data/universe.yaml` + `universe.py` | 45 liquid KSE-100 names, hand-picked (current constituents — survivorship!) | done |
| `data/returns.py` | Daily closes -> weekly log returns; warns on gaps/stale/late-listing/delisting/holidays | done |
| `engine/momentum.py` | `run_ranked_portfolio()` (shared by every factor) + momentum/reversal signal; long-short or long-only (benchmark-relative); flat `cost_bps` on turnover | done |
| `engine/value.py` | "Value" via a PRICE PROXY (long-horizon reversal) because psxdata has no fundamentals history; every trial carries `value_proxy` in params | done |
| `engine/sweep.py` | `GRIDS` keeps every grid (v1 90, v2 320, v3 160, v4_value 96, v5_combo 256 = v3+v4 in one tag); `factor` key dispatches to the engine; EVERY config logged; `sweep_tag(..., version=)` | done |
| `examples/compare_runs.py` | One health report per tag: Sharpe, effective_n, dsr_raw_n, dsr_effective_n, PBO, decay, n_folds; sorted by decay | done |
| `examples/real_psx_run.py` | Real PSX data -> sweep -> DSR + PBO + walk-forward -> report JSON; `--grid <key>` picks any grid in `GRIDS` | done |
| `docs/LIMITATIONS.md` | Why every real-data number is an upper bound | done |

Tests: 89 (58 core + 6 dashboard + 9 data layer + 16 engine). Data tests use a
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
python examples/real_psx_run.py --grid v4_value   # or v5_combo, v3, ... (any key of engine.sweep.GRIDS)
python examples/compare_runs.py           # every sweep tag side by side
```

Read `docs/LIMITATIONS.md` before believing the real-data report.

## Real results so far (calibration, not trading)

45 tickers, 2016-09 to 2026-09. Both sweeps live in `data/psx_trials.db`
under their own tags; `examples/compare_runs.py` prints this table.

Effective N is now SIGNED-correlation clustering (Phase 11). Under the
old |corr| rule it was 2 / 3 / 4 for v1 / v2 / v3; signed it is 2 / 5 / 5,
and the value and combo sweeps land at 8 and 14. Still small: every
config trades the same 45 names.

| sweep | trials | eff. N | in-sample winner | Sharpe | DSR raw N | DSR eff. N | PBO | PBO dx | decay | folds | grade |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 long-short momentum, no cost | 90 | 2 | mom lb 26w, hold 4w, quartiles | 0.17 | 0.03 | 0.51 | 0.24 | ok | -0.51 | 12 | FAIL |
| v2 + long_only x cost x direction | 320 | 5 | **reversal** lb 4w, hold 1w, long-short, **0 cost** | 0.87 | 0.05 | 0.84 | 0.14 | ok | 1.15 | 13 | FAIL |
| v3 momentum, executable only | 160 | 5 | mom lb 26w, hold 4w, quintiles, 25 bps | 0.34 | 0.00 | 0.15 | 0.53 | **no_differentiation** | -0.71 | 12 | FAIL |
| v4 value proxy, executable only | 96 | 8 | value lb 104w, skip 52, hold 4w, terciles, 25 bps | 0.77 | 0.06 | 0.50 | 0.63 | overfit | 0.63 | 7 | FAIL |
| v5 combo (v3 + v4 in one tag) | 256 | 14 | same value config as v4 | 0.77 | 0.01 | 0.28 | 0.40 | ok | 0.63 | 7 | FAIL |

Reading the five together:
- **Everything still fails DSR on the raw trial count.** The best Sharpe
  anyone found (0.87 zero-cost reversal, 0.77 value proxy) is about what
  the luckiest of a few hundred no-skill tries would show.
- **v3's PBO of 0.53 is now correctly diagnosed as "no differentiation"**:
  the momentum candidates are all the same bet, so the in-sample ranking
  carries no information. That is a different problem from v4's PBO of
  0.63, where the in-sample value winner reliably does WORSE out of sample.
- **v5 is what PBO was designed for**: genuinely different strategies in
  one set. PBO drops to 0.40 and effective N rises to 14, but the winner
  is still the same value config and DSR raw N is 0.01. Mixing factors
  made the comparison fairer; it did not make the winner real.
- **The value winner has only 7 walk-forward folds** (it needs 3 years of
  history before its first trade). Decay 0.63 is "yellow" on a thin base.

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
1b. **Effective-N convention DECIDED (2026-09-18): signed correlation.**
   distance = 1 - corr, threshold 0.5, average linkage. A strategy and its
   mirror are two selection opportunities, so they stay separate. `|corr|`
   remains available as `signed=False`. The 0.5 threshold is still a
   convention; nobody has justified it beyond "corr above 0.5 is the same
   bet".
1d. **"Value" is a price proxy, not fundamentals.** psxdata has no
   earnings/book history. `engine/value.py` ranks on multi-year past
   reversal and stamps `value_proxy: long_horizon_reversal` on every trial.
   Getting real fundamentals (annual reports, another vendor) is the only
   way to run an honest value factor here. See LIMITATIONS.md item 9.
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

The pipeline is complete end to end and every sweep so far fails
honestly. The bottleneck is now DATA, not machinery:
1. Find a source of historical PSX fundamentals (annual EPS / book value
   per share, even yearly) so `engine/value.py` can rank on a real
   earnings yield instead of the price proxy.
2. Find or build point-in-time KSE-100 membership to kill the
   survivorship bias in `data/universe.yaml`.
Either one is a bigger improvement to the credibility of every number
here than any further engine or grid change.
