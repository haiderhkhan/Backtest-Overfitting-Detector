# Build Phases

Each phase has a clear exit condition. Don't start the next phase until the
current one's boxes are checked.

## Phase 0 — Design & planning ✅ Complete (Sep 18, 2026)

- [x] Data contract (`trials` + `trial_returns` schema)
- [x] Full architecture doc
- [x] Per-module design docs (repo root)
- [x] README
- [x] PRD, rules, phases, memory docs

**Exit:** every doc above exists and Haider has reviewed it.

## ⏸ Break — mid-term exams

Build work paused here. Resumes in Claude Code once exams are done.

## Phase 1 — Trial log foundation

- [x] `trial_log.py`: `log_trial()`, `get_all_trials()`, `get_trial_returns()`, `get_returns_matrix()`
- [x] `test_trial_log.py`

**Why first:** everything else reads from this. No log, no honest trial
count, no DSR, no PBO.

**Exit:** can log a synthetic trial and read it back correctly, including
the returns matrix in wide format for PBO.

## Phase 2 — Deflated Sharpe Ratio

- [x] `deflated_sharpe.py`
- [x] `test_deflated_sharpe.py` — validated against a hand-calculated,
  known-Sharpe, zero-skew synthetic series

**Exit:** DSR matches the hand-calculated expected value within tolerance.

## Phase 3 — PBO / CSCV

- [x] `pbo.py`
- [x] `test_pbo.py` — validated against synthetic strategies with a known
  "secretly best" strategy

**Exit:** PBO direction (high vs. low) matches the synthetic setup's
expected result.

## Phase 4 — Walk-forward validation

- [x] `walk_forward.py` (incl. "insufficient history" and IS-Sharpe≈0 guards)
- [x] `test_walk_forward.py` — validated against a synthetic series with a
  deliberate performance break

**Exit:** decay ratio correctly flags the deliberate break as decay.

## Phase 5 — Health report

- [x] `report.py`
- [x] `test_report.py` — threshold boundary tests (just above / below each
  cutoff)

**Exit:** `generate_health_report()` output matches the documented JSON
shape exactly.

## Phase 6 — Integration

- [x] End-to-end example script (log → grade → report) on synthetic data
- [x] README Quick Start snippet actually runs as written

**Exit:** a stranger could clone the repo and run the example without
asking a question.

## Phase 7 — Dashboard

- [x] Streamlit "Backtest Health" panel reading `report.py`'s JSON output

**Exit:** panel renders the pass/warn/fail verdicts live.

## Phase 8 — Real PSX data + momentum engine ✅ Complete (Sep 18, 2026)

- [x] `data/` psxdata client, offline parquet cache, universe config, weekly returns
- [x] `engine/` crude momentum + 90-config sweep, every config logged
- [x] `examples/real_psx_run.py` end to end, re-runnable offline
- [x] `docs/LIMITATIONS.md`
- [x] Tests on fixtures only (no network)

**Exit:** real report printed (FAIL, driven by DSR). Next: long-only + costs.

## Phase 9 — Engine realism knobs + cross-run comparison ✅ Complete (Sep 18, 2026)

- [x] `long_only`, `cost_bps`, `direction` in `engine/momentum.py`
- [x] v2 grid (320 configs), versioned tags, v1 tag preserved
- [x] `examples/compare_runs.py`
- [x] 71 tests; v2 real run compared against v1

**Exit:** both tags FAIL on DSR. Next: v3 grid restricted to executable configs.

## Phase 10 — Effective trial count + executable-only sweep ✅ Complete (Sep 18, 2026)

- [x] `effective_num_trials()` by correlation clustering; `dsr_raw_n` + `dsr_effective_n`
- [x] `n_folds` in walk-forward result, warning below 5
- [x] v3 grid (long-only, cost > 0), v1/v2 tags preserved
- [x] `compare_runs.py` shows effective_n / both DSRs / n_folds
- [x] 77 tests

**Exit:** v3 FAILS every check. Momentum on this universe is a clean negative.
