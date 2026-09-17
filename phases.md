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

- [ ] `trial_log.py`: `log_trial()`, `get_all_trials()`, `get_trial_returns()`, `get_returns_matrix()`
- [ ] `test_trial_log.py`

**Why first:** everything else reads from this. No log, no honest trial
count, no DSR, no PBO.

**Exit:** can log a synthetic trial and read it back correctly, including
the returns matrix in wide format for PBO.

## Phase 2 — Deflated Sharpe Ratio

- [ ] `deflated_sharpe.py`
- [ ] `test_deflated_sharpe.py` — validated against a hand-calculated,
  known-Sharpe, zero-skew synthetic series

**Exit:** DSR matches the hand-calculated expected value within tolerance.

## Phase 3 — PBO / CSCV

- [ ] `pbo.py`
- [ ] `test_pbo.py` — validated against synthetic strategies with a known
  "secretly best" strategy

**Exit:** PBO direction (high vs. low) matches the synthetic setup's
expected result.

## Phase 4 — Walk-forward validation

- [ ] `walk_forward.py` (incl. "insufficient history" and IS-Sharpe≈0 guards)
- [ ] `test_walk_forward.py` — validated against a synthetic series with a
  deliberate performance break

**Exit:** decay ratio correctly flags the deliberate break as decay.

## Phase 5 — Health report

- [ ] `report.py`
- [ ] `test_report.py` — threshold boundary tests (just above / below each
  cutoff)

**Exit:** `generate_health_report()` output matches the documented JSON
shape exactly.

## Phase 6 — Integration

- [ ] End-to-end example script (log → grade → report) on synthetic data
- [ ] README Quick Start snippet actually runs as written

**Exit:** a stranger could clone the repo and run the example without
asking a question.

## Phase 7 — Dashboard

- [ ] Streamlit "Backtest Health" panel reading `report.py`'s JSON output

**Exit:** panel renders the pass/warn/fail verdicts live.
