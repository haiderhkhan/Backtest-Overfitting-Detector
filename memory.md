# Project Memory

This file is **not** a design doc — it's a running log. Its only job is to
make picking this project back up cheap, whether that's next week or after
mid-terms.

## How to use this file

After every Claude Code response, paste a short summary of what happened
under a new entry in the **Session Log** below. Keep entries short — what
changed, what was decided, what's next. Don't paste full code or full
responses.

## Decisions log

Durable decisions that shouldn't need re-litigating. Add to this list;
don't remove old entries.

| Date | Decision |
|---|---|
| 2026-09-18 | Return frequency: weekly |
| 2026-09-18 | Return type: log returns |
| 2026-09-18 | Risk-free rate: 0%, stored explicitly per trial (not hardcoded) |
| 2026-09-18 | Added optional `tag` field to `trials` for PBO comparison grouping |
| 2026-09-18 | Storage: SQLite now, Postgres-migratable later |
| 2026-09-18 | Design phase (all docs) completed before any code |
| 2026-09-18 | Assume 5–10 years of weekly history; guards warn (not raise) when thinner |
| 2026-09-18 | Thresholds: DSR red < 0.50 / yellow 0.50–0.95; decay red < 0.40 / yellow 0.40–0.70 |
| 2026-09-18 | Richer outputs: DSR returns SR_0, sigma_SR, gap; PBO returns lambdas + omegas |
| 2026-09-18 | All four modules return dicts (walk-forward is NOT a tuple) |
| 2026-09-18 | DSR kurtosis convention: non-excess (normal = 3) |
| 2026-09-18 | Per-module .md docs stay at repo root (no docs/ folder) |
| 2026-09-18 | Added `metrics.py` for shared Sharpe/drawdown; DSR consumes per-period (weekly) Sharpes, not annualized |
| 2026-09-18 | Sharpe unit is explicit everywhere: `sharpe_basis` column in DB, `sharpe_basis` arg on DSR, `get_dsr_inputs()` is the only sanctioned path from log to DSR |

## Open questions

Things not yet decided — resolve before the phase that needs them.

- Engine now real but crude (no costs, theoretical shorts). Next: long-only + cost_bps.
- Survivorship bias in universe.yaml; psxdata has no point-in-time index membership.
- Will the green/yellow/red thresholds need recalibrating once real trial data exists?

## Session log

Newest entry on top.

### 2026-09-18 — Phase 9: long_only / cost_bps / direction (Claude Code, session 7)
Engine gained long_only (benchmark-relative), flat cost_bps on turnover,
direction momentum|reversal. Sweep grid versioned (GRID_VERSION="v2",
320 configs, tag = version+universe+date; v1 tag kept). compare_runs.py
prints per-tag DSR/PBO/decay sorted by decay. 71 tests. v2 real run
(offline, ~6 min): winner = zero-cost weekly reversal, Sharpe 0.87, DSR
0.05, PBO 0.14, decay 1.15 -> still FAIL on DSR. Next: v3 grid = costs>0
and long-only only.

### 2026-09-18 — Phase 8: real PSX data + momentum engine (Claude Code, session 6)
psxdata REST API (base https://psxdata-api.fastapicloud.dev, OpenAPI at
psxdata.mintlify.app/openapi.json) works; key sent as X-API-Key though the
API did not actually reject unauthenticated calls. Built data/ (client,
parquet cache, universe.yaml with 45 names, weekly returns with PSX
warnings), engine/ (crude momentum, 90-config sweep logging every config),
examples/real_psx_run.py, docs/LIMITATIONS.md. 66 tests, zero network.
Real run: winner Sharpe 0.17 -> DSR 0.03 / PBO 0.24 / decay -0.51 = FAIL.
Offline rerun from cache verified. Gotcha: .env has a BOM; loader handles it.

### 2026-09-18 — Phase 7 dashboard + STATUS.md (Claude Code, session 4)
Added dashboard/ (formatting.py pure helpers + tests, health_panel.py
presentation-only, mock_engine.py = engine contract, app.py demo).
streamlit is a dev-only dep (requirements-dev.txt); core imports without
it. STATUS.md at root is the return-after-break handoff. All phases done.

### 2026-09-18 — Hardening pass: Sharpe unit safety (Claude Code, session 3)
metrics.py now has sharpe_per_period() / sharpe_annualized() (no ambiguous
default). trials table gained sharpe_basis + periods_per_year columns.
New trial_log.get_dsr_inputs(tag) returns DSR-ready per-period Sharpes.
deflated_sharpe_ratio() takes sharpe_basis ("per_period" | "annualized")
and converts in one place; per-period inputs > 1.0 raise as a unit
mismatch. README Quick Start rewritten and executed against real
signatures. requirements.txt added; graphify-out/ ignored. 43 tests green.

### 2026-09-18 — Phases 1–6 built (Claude Code, session 2)
Committed Phase 0 docs, then built the whole package in one pass:
`overfitting_detector/{metrics,trial_log,deflated_sharpe,pbo,walk_forward,report}.py`,
five test files (40 tests, all green), `example.py` end-to-end on synthetic
data, `.gitignore`. Added `metrics.py` (shared Sharpe/drawdown) so all
modules agree on one Sharpe definition. Walk-forward windows use a small
hand parser ("3Y"/"6M"/"26W") because pandas 3 dropped bare Y/M aliases.
DSR takes WEEKLY (per-period) Sharpes; example shows the /sqrt(52) step.
Next: Phase 6 README Quick Start check, then Phase 7 dashboard.

### 2026-09-18 — Spec reconciliation (Claude Code, session 1)
Read all existing docs. Confirmed: engine doesn't exist yet (we own the
contract), trial log starts at zero, assume 5–10y weekly history. Resolved
conflicts between the new brief and the docs — see decisions log. Updated
ARCHITECTURE (+ §4a Data sufficiency), README, all five module docs,
phases. No code written. Next: wait for "go" → Phase 1 `trial_log.py`.

### 2026-09-18 — Design phase complete
Defined the two-table data contract, wrote README (styled after the PSX War
Room v2 README), ARCHITECTURE.md, five per-module docs, PRD, rules, phases,
and this file. No code written yet. Next: mid-term break, then Phase 1
(`trial_log.py`) in Claude Code.
