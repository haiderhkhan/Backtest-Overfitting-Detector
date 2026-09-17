<div align="center">

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                    ║
║              B A C K T E S T   O V E R F I T T I N G              ║
║                                                                    ║
║                        D E T E C T O R                             ║
║                                                                    ║
║        Statistical honesty for PSX long-short factor models        ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════╝
```

<br>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Status](https://img.shields.io/badge/Status-Design%20Phase-orange?style=for-the-badge)

<br>

**2 SQLite tables · 3 independent validation checks · Zero unvalidated Sharpe ratios**

*Built for PSX long-short factor models (value, momentum, size, quality) · Lahore, Pakistan*

---

</div>

## 🤔 Wait... What Is This?

You tested 200 versions of a momentum strategy. One of them posted a Sharpe ratio of 2.1. Is it good, or did you just get lucky 200 times in a row and only remembered the win?

**This module is the thing that answers that question for you** — not with a gut feeling, but with three statistical tests from peer-reviewed finance papers, run against a permanent log of *every single backtest you ever ran*, not just the one you're proud of.

> **The golden rule:** if you tried more than one thing, the winner's raw Sharpe ratio is a lie until it's survived DSR, PBO, and walk-forward. No exceptions. No "just this once."

---

## 🏗️ Architecture — The 30-Second Version

```
┌───────────────────────────────────────────────────────────┐
│                     BACKTEST ENGINE                         │
│              (weekly log returns · not built yet)           │
└───────────────────────────┬───────────────────────────────┘
                             │
                      📒 trial_log.py
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
     ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
     │ Deflated   │    │   PBO /    │    │   Walk-    │
     │  Sharpe    │    │   CSCV     │    │  Forward   │
     │ (selection │    │ (does the  │    │ (does it   │
     │   bias)    │    │  winner    │    │  survive   │
     │            │    │  hold up?) │    │   time?)   │
     └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                     ┌───────▼───────┐
                     │  Health Report │
                     │   (report.py)  │
                     └───────┬───────┘
                             │
                    Streamlit dashboard
```

Full interfaces and the exact data schema: [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## 🕵️ The Four Checks

They log, they grade, they never let a flattering number slide.

| Check | Role | What It Does | What It Will Never Do |
|:---:|:---|:---|:---|
| 📒 **Trial Log** | The Record Keeper | Logs every backtest run — good, bad, or embarrassing | Let you "forget" a trial happened |
| 📉 **Deflated Sharpe** | The Skeptic | Corrects your best Sharpe ratio for how many times you looked | Let a flattering number go unchallenged |
| 🔀 **PBO / CSCV** | The Cross-Examiner | Checks whether your in-sample winner would've won blind | Trust a strategy that only wins once |
| ⏳ **Walk-Forward** | The Time Traveler | Rolls forward through real history to catch performance decay | Judge a strategy by one lucky period |

### Reproducibility, by design
Every trial is logged with its exact factors, params, and date range before a single stat is trusted. Same inputs, same `trial_id`, same answer — every time. No re-running a backtest five different ways until the log "agrees" with you.

---

## 🔥 Feature Highlights

<table>
<tr>
<td width="50%">

### 📒 Full Trial Ledger
Every backtest — full factor combo, single parameter tweak, rebalancing change — gets a permanent row in SQLite. This is what makes "how many times did you try?" a fact, not a guess.

### 🧮 Two-Table Data Contract
`trials` holds the summary card. `trial_returns` holds the raw weekly log-return series. Defined *before* the backtest engine exists, so the engine has to conform to it.

### 🎯 Papers, Not Vibes
DSR and PBO are implemented straight from Bailey & López de Prado's original papers — not a black-box "overfitting score" pulled from nowhere.

</td>
<td width="50%">

### 🚦 Configurable Thresholds
Green / yellow / red cutoffs for PBO, DSR, and walk-forward decay are parameters, not hardcoded law — tune them per strategy class if you need to.

### 📦 Dashboard-Ready JSON
`report.py` outputs one flat, `json.dumps()`-friendly object. Zero custom serialization logic needed on the Streamlit side.

### 🗄️ Local-First
SQLite, file-based, no server to run. Schema simple enough to migrate to Postgres later without a redesign.

</td>
</tr>
</table>

---

## 🧬 The Stack

```
Language:     Python 3.11+
Core libs:    numpy, pandas, scipy.stats, itertools
Storage:      SQLite (local-first, zero-config)
Methods:      Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
              Probability of Backtest Overfitting / CSCV (Bailey et al., 2017)
Consumer:     Streamlit "Backtest Health" dashboard panel
Coverage:     PSX long-short factor portfolios (value, momentum, size, quality)
Philosophy:   "Compute the doubt deterministically. Never let a lucky number pass unchecked."
```

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.11
```

### Installation

> Not published yet — this is the planned setup once the module ships.

```bash
git clone https://github.com/haiderhkhan/Backtest-Overfitting-Detector.git
cd Backtest-Overfitting-Detector
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Quick start (planned API)

```python
from overfitting_detector import trial_log, deflated_sharpe, pbo, walk_forward, report

trial_id = trial_log.log_trial(trial_metadata, returns=weekly_log_returns)

all_trials = trial_log.get_all_trials(tag="round1_finalists")
dsr = deflated_sharpe.deflated_sharpe_ratio(
    returns=trial_log.get_trial_returns(trial_id),
    num_trials=len(all_trials),
    trial_sharpes=all_trials["sharpe_ratio"].tolist(),
)

matrix = trial_log.get_returns_matrix(all_trials["trial_id"].tolist())
overfit = pbo.compute_pbo(matrix, num_partitions=16)

decay = walk_forward.walk_forward_validate(
    returns=trial_log.get_trial_returns(trial_id),
    train_window="3Y", test_window="6M", step="6M",
)

health = report.generate_health_report(dsr, overfit, decay)
```

---

## 📁 Project Structure

```
Backtest-Overfitting-Detector/
├── 📂 overfitting_detector/
│   ├── 📄 trial_log.py         # SQLite-backed logger for every backtest run
│   ├── 📄 deflated_sharpe.py    # DSR calculation
│   ├── 📄 pbo.py                # CSCV-based PBO calculation
│   ├── 📄 walk_forward.py       # Rolling window validator + decay metric
│   ├── 📄 report.py              # Aggregates everything into a Health Report
│   └── 📂 tests/
├── 📄 ARCHITECTURE.md            # Full system design + data contract
├── 📄 PRD.md · rules.md · phases.md · memory.md   # planning + continuity
├── 📄 trial_log.md               # Design doc per module ─┐
├── 📄 deflated_sharpe.md         #                        │
├── 📄 pbo.md                     #                        │
├── 📄 walk_forward.md            #                        │
├── 📄 report.md                  #                        ┘
└── 📄 README.md                  # You are here 👋
```

---

## 🚦 Metrics & Thresholds

| Metric | 🟢 Green | 🟡 Yellow | 🔴 Red |
|:---:|:---:|:---:|:---:|
| PBO | < 0.20 | 0.20 – 0.40 | > 0.40 |
| DSR (probability skill is genuine) | > 0.95 | 0.50 – 0.95 | < 0.50 |
| Walk-forward decay ratio | > 0.70 | 0.40 – 0.70 | < 0.40 |

Defaults, not law — override via the `thresholds` argument of `generate_health_report()`.

---

## 🧭 Roadmap

- [x] Define the two-table data contract
- [x] Write the full architecture doc
- [x] Write per-module design docs
- [ ] `trial_log.py` + tests
- [ ] `deflated_sharpe.py` + tests
- [ ] `pbo.py` + tests
- [ ] `walk_forward.py` + tests
- [ ] `report.py` + tests
- [ ] Wire up to the PSX factor-model backtest engine
- [ ] Streamlit "Backtest Health" dashboard panel

---

## 🧠 Philosophy

```
┌───────────────────────────────────────────────────────────┐
│                                                             │
│   "A backtest that looks great after one try means nothing. │
│    A backtest that survives DSR, PBO, and walk-forward —    │
│    after losing honestly along the way, too —               │
│    is the only kind worth trusting your capital to."        │
│                                                             │
│              — Backtest Overfitting Detector, design log    │
│                                                             │
└───────────────────────────────────────────────────────────┘
```

This isn't a strategy generator. It's a lie detector for the strategies you already built — and the whole point of building it from scratch is to actually understand *why* a backtest can lie to you, not just install a package that says it can't.

---

## 📚 Grounded In

- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio Management.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). *The Probability of Backtest Overfitting.* Journal of Computational Finance.

---

<div align="center">

### Built with ☕ and stubbornness in Lahore, Pakistan

<br>

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?style=flat-square&logo=python)
![Made with Pandas](https://img.shields.io/badge/Made%20with-Pandas-150458?style=flat-square&logo=pandas)
![No Shortcuts](https://img.shields.io/badge/Selection%20Bias-Not%20Today-red?style=flat-square)

<br>

*If this saved a strategy from shipping on a lucky backtest, give it a ⭐*

*If it didn't, at least you found out before your capital did.*

</div>
