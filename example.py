"""End-to-end example on SYNTHETIC data: log -> DSR + PBO -> walk-forward -> report.

Run:  python example.py
Uses a throwaway SQLite file (example_trials.db) in the current folder.
"""

import json
import os

import numpy as np
import pandas as pd

from overfitting_detector import (
    compute_pbo,
    deflated_sharpe_ratio,
    generate_health_report,
    get_all_trials,
    get_dsr_inputs,
    get_returns_matrix,
    get_trial_returns,
    log_trial,
    walk_forward_validate,
)

DB = "example_trials.db"
if os.path.exists(DB):
    os.remove(DB)

# 1. Pretend we ran 12 parameter variants of a momentum strategy over 8
#    years of weekly data. Eleven are noise; one has a small real edge.
rng = np.random.default_rng(42)
dates = pd.date_range("2016-01-08", periods=8 * 52, freq="W-FRI")
trial_ids = []
for i in range(12):
    edge = 0.003 if i == 7 else 0.0
    returns = pd.Series(rng.normal(edge, 0.02, len(dates)), index=dates)
    trial_ids.append(
        log_trial(
            {"factors": ["momentum"], "params": {"lookback_weeks": 4 + 4 * i}, "tag": "example"},
            returns,
            db_path=DB,
        )
    )

# 2. Pick the winner the way an over-eager researcher would: highest Sharpe.
trials = get_all_trials(tag="example", db_path=DB)
winner = trials.sort_values("sharpe_ratio").iloc[-1]
winner_returns = get_trial_returns(winner["trial_id"], db_path=DB)
print(f"Winner: lookback={winner['params']['lookback_weeks']}  annualized Sharpe={winner['sharpe_ratio']:.2f}")

# 3. Deflated Sharpe: correct for the 12 things we tried. get_dsr_inputs()
#    hands back per-period Sharpes, so no unit conversion happens here.
dsr = deflated_sharpe_ratio(winner_returns, **get_dsr_inputs(tag="example", db_path=DB))

# 4. PBO: does the in-sample winner hold up out of sample, across all splits?
pbo = compute_pbo(get_returns_matrix(trial_ids, db_path=DB), num_partitions=8)

# 5. Walk-forward: does the winner decay through time?
wf = walk_forward_validate(winner_returns, train_window="3Y", test_window="6M", step="6M")

# 6. One report for the dashboard.
report = generate_health_report(dsr, pbo, wf)
print(json.dumps(report["overall"], indent=2))
for name, m in report["metrics"].items():
    print(f"[{m['grade'].upper():6}] {m['verdict']}")
    for w in m["warnings"]:
        print(f"         warning: {w}")
