"""Real PSX momentum sweep, end to end, into the overfitting detector.

    python examples/real_psx_run.py            # first run: fetches + caches
    PSX_OFFLINE=1 python examples/real_psx_run.py   # later runs: cache only

Prints the Backtest Health Report as JSON. Trials go to data/psx_trials.db
(git-ignored). The tag encodes grid version + universe + date, so re-running
after a grid change never overwrites an old sweep. Compare tags with
examples/compare_runs.py.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.cache import PriceCache, fetch_universe  # noqa: E402
from data.returns import weekly_log_returns  # noqa: E402
from data.universe import load_universe  # noqa: E402
from engine.sweep import run_sweep, sweep_tag  # noqa: E402
from overfitting_detector import (  # noqa: E402
    compute_pbo,
    deflated_sharpe_ratio,
    generate_health_report,
    get_all_trials,
    get_dsr_inputs,
    get_returns_matrix,
    get_trial_returns,
    walk_forward_validate,
)

DB = str(ROOT / "data" / "psx_trials.db")


def main() -> dict:
    uni = load_universe()
    prices = fetch_universe(uni["symbols"], uni["start"], uni["end"], cache=PriceCache())
    rets = weekly_log_returns(prices, uni["start"], uni["end"])
    for w in rets["warnings"]:
        print("data warning:", w, file=sys.stderr)

    tag = sweep_tag(uni["name"], date.today().isoformat())
    sweep = run_sweep(rets["returns"], tag=tag, db_path=DB)
    for w in sweep["warnings"]:
        print("sweep warning:", w, file=sys.stderr)
    print(f"logged {len(sweep['trial_ids'])} trials under tag {tag!r}", file=sys.stderr)

    trials = get_all_trials(tag=tag, db_path=DB)
    winner = trials.sort_values("sharpe_ratio").iloc[-1]
    winner_returns = get_trial_returns(winner["trial_id"], db_path=DB)
    print(f"in-sample winner: {winner['params']}  annualized Sharpe {winner['sharpe_ratio']:.2f}", file=sys.stderr)

    dsr = deflated_sharpe_ratio(winner_returns, **get_dsr_inputs(tag=tag, db_path=DB))
    pbo = compute_pbo(get_returns_matrix(sweep["trial_ids"], db_path=DB), num_partitions=8)
    wf = walk_forward_validate(winner_returns, train_window="3Y", test_window="6M", step="6M")
    report = generate_health_report(dsr, pbo, wf)
    report["data_warnings"] = rets["warnings"]
    report["sweep_warnings"] = sweep["warnings"]
    report["winner_params"] = winner["params"]
    report["tag"] = tag
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
