"""Side-by-side health reports for every sweep tag in the trial DB.

    python examples/compare_runs.py            # table, sorted by decay ratio
    python examples/compare_runs.py --json     # full reports as JSON

Per tag: in-sample winner, its annualized Sharpe, DSR, PBO, walk-forward
decay ratio, and the overall grade. Nothing is recomputed differently from
real_psx_run.py; this just runs the same detector once per tag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def report_for_tag(tag: str, db_path: str = DB, num_partitions: int = 8) -> dict:
    trials = get_all_trials(tag=tag, db_path=db_path)
    winner = trials.sort_values("sharpe_ratio").iloc[-1]
    wr = get_trial_returns(winner["trial_id"], db_path=db_path)
    matrix = get_returns_matrix(trials["trial_id"].tolist(), db_path=db_path)
    dsr = deflated_sharpe_ratio(
        wr, **get_dsr_inputs(tag=tag, db_path=db_path), trial_returns_matrix=matrix
    )
    pbo = compute_pbo(matrix, num_partitions)
    wf = walk_forward_validate(wr, "3Y", "6M", "6M")
    rep = generate_health_report(dsr, pbo, wf)
    rep["tag"] = tag
    rep["num_trials"] = int(len(trials))
    rep["winner_params"] = winner["params"]
    rep["winner_sharpe_annualized"] = float(winner["sharpe_ratio"])
    return rep


def summary_table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        m = r["metrics"]
        rows.append({
            "tag": r["tag"],
            "trials": r["num_trials"],
            "winner": json.dumps(r["winner_params"], separators=(",", ":")),
            "sharpe": round(r["winner_sharpe_annualized"], 2),
            "effective_n": m["dsr"]["details"]["num_trials_effective"],
            "dsr_raw_n": round(m["dsr"]["details"]["dsr_raw_n"], 3),
            "dsr_effective_n": None if m["dsr"]["details"]["dsr_effective_n"] is None
                               else round(m["dsr"]["details"]["dsr_effective_n"], 3),
            "pbo": round(m["pbo"]["value"], 3),
            "pbo_dx": m["pbo"].get("diagnosis"),
            "decay": None if m["decay_ratio"]["value"] is None else round(m["decay_ratio"]["value"], 2),
            "n_folds": m["decay_ratio"]["details"]["n_folds"],
            "grade": r["overall"]["label"],
        })
    df = pd.DataFrame(rows)
    return df.sort_values("decay", ascending=False, na_position="last").reset_index(drop=True)


def main(db_path: str = DB, as_json: bool = False) -> None:
    tags = [t for t in get_all_trials(db_path=db_path)["tag"].dropna().unique()]
    if not tags:
        print("no tagged trials in", db_path)
        return
    reports = [report_for_tag(t, db_path) for t in tags]
    if as_json:
        print(json.dumps(reports, indent=2, default=str))
        return
    with pd.option_context("display.width", 200, "display.max_colwidth", 120):
        print(summary_table(reports).to_string(index=False))


if __name__ == "__main__":
    main(as_json="--json" in sys.argv)
