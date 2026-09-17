"""Standalone demo: mock engine -> trial log -> health report -> panel.

    streamlit run dashboard/app.py

Needs zero real data. Uses a throwaway SQLite file in a temp folder.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

# allow `streamlit run dashboard/app.py` from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.health_panel import render_health_panel  # noqa: E402
from dashboard.mock_engine import populate_mock_trials  # noqa: E402
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

TAG = "demo"


@st.cache_data(show_spinner="Running mock backtests and validation checks...")
def build_report(n_trials: int, years: int, seed: int, partitions: int) -> dict:
    db = str(Path(tempfile.mkdtemp()) / "demo_trials.db")
    ids = populate_mock_trials(db, TAG, n_trials=n_trials, years=years, seed=seed)
    trials = get_all_trials(tag=TAG, db_path=db)
    winner_id = trials.sort_values("sharpe_ratio").iloc[-1]["trial_id"]
    winner = get_trial_returns(winner_id, db_path=db)

    dsr = deflated_sharpe_ratio(winner, **get_dsr_inputs(tag=TAG, db_path=db))
    pbo = compute_pbo(get_returns_matrix(ids, db_path=db), num_partitions=partitions)
    wf = walk_forward_validate(winner, train_window="3Y", test_window="6M", step="6M")
    return generate_health_report(dsr, pbo, wf)


st.set_page_config(page_title="Backtest Health", layout="wide")
st.title("Backtest Health — demo on synthetic data")
st.caption("Mock engine: N parameter variants, one with a real edge. The best in-sample Sharpe is picked, then graded.")

with st.sidebar:
    st.header("Mock engine knobs")
    n_trials = st.slider("Trials tried", 2, 30, 12)
    years = st.slider("Years of weekly history", 2, 15, 8)
    partitions = st.select_slider("CSCV partitions", [4, 6, 8, 10, 12], value=8)
    seed = st.number_input("Random seed", 0, 9999, 42)

render_health_panel(build_report(n_trials, years, int(seed), partitions))
