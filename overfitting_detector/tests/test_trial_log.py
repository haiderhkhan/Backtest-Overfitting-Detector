"""Trial log round-trips: what goes in must come back out unchanged."""

import numpy as np
import pandas as pd
import pytest

from overfitting_detector import trial_log


def _weekly(n=104, seed=0, mean=0.002, sd=0.02):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    return pd.Series(rng.normal(mean, sd, n), index=idx)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "trials.db"


def test_log_and_read_back_summary(db):
    r = _weekly()
    tid = trial_log.log_trial(
        {"factors": ["momentum", "value"], "params": {"lookback": 9}, "tag": "r1"},
        r,
        db_path=db,
    )
    df = trial_log.get_all_trials(db_path=db)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["trial_id"] == tid
    assert row["factors"] == ["momentum", "value"]
    assert row["params"] == {"lookback": 9}
    assert row["num_observations"] == 104
    assert row["risk_free_rate"] == 0.0
    assert row["start_date"] == "2020-01-03"
    assert row["max_drawdown"] <= 0
    # kurtosis is stored NON-excess: a normal-ish sample sits near 3, not 0
    assert 2.0 < row["kurtosis"] < 4.5


def test_returns_round_trip_exactly(db):
    r = _weekly(n=30, seed=1)
    tid = trial_log.log_trial({"factors": ["size"], "params": {}}, r, db_path=db)
    back = trial_log.get_trial_returns(tid, db_path=db)
    np.testing.assert_allclose(back.values, r.values)
    assert list(back.index) == list(r.index)


def test_tag_filter_and_returns_matrix(db):
    ids = []
    for i in range(3):
        ids.append(
            trial_log.log_trial(
                {"factors": ["q"], "params": {"i": i}, "tag": "grp" if i < 2 else None},
                _weekly(seed=i),
                db_path=db,
            )
        )
    assert len(trial_log.get_all_trials(tag="grp", db_path=db)) == 2
    assert len(trial_log.get_all_trials(db_path=db)) == 3
    m = trial_log.get_returns_matrix(ids, db_path=db)
    assert m.shape == (104, 3)
    assert list(m.columns) == ids


def test_explicit_risk_free_rate_is_stored_and_used(db):
    r = _weekly()
    tid_a = trial_log.log_trial({"factors": [], "params": {}, "risk_free_rate": 0.0}, r, db_path=db)
    tid_b = trial_log.log_trial({"factors": [], "params": {}, "risk_free_rate": 0.10}, r, db_path=db)
    df = trial_log.get_all_trials(db_path=db).set_index("trial_id")
    assert df.loc[tid_b, "risk_free_rate"] == 0.10
    assert df.loc[tid_b, "sharpe_ratio"] < df.loc[tid_a, "sharpe_ratio"]


def test_rejects_bad_input(db):
    with pytest.raises(ValueError):
        trial_log.log_trial({"factors": []}, _weekly(), db_path=db)
    with pytest.raises(ValueError):
        trial_log.log_trial({"factors": [], "params": {}}, pd.Series(dtype=float), db_path=db)
    with pytest.raises(KeyError):
        trial_log.get_trial_returns("nope", db_path=db)


def test_dsr_inputs_are_per_period_and_match_stored_unit(db):
    from overfitting_detector.deflated_sharpe import deflated_sharpe_ratio
    for i in range(4):
        trial_log.log_trial({"factors": [], "params": {}, "tag": "g"}, _weekly(seed=i), db_path=db)
    df = trial_log.get_all_trials(tag="g", db_path=db)
    assert (df["sharpe_basis"] == "annualized").all()
    assert (df["periods_per_year"] == 52).all()
    inputs = trial_log.get_dsr_inputs(tag="g", db_path=db)
    assert inputs["num_trials"] == 4
    np.testing.assert_allclose(inputs["trial_sharpes"], df["sharpe_ratio"] / np.sqrt(52))
    # helper output plugs straight into DSR and agrees with the annualized path
    r = trial_log.get_trial_returns(df["trial_id"].iloc[0], db_path=db)
    a = deflated_sharpe_ratio(r, **inputs)
    b = deflated_sharpe_ratio(r, 4, df["sharpe_ratio"].tolist(), sharpe_basis="annualized")
    assert a["dsr"] == pytest.approx(b["dsr"])
    with pytest.raises(ValueError):
        trial_log.get_dsr_inputs(tag="missing", db_path=db)


def test_db_file_is_released_after_use(db):
    trial_log.log_trial({"factors": [], "params": {}}, _weekly(), db_path=db)
    trial_log.get_all_trials(db_path=db)
    db.unlink()  # raises PermissionError on Windows if a handle is still open
