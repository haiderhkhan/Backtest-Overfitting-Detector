"""Health report: threshold boundaries, worst-of logic, JSON safety."""

import json

import pandas as pd
import pytest

from overfitting_detector.report import DEFAULT_THRESHOLDS, generate_health_report


def _dsr(v):
    return {"dsr": v, "raw_sharpe": 0.2, "expected_max_sr": 0.1,
            "sharpe_std_error": 0.05, "overfitting_gap": 0.1, "num_trials": 5, "warnings": []}


def _pbo(v, warnings=()):
    return {"pbo": v, "lambdas": [0.5, -0.2], "omegas": [0.6, 0.45],
            "num_combinations": 2, "num_strategies": 5, "warnings": list(warnings)}


def _wf(v, folds=True):
    f = pd.DataFrame({
        "fold": [1], "is_start": [pd.Timestamp("2020-01-03")], "is_end": [pd.Timestamp("2021-12-31")],
        "oos_start": [pd.Timestamp("2022-01-07")], "oos_end": [pd.Timestamp("2022-06-24")],
        "is_sharpe": [1.0], "oos_sharpe": [0.8], "is_max_drawdown": [-0.1], "oos_max_drawdown": [-0.05],
    }) if folds else pd.DataFrame()
    return {"folds": f, "decay_ratio": v, "num_folds": len(f), "warnings": []}


@pytest.mark.parametrize("pbo,grade", [(0.19, "green"), (0.20, "yellow"), (0.40, "yellow"), (0.41, "red")])
def test_pbo_bands(pbo, grade):
    rep = generate_health_report(_dsr(0.99), _pbo(pbo), _wf(0.9))
    assert rep["metrics"]["pbo"]["grade"] == grade


@pytest.mark.parametrize("dsr,grade", [(0.96, "green"), (0.95, "yellow"), (0.50, "yellow"), (0.49, "red")])
def test_dsr_bands(dsr, grade):
    rep = generate_health_report(_dsr(dsr), _pbo(0.05), _wf(0.9))
    assert rep["metrics"]["dsr"]["grade"] == grade


@pytest.mark.parametrize("decay,grade", [(0.71, "green"), (0.70, "yellow"), (0.40, "yellow"), (0.39, "red"), (None, "unknown")])
def test_decay_bands(decay, grade):
    rep = generate_health_report(_dsr(0.99), _pbo(0.05), _wf(decay))
    assert rep["metrics"]["decay_ratio"]["grade"] == grade


def test_overall_is_worst_metric_and_names_driver():
    rep = generate_health_report(_dsr(0.99), _pbo(0.05), _wf(0.2))
    assert rep["overall"]["grade"] == "red"
    assert rep["overall"]["label"] == "FAIL"
    assert rep["overall"]["driven_by"] == "decay_ratio"
    assert "decay_ratio" in rep["overall"]["verdict"]


def test_all_green_passes():
    rep = generate_health_report(_dsr(0.99), _pbo(0.05), _wf(0.9))
    assert rep["overall"]["grade"] == "green" and rep["overall"]["label"] == "PASS"


def test_threshold_override_is_merged_not_replaced():
    rep = generate_health_report(_dsr(0.99), _pbo(0.25), _wf(0.9), thresholds={"pbo": {"green": 0.30}})
    assert rep["metrics"]["pbo"]["grade"] == "green"
    assert rep["thresholds"]["pbo"]["yellow"] == DEFAULT_THRESHOLDS["pbo"]["yellow"]
    assert rep["thresholds"]["dsr"] == DEFAULT_THRESHOLDS["dsr"]


def test_warnings_surface_and_output_is_json_safe():
    rep = generate_health_report(_dsr(0.9), _pbo(0.3, ["only 3 strategies"]), _wf(None, folds=False))
    assert rep["metrics"]["pbo"]["warnings"] == ["only 3 strategies"]
    assert rep["metrics"]["decay_ratio"]["details"]["folds"] == []
    json.dumps(rep)  # must not raise
    rep2 = generate_health_report(_dsr(0.9), _pbo(0.3), _wf(0.8))
    assert rep2["metrics"]["decay_ratio"]["details"]["folds"][0]["is_start"] == "2020-01-03"
    d = rep2["metrics"]["dsr"]["details"]
    assert d["num_trials_mode"] == "raw" and d["dsr_raw_n"] == 0.9 and d["dsr_effective_n"] is None
    assert rep2["metrics"]["decay_ratio"]["details"]["n_folds"] == 1
    json.dumps(rep2)
