"""Backtest Health Report: one JSON-safe verdict for the dashboard.

Takes the three result dicts (DSR, PBO, walk-forward), grades each one
green / yellow / red against configurable thresholds, writes a one-line
plain-English verdict per metric, and picks the WORST of the three as the
overall verdict.
"""

from __future__ import annotations

import math

# These bands are CONVENTIONS, not laws. DSR 0.95 is the usual
# "statistically significant" line; 0.50 is a coin flip. PBO and decay
# bands come from common practice, not a theorem. Override per call via
# the `thresholds` argument once real trial data suggests better cutoffs.
DEFAULT_THRESHOLDS = {
    "pbo": {"green": 0.20, "yellow": 0.40},  # lower is better
    "dsr": {"green": 0.95, "yellow": 0.50},  # higher is better
    "decay_ratio": {"green": 0.70, "yellow": 0.40},  # higher is better
}

_SEVERITY = {"green": 0, "yellow": 1, "red": 2, "unknown": 1}
_LABEL = {"green": "PASS", "yellow": "CAUTION", "red": "FAIL", "unknown": "CAUTION"}


def _grade(value, band: dict, higher_is_better: bool) -> str:
    """Map a number onto green / yellow / red; None or NaN -> 'unknown'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    if higher_is_better:
        if value > band["green"]:
            return "green"
        return "yellow" if value >= band["yellow"] else "red"
    if value < band["green"]:
        return "green"
    return "yellow" if value <= band["yellow"] else "red"


def _pbo_verdict(pbo: float, grade: str) -> str:
    odds = f"about a 1-in-{round(1 / pbo)} chance" if pbo > 0 else "essentially no chance"
    return {
        "green": f"PBO = {pbo:.2f} — low risk: {odds} this result is a selection artifact.",
        "yellow": f"PBO = {pbo:.2f} — moderate risk: {odds} this result is a selection artifact, not real skill.",
        "red": f"PBO = {pbo:.2f} — high risk: {odds} the in-sample winner underperforms out of sample.",
    }[grade]


def _dsr_verdict(dsr: float, grade: str) -> str:
    return {
        "green": f"DSR = {dsr:.2f} — {dsr:.0%} probability the Sharpe is real after correcting for trials tried.",
        "yellow": f"DSR = {dsr:.2f} — only {dsr:.0%} confidence the Sharpe beats what luck alone would produce.",
        "red": f"DSR = {dsr:.2f} — {dsr:.0%}: the Sharpe is no better than the luckiest of the trials you ran.",
    }[grade]


def _decay_verdict(ratio, grade: str) -> str:
    if grade == "unknown":
        return "Decay ratio undefined — in-sample Sharpe is ~0 or no folds ran."
    return {
        "green": f"Decay ratio = {ratio:.2f} — out-of-sample keeps {ratio:.0%} of in-sample Sharpe; holds up through time.",
        "yellow": f"Decay ratio = {ratio:.2f} — out-of-sample keeps only {ratio:.0%} of in-sample Sharpe; noticeable decay.",
        "red": f"Decay ratio = {ratio:.2f} — out-of-sample performance largely disappears; likely curve-fit to history.",
    }[grade]


def _folds_as_records(folds) -> list[dict]:
    """DataFrame -> list of dicts with ISO date strings (JSON-safe)."""
    if folds is None or len(folds) == 0:
        return []
    out = folds.copy()
    for col in ("is_start", "is_end", "oos_start", "oos_end"):
        out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


def generate_health_report(
    dsr_result: dict,
    pbo_result: dict,
    walk_forward_result: dict,
    thresholds: dict | None = None,
) -> dict:
    """Combine the three checks into one JSON-serializable report.

    Output shape:
      {
        "overall": {"grade", "label", "driven_by", "verdict"},
        "metrics": {
          "dsr":         {"value", "grade", "verdict", "warnings", "details"},
          "pbo":         {...},
          "decay_ratio": {...},
        },
        "thresholds": <the bands actually used>,
      }
    Any module warning is surfaced under that metric's "warnings" so the
    dashboard can show an "insufficient history" note.
    """
    overrides = thresholds or {}
    th = {k: {**DEFAULT_THRESHOLDS[k], **overrides.get(k, {})} for k in DEFAULT_THRESHOLDS}

    pbo = float(pbo_result["pbo"])
    dsr = float(dsr_result["dsr"])
    decay = walk_forward_result.get("decay_ratio")

    pbo_grade = _grade(pbo, th["pbo"], higher_is_better=False)
    dsr_grade = _grade(dsr, th["dsr"], higher_is_better=True)
    decay_grade = _grade(decay, th["decay_ratio"], higher_is_better=True)

    metrics = {
        "dsr": {
            "value": dsr,
            "grade": dsr_grade,
            "verdict": _dsr_verdict(dsr, dsr_grade),
            "warnings": list(dsr_result.get("warnings", [])),
            "details": {
                "raw_sharpe": dsr_result["raw_sharpe"],
                "expected_max_sr": dsr_result["expected_max_sr"],
                "sharpe_std_error": dsr_result["sharpe_std_error"],
                "overfitting_gap": dsr_result["overfitting_gap"],
                "num_trials": dsr_result.get("num_trials"),
            },
        },
        "pbo": {
            "value": pbo,
            "grade": pbo_grade,
            "verdict": _pbo_verdict(pbo, pbo_grade),
            "warnings": list(pbo_result.get("warnings", [])),
            "details": {
                "num_combinations": pbo_result.get("num_combinations"),
                "num_strategies": pbo_result.get("num_strategies"),
                "lambdas": [float(x) for x in pbo_result.get("lambdas", [])],
            },
        },
        "decay_ratio": {
            "value": None if decay is None else float(decay),
            "grade": decay_grade,
            "verdict": _decay_verdict(decay, decay_grade),
            "warnings": list(walk_forward_result.get("warnings", [])),
            "details": {
                "num_folds": walk_forward_result.get("num_folds"),
                "folds": _folds_as_records(walk_forward_result.get("folds")),
            },
        },
    }

    worst_name = max(metrics, key=lambda k: _SEVERITY[metrics[k]["grade"]])
    worst_grade = metrics[worst_name]["grade"]
    label = _LABEL[worst_grade]
    return {
        "overall": {
            "grade": worst_grade,
            "label": label,
            "driven_by": worst_name,
            "verdict": f"{label}: worst check is {worst_name} — {metrics[worst_name]['verdict']}",
        },
        "metrics": metrics,
        "thresholds": th,
    }
