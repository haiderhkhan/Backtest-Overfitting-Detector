"""Pure formatting helpers for the dashboard. No Streamlit, no statistics.

Everything here turns values that ALREADY exist in a health report dict
into strings and colours. Nothing is recomputed. Tested in
dashboard/tests/test_formatting.py.
"""

from __future__ import annotations

GRADE_COLORS = {
    "green": "#2e7d32",
    "yellow": "#f9a825",
    "red": "#c62828",
    "unknown": "#757575",
}

GRADE_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴", "unknown": "⚪"}

METRIC_TITLES = {
    "dsr": "Deflated Sharpe Ratio",
    "pbo": "Probability of Backtest Overfitting",
    "decay_ratio": "Walk-Forward Decay Ratio",
}


def grade_color(grade: str) -> str:
    """Hex colour for a grade string; grey for anything unexpected."""
    return GRADE_COLORS.get(grade, GRADE_COLORS["unknown"])


def grade_emoji(grade: str) -> str:
    return GRADE_EMOJI.get(grade, GRADE_EMOJI["unknown"])


def metric_title(name: str) -> str:
    return METRIC_TITLES.get(name, name)


def format_value(value, decimals: int = 2) -> str:
    """None -> 'n/a'; numbers -> fixed decimals; anything else -> str()."""
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def format_pct(value) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def collect_warnings(report: dict) -> list[tuple[str, str]]:
    """Every warning in the report as (metric name, message), in metric order."""
    out = []
    for name, m in report.get("metrics", {}).items():
        for w in m.get("warnings", []):
            out.append((name, w))
    return out


def has_insufficient_history(report: dict) -> bool:
    """True if any module flagged thin data. Text match on the messages the
    modules emit ('insufficient history', 'T=..', 'rows per chunk', ...)."""
    keys = ("insufficient history", "unreliable", "rows per chunk", "only ")
    return any(any(k in msg for k in keys) for _, msg in collect_warnings(report))


def threshold_caption(name: str, thresholds: dict) -> str:
    """One-line 'green < 0.20 | yellow <= 0.40 | red > 0.40' style caption."""
    band = thresholds[name]
    if name == "pbo":
        return f"green < {band['green']:.2f} · yellow ≤ {band['yellow']:.2f} · red > {band['yellow']:.2f}"
    return f"green > {band['green']:.2f} · yellow ≥ {band['yellow']:.2f} · red < {band['yellow']:.2f}"
