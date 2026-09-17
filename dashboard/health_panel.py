"""Streamlit view of one Backtest Health Report. Pure presentation.

RULE: zero statistics in this file. If a number is not already in the
report dict from report.generate_health_report(), it is not shown.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.formatting import (
    collect_warnings,
    format_value,
    grade_color,
    grade_emoji,
    has_insufficient_history,
    metric_title,
    threshold_caption,
)


def _banner(text: str, color: str) -> None:
    st.markdown(
        f'<div style="background:{color};color:white;padding:0.9rem 1.2rem;'
        f'border-radius:0.5rem;font-size:1.05rem;margin-bottom:1rem">{text}</div>',
        unsafe_allow_html=True,
    )


def _caution_banner(report: dict) -> None:
    warnings = collect_warnings(report)
    if not warnings:
        return
    title = "Insufficient history" if has_insufficient_history(report) else "Data warnings"
    lines = "".join(f"<li><b>{metric_title(m)}</b>: {w}</li>" for m, w in warnings)
    _banner(f"⚠️ <b>{title} — read the verdict with care.</b><ul>{lines}</ul>", grade_color("yellow"))


def _metric_card(name: str, m: dict, thresholds: dict) -> None:
    grade = m["grade"]
    st.markdown(
        f'<div style="border-left:6px solid {grade_color(grade)};padding:0.4rem 0.8rem">'
        f'<div style="font-size:0.85rem;color:#666">{metric_title(name)}</div>'
        f'<div style="font-size:1.8rem;font-weight:600">{grade_emoji(grade)} {format_value(m["value"])}</div>'
        f'<div style="font-size:0.9rem">{m["verdict"]}</div>'
        f'<div style="font-size:0.75rem;color:#888;margin-top:0.3rem">{threshold_caption(name, thresholds)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _pbo_histogram(lambdas: list[float]) -> None:
    st.subheader("PBO: out-of-sample rank of each in-sample winner")
    if not lambdas:
        st.info("No lambda values in the report.")
        return
    df = pd.DataFrame({"lambda": lambdas})
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("lambda:Q", bin=alt.Bin(maxbins=30), title="logit of relative OOS rank (λ)"),
        y=alt.Y("count()", title="splits"),
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#c62828", strokeWidth=2).encode(x="x:Q")
    st.altair_chart(bars + zero, use_container_width=True)
    st.caption("Left of the red line = the in-sample winner did WORSE than the median strategy out of sample. PBO is the share of splits on that side.")


def _walk_forward_table(m: dict) -> None:
    st.subheader("Walk-forward folds")
    folds = m["details"].get("folds") or []
    if not folds:
        st.info("No folds in the report.")
        return
    df = pd.DataFrame(folds)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Decay ratio = mean OOS Sharpe ÷ mean IS Sharpe = {format_value(m['value'])}")


def render_health_panel(report: dict) -> None:
    """Draw the whole panel from one report dict. Nothing is computed here."""
    _caution_banner(report)

    overall = report["overall"]
    _banner(
        f"{grade_emoji(overall['grade'])} <b>{overall['label']}</b> — driven by "
        f"<b>{metric_title(overall['driven_by'])}</b><br/>{overall['verdict']}",
        grade_color(overall["grade"]),
    )

    metrics = report["metrics"]
    cols = st.columns(3)
    for col, name in zip(cols, ("dsr", "pbo", "decay_ratio")):
        with col:
            _metric_card(name, metrics[name], report["thresholds"])

    st.divider()
    _pbo_histogram(metrics["pbo"]["details"].get("lambdas", []))
    st.divider()
    _walk_forward_table(metrics["decay_ratio"])

    with st.expander("Raw report JSON"):
        st.json(report)
