# `report.py`

## Purpose

Everything else in this module produces one piece of the picture. This
one combines all three into a single object the Streamlit dashboard can
just read and display — no dashboard-side math required.

## Interface

```python
generate_health_report(
    dsr_result: dict,             # output of deflated_sharpe_ratio()
    pbo_result: dict,              # output of compute_pbo()
    walk_forward_result: dict,     # output of walk_forward_validate()
    thresholds: dict | None = None,
) -> dict
```

`thresholds` lets you override the defaults in
[`../ARCHITECTURE.md#4-metrics--thresholds`](../ARCHITECTURE.md) without
touching this module's code.

## Output shape

```python
{
    "generated_at": "2026-09-18T00:00:00Z",
    "trial_id": "abc123",
    "metrics": {
        "deflated_sharpe": {
            "value": 0.62,
            "status": "yellow",
            "verdict": "Some of this Sharpe ratio is likely selection bias."
        },
        "pbo": {
            "value": 0.18,
            "status": "green",
            "verdict": "Low risk this strategy is a search artifact."
        },
        "walk_forward_decay": {
            "value": 0.81,
            "status": "green",
            "verdict": "Performance held up well out of sample."
        }
    },
    "overall_status": "yellow",
    "overall_verdict": "Trustworthy with caveats — see deflated Sharpe."
}
```

## Rules for combining statuses

- `overall_status` is the **worst** of the three individual statuses
  (red beats yellow beats green) — one weak leg is enough to distrust
  the whole result.
- Each `verdict` string is a short, plain-English sentence, not a number
  restated — the dashboard shows the number separately.

## Why this shape

Every field is a plain string, float, or nested dict — no custom Python
objects — so `json.dumps()` on the output works with zero extra
serialization logic on the Streamlit side.

## Testing approach

Feed it hand-constructed `dsr_result` / `pbo_result` / `walk_forward_result`
dicts at each threshold boundary (just above/below green-yellow-red cutoffs)
and confirm the status labels and `overall_status` come out as expected.
