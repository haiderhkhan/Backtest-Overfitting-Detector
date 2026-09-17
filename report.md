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
[`ARCHITECTURE.md#4-metrics--thresholds`](./ARCHITECTURE.md#4-metrics--thresholds-configurable-defaults-not-hardcoded-law) without
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
            "verdict": "DSR = 0.62 — moderate: a 38% chance this Sharpe is luck plus selection, not skill.",
            "warnings": []
        },
        "pbo": {
            "value": 0.18,
            "status": "green",
            "verdict": "PBO = 0.18 — low risk: fewer than 1-in-5 splits saw the winner flop out of sample.",
            "warnings": []
        },
        "walk_forward_decay": {
            "value": 0.81,
            "status": "green",
            "verdict": "Decay = 0.81 — out-of-sample Sharpe kept ~80% of in-sample; held up well.",
            "warnings": ["insufficient_history: only 3 folds"]
        }
    },
    "overall_status": "yellow",
    "overall_verdict": "Trustworthy with caveats — driven by deflated Sharpe (yellow).",
    "thresholds_used": { "pbo": {...}, "dsr": {...}, "decay_ratio": {...} }
}
```

`thresholds_used` echoes the exact bands applied, so a dashboard reading an
old report knows what "yellow" meant at the time.

## Rules for combining statuses

- `overall_status` is the **worst** of the three individual statuses
  (red beats yellow beats green) — one weak leg is enough to distrust
  the whole result.
- Each `verdict` string is one plain-English sentence that includes the
  number **and** what it means in odds terms (see examples above).
- `overall_verdict` names which metric is driving the overall status.
- `warnings` from each module are passed through untouched; a metric with
  an `insufficient_history` warning keeps its colour but the dashboard can
  flag it as low-confidence.

## Why this shape

Every field is a plain string, float, or nested dict — no custom Python
objects — so `json.dumps()` on the output works with zero extra
serialization logic on the Streamlit side.

## Testing approach

Feed it hand-constructed `dsr_result` / `pbo_result` / `walk_forward_result`
dicts at each threshold boundary (just above/below green-yellow-red cutoffs)
and confirm the status labels and `overall_status` come out as expected.
