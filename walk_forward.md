# `walk_forward.py`

## The problem, in plain terms

DSR and PBO both look backward at a fixed history. Walk-forward asks a
different question: **if I'd only known the past at each point in time,
would this strategy have kept working as time actually moved forward?**

A strategy that only works on one lucky static period is exactly what
walk-forward is built to catch.

## How it works

1. Pick a training window (e.g. 3 years) and a test window (e.g. 6 months).
2. Train/select on the training window, then measure performance on the
   immediately following test window — data the "selection" never saw.
3. Roll both windows forward by a step size (e.g. 6 months) and repeat
   across the full history.
4. For each fold, record: in-sample Sharpe, out-of-sample Sharpe,
   in-sample max drawdown, out-of-sample max drawdown.

With 5–10 years of PSX data and a 3-year train / 6-month test / 6-month
step setup, expect roughly 8–16 folds — enough for the decay ratio below
to mean something statistically.

## Interface

```python
walk_forward_validate(
    returns: pd.Series,
    train_window: str,  # e.g. "3Y"
    test_window: str,   # e.g. "6M"
    step: str,            # e.g. "6M"
) -> dict
```

Returns:

```python
{
    "folds": pd.DataFrame,       # columns: fold, is_start, is_end, oos_start, oos_end,
                                 #          is_sharpe, oos_sharpe, is_max_dd, oos_max_dd
    "decay_ratio": float | None, # see below; None when mean IS Sharpe is ~0
    "num_folds": int,
    "warnings": list[str],       # e.g. "only 2 folds: insufficient history"
}
```

A dict, not a tuple — same shape convention as every other module, so
`report.py` never has to remember positional order.

## Decay ratio

```
decay_ratio = average(out-of-sample Sharpe) / average(in-sample Sharpe)
```

| Decay ratio | Meaning |
|---|---|
| > 0.70 | Performance holds up well out of sample |
| 0.40 – 0.70 | Some decay — worth investigating |
| < 0.40 | Strategy likely doesn't generalize |

Guard: if mean IS Sharpe is ~0 the ratio is meaningless (dividing by
nothing) — return `None` and a warning instead of a giant number. Fewer
than ~4 folds → warn "insufficient history"; the report surfaces it.

A ratio near 1.0 means the strategy performs about as well going forward
as it did in training. A ratio well below 0.5 is a strong overfitting
signal on its own, independent of what DSR or PBO say.

## Testing approach

Construct a synthetic return series with a known, deliberate performance
break partway through (e.g. genuine signal in the first half, pure noise
in the second) and confirm the decay ratio comes out low, as expected.
