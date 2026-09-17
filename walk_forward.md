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
    "folds": pd.DataFrame,   # one row per fold, all four metrics above
    "decay_ratio": float,     # see below
}
```

## Decay ratio

```
decay_ratio = average(out-of-sample Sharpe) / average(in-sample Sharpe)
```

| Decay ratio | Meaning |
|---|---|
| > 0.75 | Performance holds up well out of sample |
| 0.50 – 0.75 | Some decay — worth investigating |
| < 0.50 | Strategy likely doesn't generalize |

A ratio near 1.0 means the strategy performs about as well going forward
as it did in training. A ratio well below 0.5 is a strong overfitting
signal on its own, independent of what DSR or PBO say.

## Testing approach

Construct a synthetic return series with a known, deliberate performance
break partway through (e.g. genuine signal in the first half, pure noise
in the second) and confirm the decay ratio comes out low, as expected.
