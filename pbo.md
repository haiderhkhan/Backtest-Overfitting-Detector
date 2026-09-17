# `pbo.py`

Source: Bailey, Borwein, López de Prado, Zhu, *"The Probability of
Backtest Overfitting"* (2017).

## The problem, in plain terms

DSR asks "is this one strategy's Sharpe ratio too good to be luck?" PBO
asks a related but different question: **if I'd only picked my "best"
strategy from half the data, would it still have been the best on the
other half?**

If the answer is usually no, your model-selection process — not just one
number — is the thing that's overfit.

## How CSCV works, step by step

1. Split the full return history into an even number of contiguous,
   non-overlapping chunks (default: 16).
2. Generate every way of splitting those chunks into two equal halves:
   an in-sample set **S** and its complementary out-of-sample set **S̄**.
3. For each split: on **S**, rank every candidate strategy by a
   performance metric (Sharpe ratio) and note the in-sample winner.
4. Take that same winner and check its rank on **S̄**.
5. If the winner lands *below the median* rank out-of-sample, that split
   counts as an overfitting instance.
6. **PBO = the fraction of all splits where that happens.**

## Interface

```python
compute_pbo(
    strategy_returns_matrix: pd.DataFrame,  # dates x strategies, wide format
    num_partitions: int = 16,
) -> dict
```

`strategy_returns_matrix` is exactly what `trial_log.get_returns_matrix()`
produces — this is why that function exists.

Returns:

```python
{
    "pbo": float,                    # 0-1 probability of overfitting
    "rank_distribution": list[float] # out-of-sample ranks, for plotting
}
```

## Reading the output

| PBO | Meaning |
|---|---|
| < 0.20 | Low overfitting risk |
| 0.20 – 0.40 | Moderate — worth a closer look |
| > 0.40 | High — the "best" strategy is likely an artifact of the search process itself |

These thresholds are configurable defaults in `report.py`, not hardcoded
law — see [`../ARCHITECTURE.md#4-metrics--thresholds`](../ARCHITECTURE.md).

## A note on which trials go in

PBO compares *candidates you were actually choosing between*, not
necessarily every trial ever logged (some logged trials might be broken
test runs). This is what the `tag` field on `trials` is for — you decide
which group of trial_ids belongs in one PBO run.

## Testing approach

Use a small set of synthetic strategies where you control which one is
"secretly best" on the full sample vs. which one wins by chance on
sub-samples, so the expected PBO direction is known before you run it.
