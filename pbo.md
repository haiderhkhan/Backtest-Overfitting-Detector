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

Per split, the maths is:

```
omega  = rank_OOS(n*) / (N + 1)        # relative rank, 1 = worst, in (0, 1)
lambda = ln( omega / (1 - omega) )      # logit; < 0 means below OOS median
```
A split is an "overfit instance" when `lambda < 0`. Splits are enumerated
with `itertools.combinations`; `num_partitions` must be even (raise if not).

Returns:

```python
{
    "pbo": float,               # overfit instances / total splits
    "lambdas": list[float],     # one per split — histogram this on the dashboard
    "omegas": list[float],      # one per split — raw relative OOS ranks
    "num_combinations": int,    # C(S, S/2), e.g. 12870 for S=16
    "warnings": list[str],      # e.g. "only 3 strategies: PBO not meaningful"
}
```

Guard: warn if fewer than ~5 strategies are compared, or chunks are shorter
than ~15 weeks. CSCV needs enough candidates and enough history per chunk to
be statistically meaningful, not just enough to run.

## Reading the output

| PBO | Meaning |
|---|---|
| < 0.20 | Low overfitting risk |
| 0.20 – 0.40 | Moderate — worth a closer look |
| > 0.40 | High — the "best" strategy is likely an artifact of the search process itself |

These thresholds are configurable defaults in `report.py`, not hardcoded
law — see [`ARCHITECTURE.md#4-metrics--thresholds`](./ARCHITECTURE.md#4-metrics--thresholds-configurable-defaults-not-hardcoded-law).

## A note on which trials go in

PBO compares *candidates you were actually choosing between*, not
necessarily every trial ever logged (some logged trials might be broken
test runs). This is what the `tag` field on `trials` is for — you decide
which group of trial_ids belongs in one PBO run.

## Testing approach

Use a small set of synthetic strategies where you control which one is
"secretly best" on the full sample vs. which one wins by chance on
sub-samples, so the expected PBO direction is known before you run it.
