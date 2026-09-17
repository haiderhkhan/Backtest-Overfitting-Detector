# `deflated_sharpe.py`

Source: Bailey & López de Prado, *"The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting and Non-Normality"* (2014).

## The problem, in plain terms

A Sharpe ratio measures return per unit of risk. If you flip a fair coin
20 times, the odds of getting 15+ heads are low — but if 200 people each
flip a coin 20 times, someone almost certainly gets 15+ heads by pure luck.

Trying 200 factor/parameter combinations and keeping the best Sharpe ratio
is the same thing. The "best" number is inflated by how many times you
looked, not just by how good the strategy is. The standard Sharpe ratio
has no way to know how many times you looked — DSR fixes that.

## What DSR corrects for

1. **Number of trials (N)** — more trials tried → higher expected maximum
   Sharpe by chance alone → higher bar the observed Sharpe has to clear.
2. **Skewness and kurtosis of the strategy's own returns** — the Sharpe
   ratio's math assumes normally distributed returns. Financial returns
   usually aren't (fat tails, skew), so the significance test's standard
   error needs correcting.
3. **Variance of Sharpe ratios across all N trials** — needed to estimate
   what the *expected maximum* Sharpe ratio would be if none of the trials
   had any real skill (the null hypothesis).

## Interface

```python
deflated_sharpe_ratio(
    returns: pd.Series,       # the selected strategy's weekly log returns
    num_trials: int,           # N, from trial_log.get_all_trials()
    trial_sharpes: list[float] # Sharpe ratios of ALL N trials, for variance
) -> dict
```

Returns:

```python
{
    "raw_sharpe": float,        # standard, un-corrected Sharpe ratio
    "deflated_sharpe": float,   # DSR statistic
    "prob_skill_genuine": float # 0-1 probability the true Sharpe > 0
}
```

## How to read the output

Compare `raw_sharpe` to `deflated_sharpe` — the gap between them **is**
the overfitting cost of how many times you tried. A large gap means most
of your "great" Sharpe ratio came from trying many things, not from the
strategy actually being good.

`prob_skill_genuine` is the number that matters most: it's the deflated
version of "how confident are we this isn't just luck?"

## Required inputs (recap)

- Observed Sharpe ratio of the selected strategy
- N — number of trials, from the trial log, never estimated
- Variance of Sharpe ratios across all N trials
- Skewness and kurtosis of the selected strategy's own return series
- Number of return observations (track record length)

## Testing approach

Validate against a synthetic return series with a *known* Sharpe ratio and
zero skew/kurtosis (i.e. close to normal) — the DSR formula should reduce
to something hand-calculable in that special case, which is what
`tests/test_deflated_sharpe.py` checks against.
