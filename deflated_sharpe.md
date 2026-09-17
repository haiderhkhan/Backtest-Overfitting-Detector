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
    "raw_sharpe": float,        # SR — observed Sharpe of the candidate
    "expected_max_sr": float,   # SR_0 — the best Sharpe luck alone would give you over N trials
    "sharpe_std_error": float,  # sigma_SR — how noisy SR is, corrected for skew/kurtosis
    "dsr": float,               # 0-1 probability the true Sharpe > 0
    "overfitting_gap": float,   # SR - SR_0
    "warnings": list[str],      # e.g. "N=1: no multiple-testing correction possible"
}
```

## The math (implement exactly this)

**Step 1 — standard error of the Sharpe estimate**

```
sigma_SR = sqrt( (1 - g3*SR + ((g4 - 1)/4) * SR^2) / (T - 1) )
```
`g3` = skewness, `g4` = **non-excess** kurtosis (normal = 3, not 0), `T` =
number of return observations. Use `scipy.stats.kurtosis(x, fisher=False)`.
Feeding excess kurtosis here is the single most common DSR implementation
bug — the code comment must say which convention is used.

**Step 2 — expected max Sharpe under "nobody has skill"**

```
SR_0 = sqrt(V) * [ (1 - gamma) * Z(1 - 1/N) + gamma * Z(1 - 1/(N*e)) ]
```
`V` = variance of Sharpe ratios across all N trials (from the trial log,
never guessed), `gamma` = Euler–Mascheroni ≈ 0.5772, `Z` = `scipy.stats.norm.ppf`.

**Step 3 — deflate**

```
DSR = Phi( (SR - SR_0) / sigma_SR )       # Phi = scipy.stats.norm.cdf
```

## How to read the output

`overfitting_gap = SR - SR_0` **is** the cost of how many times you tried.
If it's negative, your winner didn't even beat what pure luck would have
produced over N attempts.

`dsr` is the number that matters most: "given how many things I tried and
how non-normal these returns are, how confident am I the true Sharpe is
above zero?"

## Edge cases

- `N = 1` — no multiple-testing correction is possible. Return a warning;
  don't silently report DSR as plain SR significance.
- `T < 30` — skew/kurtosis estimates are junk on that little data. Warn.

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
