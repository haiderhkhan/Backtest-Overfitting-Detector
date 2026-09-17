# Limitations — read before believing any number

Blunt list of why every result from `examples/real_psx_run.py` is an
**upper bound**, not a forecast.

1. **Survivorship bias.** `data/universe.yaml` is today's liquid KSE-100
   subset applied to ten years of history. Companies that shrank, delisted
   or collapsed are not in it. Backtesting only the survivors inflates
   returns and hides drawdowns. The psxdata API has no point-in-time index
   membership, so this cannot be fixed with this data source alone.
2. **No transaction costs, slippage or borrow cost.** The engine trades at
   the weekly close for free. Weekly rebalancing of a long-short book on
   PSX would in reality cost tens of basis points per turn. Short-lookback,
   short-holding configs are the most flattered by this.
3. **PSX liquidity.** Many names trade thinly. The stale-price warnings in
   `data/returns.py` are the symptom; the disease is that fills at the
   printed close are fiction for illiquid stocks, and equal-weighting a
   thin name is not achievable at size.
4. **Short selling is largely theoretical on PSX.** Regulated short
   selling exists only for a limited eligible-scrips list with margin and
   uptick constraints, and stock borrow is scarce. The short leg of the
   long-short portfolio should be read as "what a hedge would have done",
   not as something you could have executed. A long-only variant is the
   honest next step.
5. **Thin history vs walk-forward folds.** Ten years of weekly data with
   3-year train / 6-month test windows yields roughly 14 folds; shorter
   trial series (long lookbacks) yield fewer. The decay ratio is a noisy
   average of noisy Sharpes. Warnings fire below 4 folds, but even 10 is
   not a lot.
6. **The universe is current-constituent and hand-picked.** Forty-odd
   names is enough to run the machinery, not enough for a credible
   quintile sort (top/bottom quintile = ~9 names each).
7. **Corporate actions.** Prices come as the API serves them. Whether they
   are adjusted for splits, bonus shares and rights issues is not
   documented; unadjusted prices create fake momentum breaks.
8. **One factor, one engine.** Momentum only. The detector will grade
   whatever you feed it; it cannot tell you the engine itself is crude.
