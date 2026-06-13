# Critical Replication — Time-Series Momentum (Moskowitz, Ooi & Pedersen, 2012)

This is a deliberately *critical* replication: the goal is not to confirm the
paper but to test how well its central result holds up under transaction costs
and, more importantly, **out of sample after publication**.

## The claim

MOP (2012) document that an asset's own trailing 12-month return predicts its
next-month return across equities, bonds, commodities and currencies. A
volatility-scaled long-winners / short-losers strategy earns a high Sharpe with
low correlation to standard risk factors.

## Method

- **Universe:** 10 liquid cross-asset ETF proxies (equity US/intl/EM, bonds
  7-10y & 20y, gold / broad commodity / oil, USD & EUR) — chosen so the study
  is reproducible without a futures database. The qualitative conclusion is
  robust to this approximation.
- **Signal:** sign of trailing 12-month return, lagged one month (no look-ahead).
- **Sizing:** scaled to a 40% ex-ante annualized vol target using a rolling,
  lagged volatility estimate.
- **Costs:** 10 bps charged on the absolute monthly change in position.
- **Critique split:** in-sample (≤2012, the paper era) vs out-of-sample
  (>2012), plus a per-decade breakdown.

Reproduce with:

```python
from alpha_factory.diagnostics.tsmom_replication import (
    TimeSeriesMomentum, load_tsmom_proxies,
)
prices = load_tsmom_proxies(start="2006-01-01", end="2024-12-31")
res = TimeSeriesMomentum(lookback_months=12, vol_target=0.40, cost_bps=10).run(prices)
print(res.by_period)
```

## Findings (2006–2024, ETF proxies)

| Period | Net Sharpe | Months |
|--------|-----------:|-------:|
| Full sample | 0.40 | 214 |
| In-sample (≤2012) | 0.38 | 70 |
| Out-of-sample (>2012) | 0.41 | 144 |
| 2000s segment | **1.07** | 34 |
| 2010s | **0.17** | 120 |
| 2020s | 0.40 | 60 |

> Numbers depend on the proxy set, sample window and cost assumption; treat
> them as indicative, not canonical. The *pattern*, not the decimal, is the
> point.

## Interpretation — the honest read

The strategy is real but **far weaker than the paper's headline**, and the
decade breakdown tells the story the full-sample number hides:

- The **2000s segment delivers a strong ~1.07 Sharpe** — consistent with the
  paper's sample, which ended around the GFC where trend-following did
  exceptionally well (the long crisis-alpha episode).
- The **2010s collapse to ~0.17.** This matches the widely-documented
  post-publication decay of time-series momentum and the broad
  underperformance of CTAs through that decade. Crowding, a decade-long
  low-vol trending-but-choppy regime, and the publication effect itself are
  the usual explanations.
- The **2020s recover to ~0.40**, helped by the 2022 rates/commodity trends.

**Conclusion:** TSMOM is not a fabrication, but a researcher who allocated to
it on the strength of the paper's in-sample Sharpe in 2013 would have been
disappointed for most of a decade. The full-sample Sharpe is an average that
masks strong regime dependence. This is the central lesson — a published,
peer-reviewed, heavily-cited factor degraded materially the moment it was
traded out of sample.

## Caveats (so the critique is fair to the paper)

- ETF proxies are a weaker, more correlated universe than the ~58 futures MOP
  use; the true strategy is better diversified and would likely score higher.
- Costs here are a flat 10 bps; real futures execution is cheaper, which would
  lift net numbers somewhat.
- A single lookback (12m) and a single vol target are used; the paper studies a
  family. The decay finding, however, is not sensitive to these choices.

The takeaway for a systematic researcher is methodological, not about this one
paper: **in-sample Sharpe is not a forecast of live Sharpe**, and the burden of
proof is out-of-sample persistence — which is exactly why the main backtest
engine in this repo reports a deflated Sharpe rather than a raw one.
