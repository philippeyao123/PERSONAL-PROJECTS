# Multi-Asset Alpha Factory

![CI](https://github.com/USERNAME/alpha-factory/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-19%20passing-brightgreen)

> Replace `USERNAME` in the CI badge URL with your GitHub handle after pushing.


A systematic cross-sectional alpha research platform: ingest a multi-asset
panel, compute a library of factors, combine them into a tradable signal,
backtest with walk-forward discipline and realistic costs, and — crucially —
subject the result to the statistical tests that separate genuine edge from
data-mined noise.

This repository is built to demonstrate **research process**, not to publish a
live trading signal. The headline questions it is designed to answer honestly
are: *is the backtest free of look-ahead and survivorship bias? Does the Sharpe
survive after costs? Does it survive correction for the number of trials? At
what AUM does the edge die?*

---

## Why this project is built the way it is

Most cross-sectional backtests on the open internet overstate performance for
three avoidable reasons. This project is structured to refuse all three:

1. **Look-ahead / survivorship bias.** Fundamentals are lagged to public-
   availability dates (`PITDataLoader`, default 90-day reporting lag), the
   universe retains delisted names, and at each rebalance `t` a factor sees
   only data up to `t`. There is a unit test (`test_factor_only_uses_past_data`)
   asserting that a factor's output at `t` is invariant to data after `t`.

2. **Gross Sharpe theatre.** Every headline number is reported **net** of a
   transaction-cost model: half-spread, square-root market impact
   (Almgren-style), and borrow on the short book. The metric that matters is
   net Sharpe and turnover-adjusted return, not the gross curve.

3. **Multiple-testing inflation.** When you try N factor/parameter configs and
   keep the best, the winning Sharpe is upward-biased. The pipeline computes
   the **Deflated Sharpe Ratio** (Bailey & López de Prado) and the
   **Probabilistic Sharpe Ratio**, correcting for the number of trials,
   skew, kurtosis, and sample length. A signal that does not clear DSR > 0.95
   is reported as inconclusive — including when that is the inconvenient
   answer.

The intent is to look like research from someone who already knows *why* most
backtests are wrong.

---

## Pipeline

```
PIT data  ->  factor library  ->  standardize / neutralize / combine
          ->  portfolio construction  ->  walk-forward backtest (net of costs)
          ->  diagnostics (perf, IC, DSR/PSR, capacity)
```

| Stage | Module | What it does |
|-------|--------|--------------|
| Data | `data/loader.py` | PIT-lagged panel, delisted names retained |
| Factors | `factors/library.py` | Momentum, reversal, low-vol, value, illiquidity |
| Combine | `factors/combiner.py` | Winsorize → sector-neutralize → z-score → equal/IC/ridge |
| Portfolio | `portfolio/construction.py` | Dollar-neutral quantile or rank book, position limits |
| Costs | `backtest/costs.py` | Spread + √-impact + borrow; turnover |
| Backtest | `backtest/engine.py` | Walk-forward, periodic rebalance, no look-ahead |
| Diagnostics | `diagnostics/metrics.py` | Sharpe/Sortino/Calmar, IC/IR, **DSR**, **PSR**, capacity |

---

## Quickstart

```bash
pip install -e ".[dev]"
python -m alpha_factory.pipeline      # runs on synthetic data with a planted signal
pytest                                # 14 tests
```

The synthetic generator embeds a weak but real cross-sectional signal so the
pipeline has something to find; the value factors correctly surface as the
highest-IC factors, which is a sanity check on the plumbing.

### On real equity data

A real loader (`data/equity.py`) pulls a liquid US large-cap universe via
yfinance, derives dollar-ADV (for capacity work), and caches to parquet:

```python
from alpha_factory.data.equity import EquityDataLoader
panel, dollar_adv = EquityDataLoader().load(start="2015-01-01")
```

It is explicit about its own limitation: a universe of *currently listed*
tickers is survivorship-biased, and the loader logs a warning to that effect.
The architecture accepts a point-in-time membership/delisting source (CRSP-
style) unchanged, because everything downstream consumes `PanelData`.

**Honest result on 51 large-caps, 2015–2024, simple price-based factors:**
net Sharpe ≈ 0.12, deflated Sharpe ≈ 0.06 — i.e. **no edge survives.** This is
the expected and instructive outcome: the synthetic demo has a planted signal,
real liquid large-caps with naive factors do not. Reporting the disappointing
number is the point.

## Critical replication study

[`REPLICATION_TSMOM.md`](REPLICATION_TSMOM.md) replicates Time-Series Momentum
(Moskowitz-Ooi-Pedersen 2012), stresses it with costs, and tests
out-of-sample persistence. Headline finding: a strong ~1.07 Sharpe in the
paper-era 2000s collapses to ~0.17 in the 2010s — the classic post-publication
decay. A heavily-cited factor degraded the moment it was traded live. Run it:

```python
from alpha_factory.diagnostics.tsmom_replication import (
    TimeSeriesMomentum, load_tsmom_proxies,
)
res = TimeSeriesMomentum().run(load_tsmom_proxies())
print(res.by_period)
```

---

## Sample output (synthetic)

```
Metric                       Gross         Net
----------------------------------------------
Sharpe                       1.687       1.534
Sortino                      3.144       2.864
Max drawdown                -0.065      -0.074
Mean turnover                0.414 (per rebal)

Deflated Sharpe Ratio (net)            : 0.953
Verdict                                : SURVIVES deflation (DSR > 0.95)
Estimated capacity                     : ~$46B AUM
```

> These numbers are on **synthetic data with a deliberately planted signal**.
> They demonstrate the machinery, not a tradable edge. On real data the
> honest expectation for a simple long-only-bias-free factor blend is a far
> lower net Sharpe — and the DSR is there precisely to say so.

---

## Design choices worth defending in interview

- **Light combiner on purpose.** Equal / IC / ridge, not a deep net. Heavy ML
  on a few hundred monthly cross-sections overfits; restraint here is a signal,
  not a gap.
- **Spearman IC.** Rank-based, robust to outliers and to the non-linearity of
  the factor→return map.
- **Sector neutralization** so the signal is a within-sector bet rather than an
  implicit sector tilt.
- **IC weighting uses a trailing window that excludes the current period**, so
  the live weighting is not contaminated by the period it is trying to predict.

## Limitations (stated, not hidden)

- Synthetic data has no regime shifts, no crowding, no realistic fundamental
  release calendar; real data will be harsher.
- The impact model is reduced-form; true capacity work needs per-name ADV.
- No alpha decay / signal-blending across horizons yet — natural next step.

## Roadmap

- [x] Real equity panel via yfinance with dollar-ADV + parquet cache
- [x] Critical replication of a published factor (TSMOM out-of-sample decay)
- [ ] Point-in-time index membership (CRSP-style) for true survivorship-free runs
- [ ] Per-factor orthogonalization & crowding diagnostics
- [ ] Mean-variance construction with turnover penalty as an alternative book
- [ ] Real fundamentals (Compustat-style) wired into the value factors

## License

MIT
