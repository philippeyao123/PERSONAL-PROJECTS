# Commodity Trading Signal Generator

Systematic multi-signal trading strategy on a 15-asset commodity futures universe (energy, metals, agriculture, softs), with volatility-targeted portfolio construction, transaction-cost-aware execution, and a full performance attribution framework.

**Stack:** Python, NumPy, pandas, SciPy, Matplotlib, yfinance.

---

## TL;DR — Results (2010–2026, net of 3 bps costs)

| Metric | Value |
|---|---|
| Annualised return | +0.8% |
| Annualised volatility | 10.3% (target: 10%) |
| **Sharpe ratio (net)** | **+0.08** (SE ≈ 0.25, Lo 2002) |
| Max drawdown | −38.5% |
| Sharpe 2011–2015 | +0.30 |
| Sharpe 2015–2020 | −0.29 (the "trend winter") |
| **Sharpe 2020–present** | **+0.50** |

**The honest headline:** over the full sample, the composite is statistically indistinguishable from zero — and that is precisely what the post-publication literature on commodity factors predicts for this period. The strategy clearly captures the post-2020 trend revival (Sharpe ~0.5), and clearly suffers the well-documented 2015–2019 trend winter. No parameter was tuned to hide either regime.

---

## Signal library

All signals output scores in [−1, 1], computed strictly on data available at *t*, executed at *t+1*.

| Sleeve | Weight | Construction | Reference |
|---|---|---|---|
| **TSMOM** | 40% | sign(12-month return) — canonical specification | Moskowitz, Ooi & Pedersen (2012, JFE) |
| **Donchian breakout** | 20% | price position in 55-day channel, rescaled to [−1, 1] | CTA practice |
| **Short-term reversal** | 15% | cross-sectional rank of vol-adjusted 5-day returns, sign-flipped | Nagel (2012, RFS) — liquidity provision |
| **Skewness premium** | 15% | short positive-skew, long negative-skew (12m rolling, XS rank) | Fernandez-Perez, Frijns, Fuertes & Miffre (2018, JBF) |
| **XSMOM 12m−1m** | 10% | cross-sectional momentum rank, last month skipped | Miffre & Rallis (2007, JBF) |

Cross-sectional sleeves are dollar-neutral by construction (centred ranks).

## Portfolio construction (standard CTA stack)

1. **Inverse-volatility raw book** — w_i ∝ signal_i / σ_i (EWMA, 63d), so each asset contributes comparable risk per unit of signal.
2. **Single scaling step to target vol** — σ_p(t) = √(wᵀ Σ_t w) with Σ_t an EWMA (RiskMetrics, λ = 0.97) covariance matrix; weights scaled to a 10% annualised portfolio vol. No trading during the 1-year covariance warm-up.
3. **Safety caps** — per-asset (±0.6× notional) and gross leverage (5×), sized to bind only exceptionally; they are *not* the sizing mechanism.
4. **No-trade band (1% notional)** — trade toward target only when the deviation exceeds the band (simplified Gârleanu & Pedersen, 2013). Cuts turnover ~45% (52× → 24× gross p.a.) while preserving fast-signal alpha.
5. **Execution** — t+1 lag, 3 bps one-way costs on effective turnover.

## Design decisions that came from the data (debugging journal)

This project deliberately documents its own failure modes — each was diagnosed layer-by-layer, not patched blindly:

1. **Negative WTI (20 Apr 2020, −$37.63).** Yahoo front-month splices the actual print; log-returns silently NaN and percentage returns through negative prices are meaningless. Non-positive prices are masked and forward-filled.
2. **Mean-reversion vs trend cancellation.** A first 50-day contrarian z-score sleeve had a **−0.91 PnL correlation** with the Donchian sleeve — the two legs netted each other out. Replaced with a genuinely short-horizon (5-day) cross-sectional reversal: correlation to trend falls to −0.4 and standalone gross Sharpe is +0.25.
3. **Realized-proxy vol targeting is flawed.** Estimating portfolio vol from the realized vol of a *changing* book cost ~0.34 Sharpe: when signals rotate, the proxy no longer represents the current book. Replaced with proper ex-ante √(wᵀΣw).
4. **Weekly rebalancing destroys fast alpha.** Weekly freezing (any weekday) cost ~0.15–0.25 Sharpe vs daily — the 5-day reversal sleeve decays intraweek. Daily targets + a no-trade band recovers the alpha at acceptable turnover.
5. **t-stat TSMOM underperforms sign TSMOM here** (~0.10 vs ~0.17 net standalone). The canonical Moskowitz et al. sign specification was retained — simpler and more robust on this sample.

## Carry / basis factor (extension)

The most documented commodity premium (Gorton & Rouwenhorst 2006; Erb & Harvey 2006; Koijen, Moskowitz, Pedersen & Vrugt 2018): long backwardated curves, short contangoed ones. Free data sources delist expired contracts, so the historical sleeve cannot be backtested without paid curve data — the module (`src/carry.py`) therefore ships in two parts:

1. **Live signal** — the full term structure of every commodity is rebuilt from individually listed Yahoo contracts (month-code chains per exchange: NYM/CMX/CBT/NYB). Carry is measured on a **12-month seasonal pair** (same calendar month, one year apart), which neutralises curve seasonality — essential for natural gas (the winter hump reaches +40% of the front price mid-curve) and agriculture, where a raw calendar slope conflates seasonality with the risk premium. Fallback: regression slope of ln(F) on maturity. Output: `output/current_carry.csv` + `output/carry_snapshot.png` (carry ranking + live curves).
2. **Plug-ready backtest** — `carry_signal_from_history(f1, f2, dt_years)` produces the dollar-neutral cross-sectional sleeve from any F1/F2 history (Nasdaq Data Link CHRIS, Bloomberg `CL1/CL2 Comdty`, Refinitiv). Wire it into `build_composite` via config once curve history is available.

Snapshot at the time of writing: the entire petroleum complex (HO, RB, CL, BZ) sits in strong backwardation (+15–18% annualised carry) while precious metals and grains are in contango — textbook cost-of-carry.

## Limitations (read before quoting the Sharpe)

- **Front-month continuous series from Yahoo are not properly roll-adjusted.** Splice gaps contaminate returns; a production system would build back-adjusted series from individual contract data (e.g. ratio-adjusted at roll dates).
- **The carry sleeve is live-only** — its historical contribution to the composite is not measurable without paid curve data (see above).
- **No hedging-pressure factor** (CFTC COT positioning) for the same data-access reason.
- Costs are flat 3 bps; a finer model would differentiate by contract (NG ≠ GC) and include roll costs.
- Sleeve weights are informed judgment, not optimised — with SE(Sharpe) ≈ 0.25 over 16 years, any in-sample weight optimisation would be fitting noise.

## Project structure

```
commodity-signal-generator/
├── main.py                 # End-to-end pipeline
├── config.py               # Universe, signal & portfolio parameters
├── requirements.txt
├── src/
│   ├── data_loader.py      # yfinance + synthetic offline fallback
│   ├── signals.py          # 5 signal sleeves + composite
│   ├── portfolio.py        # Inverse-vol, ex-ante vol targeting, no-trade band
│   ├── backtester.py       # Vectorised, t+1 execution, costs, attribution
│   ├── carry.py            # Carry live (courbes Yahoo) + backtest plug-ready
│   ├── metrics.py          # Sharpe (+SE), Sortino, DD, VaR/ES, turnover…
│   └── plotting.py         # Tearsheet + current-signal snapshot
├── tests/
│   └── test_pipeline.py    # No-look-ahead, execution lag, vol sanity
└── output/                 # tearsheet.png, current_signals.{png,csv}, …
```

## Usage

```bash
pip install -r requirements.txt
python main.py              # full pipeline → console report + output/
python tests/test_pipeline.py
```

The data loader falls back to a synthetic generator (correlated stochastic-vol paths, seeded) if the network is unavailable, so the pipeline always runs.

## References

- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, Journal of Financial Economics.
- Miffre & Rallis (2007), *Momentum strategies in commodity futures markets*, Journal of Banking & Finance.
- Fernandez-Perez, Frijns, Fuertes & Miffre (2018), *The skewness of commodity futures returns*, Journal of Banking & Finance.
- Nagel (2012), *Evaporating Liquidity*, Review of Financial Studies.
- Gârleanu & Pedersen (2013), *Dynamic Trading with Predictable Returns and Transaction Costs*, Journal of Finance.
- Lo (2002), *The Statistics of Sharpe Ratios*, Financial Analysts Journal.
