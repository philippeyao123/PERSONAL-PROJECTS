# Forecasting Electricity Price Spikes in the GB Power Market

**Weather, grid fundamentals and the non-linearity of imbalance prices**

Philippe — June 2026

---

## 1. Research question

Can weather variables and grid fundamentals predict price spikes in the GB power market — and how much does the answer depend on *when* you are allowed to know what?

The central methodological choice of this project is to treat the information set as a first-class object. Two prediction problems are studied on the same target:

- **Day-ahead (DA) model.** Predicts half-hourly spike probability for delivery day D using only data published by D−1 ~16:30 (NESO demand and wind forecast vintages, weather, calendar, and price history lagged ≥ 48 settlement periods). This is the information set of a trader at the day-ahead auction.
- **Nowcast model.** Adds information that only materialises near delivery: the **wind forecast error** (outturn minus D−1 forecast), realized residual demand, and short price/NIV lags. This is the information set of an intraday/imbalance desk.

The target is the **system (imbalance) price**, not the day-ahead auction price, because that is where GB spikes actually live: over the sample the system price reaches **£2,900/MWh** against a mean of £82.

## 2. Data

All data are free and pulled programmatically (no API keys). Two full years, **June 2024 – May 2026**, half-hourly (33,215 settlement periods after warm-up).

| Series | Source | Vintage discipline |
|---|---|---|
| System price, NIV, scarcity price | Elexon Insights API | Settlement outturn (target) |
| Demand outturn, generation by fuel type | Elexon Insights API | Realized (nowcast set + EDA only) |
| Market Index price (APX MID) | Elexon Insights API | Proxy for the traded power price in the P&L exercise |
| **Day-ahead demand forecast (NDF)** | Elexon `/forecast/demand/day-ahead/history` | Vintage published **D−1 16:30** |
| **Day-ahead wind forecast** | Elexon `/forecast/generation/wind/history` | Vintage published **D−1 16:30** |
| Temperature, 100 m wind speed, solar irradiance, humidity | Open-Meteo archive (6 sites: 4 demand-weighted cities + Hornsea & Irish Sea wind belts) | Reanalysis actuals, used as a **perfect-foresight weather forecast proxy** (limitation, §7) |

The forecast series are *historical vintages*, re-fetched at a fixed publish time for every one of the 731 days, so the day-ahead feature matrix contains nothing that was unknown at the auction. This is the single most common source of inflated results in published spike-prediction work.

## 3. Feature engineering

- **Tightness:** forecast residual demand (NDF − wind forecast), forecast renewable penetration; realized counterparts for the nowcast set.
- **Weather:** GB demand-weighted temperature/humidity/solar, offshore 100 m wind speed, heating and cooling degree variables (base 15.5 °C / 22 °C).
- **Forecast error:** wind outturn − D−1 wind forecast (nowcast only).
- **Calendar:** cyclical hour/day-of-year encodings, evening-peak flag, weekend.
- **History (DA-safe, lags ≥ 48 SP):** previous-day price level/max, weekly volatility, lagged NIV, trailing 30-day spike frequency.

## 4. Spike definition

`spike = 1{ system_price > rolling 90-day 95th percentile }`, with the quantile computed on **past data only** (shifted one period). A static 95th percentile would mostly relabel winter as "spike season" and let the model win by learning the calendar; the rolling threshold makes the label *seasonally adaptive* and keeps the base rate near 6.2% throughout. The threshold itself is therefore also leakage-free.

## 5. The non-linearity (EDA)

`figs/01_nonlinearity.png` shows the half-hourly system price against realized residual demand. The median price is roughly linear in tightness, but the **95th percentile is flat near £110–130/MWh up to ~30 GW of residual demand and then bends sharply, exceeding £240/MWh beyond ~32 GW** — the classic hockey stick of the merit order running out of cheap capacity. The breakpoint is *estimated from the data* (binned quantiles), not assumed. The mirror image appears against renewable penetration: spike risk concentrates below ~10% penetration.

This is the economic justification for the modelling choices: the conditional **mean** is nearly linear, the conditional **tail** is not — which is exactly why the problem is framed as tail classification rather than price regression.

## 6. Models and evaluation

- **Models:** Logistic Regression (scaled, class-balanced) as the linear baseline; Random Forest; LightGBM.
- **Validation:** strict expanding-window **walk-forward** — 6 folds of 2 months covering the final 12 months; no shuffling, no peeking.
- **Headline metric: PR-AUC.** At a 6% base rate, ROC-AUC is flattering (a useless model gets ~0.5 "for free" and class imbalance hides false positives); average precision measures what a desk cares about — precision among the alarms raised.
- **Thresholding protocol:** the operating threshold τ is selected by max-F1 on the *first half* of the out-of-sample window and evaluated on the second half, so the reported F1/precision/recall are themselves out-of-sample.

### Results (out-of-sample, 12 months, base rate 6.2%)

**Day-ahead information set**

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | **0.269** | 0.799 | 0.306 | 0.437 | 0.235 |
| Random Forest | 0.265 | **0.802** | 0.248 | 0.535 | 0.161 |
| LightGBM | 0.240 | 0.778 | 0.282 | 0.355 | 0.234 |

**Nowcast information set**

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | **0.392** | 0.869 | **0.432** | 0.550 | 0.356 |
| Random Forest | 0.392 | **0.888** | 0.380 | 0.429 | 0.341 |
| LightGBM | 0.384 | 0.862 | 0.383 | 0.444 | 0.337 |

### Reading the results honestly

1. **The information set matters more than the model class.** Moving from day-ahead to nowcast lifts PR-AUC from ~0.26 to ~0.39 (vs a 0.062 base rate — a 6× lift over chance) for *every* model. The wind forecast error and realized tightness are worth far more than any amount of hyper-parameter tuning.
2. **The linear baseline is competitive — and that is the interesting finding.** The canonical narrative ("linear regression fails, ML wins") does not survive contact with properly engineered features: residual demand and renewable penetration already *encode* the merit-order non-linearity, so a linear classifier on top of them performs well. The non-linearity is real (§5), but it can be absorbed by feature engineering. The trees' genuine edge shows in ROC-AUC on the nowcast set (RF 0.888), i.e. in ranking quality where interactions (tightness × hour × season) matter.
3. **SHAP confirms the physics** (`figs/04`, `figs/05`): residual demand (forecast and realized), seasonality, recent price level, day-ahead wind forecast, and renewable penetration dominate; high wind forecast / penetration push spike probability *down*, tightness and winter push it *up*. Weather variables act mostly *through* the fundamentals, with offshore wind speed retaining direct marginal signal. SHAP is used instead of LightGBM's native gain importance, which is biased toward high-cardinality continuous features.

## 7. From classification to P&L

A deliberately toy strategy converts the day-ahead model into economics: whenever P(spike) ≥ τ, buy 1 MWh at the day-ahead stage (MID price as the traded-price proxy) and settle at the imbalance price.

Over 12 out-of-sample months: **+£3,020 per MWh of unit size, Sharpe 1.44 (daily, annualized), 60% hit rate, 1,103 trades, max drawdown −£1,749** (`figs/06_backtest.png`). The point is not the Sharpe — it is the demonstration that classification skill at a 6% base rate translates into positive expectancy on a long-tail payoff, because the asymmetry of spikes means a 60% hit rate is more than enough.

**Stated limitations:** no transaction costs or bid-ask, MID is an imperfect proxy for executable DA prices, unit size ignores risk limits, and the weather features assume perfect foresight (a production version would use archived NWP forecasts, e.g. Open-Meteo's historical-forecast API — likely to *hurt* the DA model and *help* the case for the nowcast split). Sample covers only two winters; spike regimes are policy- and capacity-dependent and non-stationary.

## 8. Next steps

- Replace perfect-foresight weather with archived ECMWF/ICON forecast vintages.
- Add NESO de-rated margin / LOLP vintages as direct scarcity features.
- Quantile gradient boosting on the price level (pinball loss at q95/q99) to price the *magnitude* of spikes, not just the event.
- Sequence models (LSTM/temporal fusion) only once more history is available — with ~33k observations and tabular features, gradient boosting is the efficient frontier; this is a data-scale statement, not a modelling-fashion one.
- Cost-sensitive threshold choice: pick τ from the asymmetric trading payoff rather than F1.

## Repository

```
uk-price-spikes/
├── README.md             this report
├── src/
│   ├── ingest.py         Elexon + Open-Meteo ingestion (vintage-aware, concurrent)
│   ├── features.py       dataset assembly + feature engineering
│   └── model.py          walk-forward training, metrics, SHAP, backtest, figures
├── data/
│   ├── dataset.csv       33,215 half-hourly rows, all features + target
│   ├── metrics.json      out-of-sample metrics
│   └── backtest.json     strategy statistics
└── figs/                 01 non-linearity · 02 price series · 03 PR curves
                          04–05 SHAP · 06 backtest equity
```

Reproduce: `python src/ingest.py && python src/features.py && python src/model.py` (Python 3.12, pandas, scikit-learn, lightgbm, shap; ~15 min of API pulls).
