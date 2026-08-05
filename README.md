# Quantitative Finance — Personal Projects

> **Bathaix Philippe-Emmanuel Yao**
> MSc Financial Mathematics · London School of Economics
> MSc Data Science · University of Essex · FRM
> 📍 London · [LinkedIn](https://www.linkedin.com/) · yaophilippeemmanuel@gmail.com

---

A curated portfolio of quantitative finance projects with a common thread:
**building models and independently validating them**. Every flagship project
is benchmarked against an independent reference (QuantLib, Monte Carlo,
analytic solutions), ships with a documented error budget and explicit
limitations, and treats risk controls — realistic costs, point-in-time
discipline, stress testing, backtest honesty — as first-class requirements
rather than afterthoughts. Coverage spans **counterparty credit & XVA**,
**market and credit risk**, **rates and IR derivatives**, **systematic
trading**, **commodities** and **volatility modelling**.

A subset of these projects are built as standalone, installable packages with
unit tests and continuous integration — marked **🟢 production-grade** below.
The rest are research notebooks and prototypes.

---

## 🎯 Risk & Model Validation Highlights

> The projects most relevant to credit, counterparty and model risk. If you
> are reviewing this portfolio from a risk perspective, start here.

- **[XVA Pricing Engine — Interest Rate Swaps](./Rates%20and%20IR%20Derivatives/XVA%20Pricing%20Engine%20-%20Interest%20Rate%20Swaps)** —
  counterparty credit risk end to end: EPE/ENE exposure simulation under
  Hull-White, CVA/DVA/FVA, wrong-way risk, netting-set aggregation and SIMM
  capital.
- **[Quadrature-order Requirements for G2++ Swaption Pricing (SSRN, 2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7211484)** —
  a reproducible **model-validation study**: a predeclared 135-cell test grid,
  two independent reference engines (QuantLib and Monte Carlo), and a full
  error budget for a pricing approximation. The companion C++20 artifact
  regenerates every table and figure.
- **[Risk Forecasting — VaR & Expected Shortfall](#risk--credit)** —
  historical, parametric and GARCH VaR/ES with EVT tails, and formal Kupiec /
  Christoffersen backtesting.
- **[Merton Structural Credit Model](#risk--credit)** — distance-to-default,
  PD extraction and portfolio loss analysis.
- **[Credit Risk Modelling](#risk--credit)** — logistic-regression and ML PD
  models with ROC/AUC/KS evaluation.
- **[Multi-Asset Portfolio Risk Terminal](#risk--credit)** — VaR/CVaR,
  historical stress scenarios (2008, COVID, rate-hike, stagflation) and factor
  attribution in an interactive dashboard.

---

## 🔑 Flagship Projects

> Six desk-oriented projects spanning counterparty risk, rates, systematic
> research, commodities and power.

| Project | Asset Class | What makes it desk-relevant | Status |
|---|---|---|---|
| [XVA Pricing Engine](./Rates%20and%20IR%20Derivatives/XVA%20Pricing%20Engine%20-%20Interest%20Rate%20Swaps) | Credit / Counterparty | CVA / DVA / FVA on IR swaps; exposure simulation under Hull-White, wrong-way risk, netting-set aggregation, SIMM capital | Engine + analysis |
| [qf-rates — Rates Pricing & Risk Library](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/qf-rates) | Rates | C++20 library covering curves, swaps, Black-76/Bachelier, G2++, European and Bermudan swaptions, calibration, Monte Carlo, LSM, DV01, volatility scenarios and XVA; independently **validated against QuantLib and Monte Carlo** | 🟢 package + tests + CI |
| [Systematic Research — Point-in-Time Research Library](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/systematic-research) | Systematic | Python research framework with **point-in-time data**, survivorship-bias controls, realistic costs and market impact, walk-forward validation, PSR/DSR, capacity, attribution and deterministic experiment tracking | 🟢 package + tests + CI |
| [Commodity Price Modeling & Trading](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/Commodity/Commodity%20price%20modelling) | Energy Commodities | Schwartz, Schwartz–Smith and three-factor models; futures-curve calibration by Kalman filtering, jump diffusion, Monte Carlo and strictly out-of-sample WTI/Brent and calendar-spread research | Research pipeline |
| [Commodity Trading Signal Generator](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/Commodity/Commodity%20signal%20generator) | Systematic Commodities | Five point-in-time signals across 15 futures, ex-ante volatility targeting, turnover controls, transaction costs, live carry analytics and honest regime-by-regime attribution | 🟢 tests |
| [European Power Fair Value](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/Commodity/European%20Power%20Fair%20Value) | European Power | Leakage-controlled DE-LU day-ahead hourly forecasting using weather and TSO fundamentals; walk-forward LightGBM, prompt-curve views, automated QA and audited LLM commentary | Pipeline + QA |

---

## 📂 Full Project Index

### Risk & Credit

#### XVA Pricing Engine — Interest Rate Swaps
CVA, DVA and FVA for vanilla IR swaps. Exposure (EPE/ENE) simulation under
Hull-White, wrong-way risk, netting-set aggregation, and SIMM capital.
`CVA · DVA · FVA · Wrong-way risk · Exposure simulation · SIMM`

#### Risk Forecasting — VaR & Expected Shortfall
Historical, parametric and GARCH VaR/ES; EVT tail estimation; Kupiec and
Christoffersen backtesting under volatility clustering.
`VaR · ES · GARCH · EVT · Kupiec · Christoffersen`

#### Merton Structural Credit Model
Firm-value model, distance-to-default, PD extraction, portfolio loss and
term-structure analysis.
`Merton · Distance-to-default · Structural credit`

#### Credit Risk Modelling
Logistic-regression and ML PD models with ROC/AUC/KS evaluation.
`PD modelling · Scorecard · Classification`

#### Multi-Asset Portfolio Risk Terminal (Baraka App)
Michaud resampling (500 iterations), Monte Carlo (10,000 paths), historical
stress scenarios (2008, COVID, rate-hike, stagflation), VaR/CVaR, diversification
ratio and factor attribution. Deployed as an interactive Streamlit dashboard.
`Michaud resampling · Monte Carlo · Stress testing · VaR/CVaR · Streamlit`

---

### 🟢 Rates & IR Derivatives

#### [qf-rates — Rates Pricing & Risk Library](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/qf-rates) *(production-grade: package + tests + CI)*
A modern C++20 quantitative-finance library for interest-rate derivatives
pricing, calibration and risk. It implements deterministic and interpolated
yield curves, vanilla swaps, Black-76 and Bachelier options, the two-factor
Gaussian G2++ model, European swaptions by joint-Gaussian quadrature and
independent Monte Carlo, and Bermudan swaptions by Longstaff-Schwartz.

The library also includes normal-volatility calibration, formal
variance-reduction measurements, LSM convergence tables, bucketed and parallel
DV01, vega, curve and volatility scenarios, exposure simulation, netting,
wrong-way risk and simplified CVA/DVA/FVA/SIMM/MVA. Python bindings expose the
principal pricing and risk workflow. Vanilla swaps match QuantLib to machine
precision, the G2++ European price is within 1%, and calibration is
cross-checked against an independent QuantLib implementation. The validation
methodology is written up in a companion paper:
[Quadrature-order Requirements for G2++ Swaption Pricing (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7211484).

`C++20 · CMake · G2++ · Bachelier · Monte Carlo · LSM · DV01 · XVA · pybind11 · QuantLib`

#### Hull-White One-Factor Model
Hull-White calibration to the yield curve; Bermudan swaption pricing via PDE
and LSM. *(Superseded by the G2++ engine above for two-factor curve risk.)*
`Hull-White · Bermudan swaptions · Yield-curve calibration`

---

### 🟢 Systematic Trading & Alpha

#### [Systematic Research — Point-in-Time Research Library](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/systematic-research) *(production-grade: package + tests + CI)*
A typed Python quantitative-research library designed to turn a hypothesis into
a reproducible and auditable result. Its point-in-time data contract retains
removed assets, records when information became available and rejects signals
that use future information. The research pipeline covers features, signals,
constrained portfolio construction, cost-aware vectorized backtesting,
walk-forward validation, risk, capacity, attribution and reporting.

Linear transaction costs, square-root market impact, turnover and participation
limits prevent gross-Sharpe theatre. Statistical validation includes
probabilistic and deflated Sharpe ratios, IC/IR, IC decay, benchmarks and
placebo tests. Every experiment records its configuration, seed, data hash and
software versions. The package supports Python 3.9–3.12 and ships with 40 tests,
more than 80% branch coverage, Ruff, mypy and CI.

`Python · Point-in-time data · PSR/DSR · IC/IR · Capacity · Walk-forward · pytest · mypy`

#### Volatility Arbitrage — Implied vs Realised
Implied–realised spread strategy with delta-hedged gamma positions and P&L
attribution.
`Vol arb · Gamma trading · P&L attribution`

#### Statistical Arbitrage — Cointegration Mean Reversion
Engle-Granger / Johansen pair selection, OU z-score signals, full backtest with
Sharpe, drawdown and turnover; multi-pair portfolio and cost sensitivity.
`Stat arb · Cointegration · OU process · Backtesting`

#### Multi-Strategy Portfolio Backtester
Modular framework combining systematic strategies with portfolio-level risk
aggregation, dynamic allocation, stress testing and liquidity-risk analysis.
`Backtesting · Risk aggregation · Stress testing`

#### Market-Making Simulator
Avellaneda-Stoikov optimal quoting under inventory risk and adverse selection;
optimal execution, multi-asset MM, and an RL quoter; P&L decomposition.
`Market making · Avellaneda-Stoikov · Inventory risk · RL`

#### Bid-Ask Spread Replication
Roll, Corwin-Schultz and Avellaneda-Stoikov spread reconstruction from data,
with a feature/modelling/quoting pipeline and dashboard.
`Roll · Corwin-Schultz · Spread estimation · Microstructure`

---

### 🟠 Commodities & Power

#### [Commodity Price Modeling & Trading](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/Commodity/Commodity%20price%20modelling)
An end-to-end energy-commodities research pipeline for WTI, Brent and Henry Hub
natural gas. It combines Fourier seasonality, an exact-MLE Schwartz one-factor
model, Merton jump diffusion and a Schwartz–Smith two-factor state-space model.
The latter is calibrated to a futures-curve panel by maximum-likelihood Kalman
filtering, extracts short- and long-term latent factors and generates 10,000
exact-discretisation Monte Carlo paths.

Extensions add a three-factor spot/convenience-yield/rates model, a strictly
out-of-sample χ-driven calendar-spread strategy and Leung–Li optimal
mean-reversion bands. The research explicitly documents identification
problems, unstable mean reversion and weak strategy results rather than
optimising them away: the WTI–Brent strategy produces a net Sharpe near 0.1,
while the limited OOS calendar sample supports only a few round trips.

`Python · Schwartz–Smith · Kalman filter · Merton jumps · Futures curves · Monte Carlo · Leung–Li`

#### [Commodity Trading Signal Generator](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/Commodity/Commodity%20signal%20generator)
A systematic multi-signal strategy across 15 energy, metals, agriculture and
soft-commodity futures. Its five sleeves combine canonical time-series
momentum, Donchian breakout, short-term cross-sectional reversal, the skewness
premium and 12-minus-1-month cross-sectional momentum. Every score uses
information available at time \(t\) and is executed at \(t+1\).

Portfolio construction uses inverse-volatility sizing, an EWMA covariance
matrix, a single ex-ante 10% volatility-targeting step, per-asset and gross
leverage caps, and a no-trade band that reduces turnover by roughly 45%.
Results include costs, Sharpe uncertainty, drawdown, VaR/ES and sleeve-level
attribution. The full-period net Sharpe is close to zero, with the 2015–2020
trend winter and post-2020 recovery reported separately. A live carry module
rebuilds commodity futures curves and provides a plug-ready historical API
without pretending that unavailable paid curve history was backtested.

`Python · TSMOM · XSMOM · Carry · Vol targeting · Transaction costs · Attribution · pytest`

#### [European Power Fair Value — DE-LU](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/Commodity/European%20Power%20Fair%20Value)
A daily power-trading prototype that forecasts the next 24 Germany–Luxembourg
EPEX day-ahead hourly prices before the D-1 auction. Public data ingestion
combines auction prices, TSO solar/wind forecasts and weather forecasts as they
were originally issued. All raw responses are cached and eight structural and
market-data QA checks cover missing hours, duplicates, physical ranges,
negative prices, stale feeds, spikes and 23/24/25-hour DST days.

Walk-forward validation uses expanding windows, weekly refits and complete
delivery-day forecasts. LightGBM achieves MAE 13.97 EUR/MWh and 59% skill
against the same-hour weekly naive benchmark on the recorded sample. The model
translates hourly forecasts into a baseload fair value and a front-week view,
with explicit invalidation rules for forecast revisions, outages and
fuel/carbon shocks. One structured LLM call turns pipeline-owned numbers into a
morning note; prompts and raw responses are logged, and the LLM never generates
quantitative values.

`Python · Power markets · LightGBM · Walk-forward · TSO fundamentals · Data QA · LLM audit`

---

### Exotic & Structured Products Pricing

#### FX Options Pricing Engine
Garman-Kohlhagen analytic pricing, Dupire local-vol surface, Monte Carlo and
PDE pricing of barriers; Bergomi LSV extension; RL-based barrier hedging; SABR
calibration and risk-reversal/butterfly term structure.
`GK · Dupire · Bergomi LSV · Barriers · PDE · SABR · RL hedging`

#### Autocallable Pricer
Autocallable structured-product engine: path-dependent payoffs, Monte Carlo
with early-termination logic, Greeks, CVA adjustment, scenario analysis.
`Autocallables · Path-dependency · Greeks · CVA`

#### Derivatives Pricing Analytics
Consolidated analytics notebook across pricing methods and sensitivities.
`Pricing · Greeks · Analytics`

---

### Volatility Surface & Modelling

#### Implied Volatility & Heston
IV extraction, SVI fit, Heston calibration, local-vol surface, and 3-D
surface/heatmap construction with arbitrage checks.
`Heston · SVI · Local vol · IV surface`

#### Machine Learning for Option Pricing
Neural-network approximation of prices and Greeks, benchmarked against analytic
and Monte Carlo baselines.
`Neural networks · Option pricing · Greeks`

---

### Research & Macro Quant

#### Quantitative Forecasting for High-Yield Bonds *(MSc dissertation)*
LSTM, GRU and gradient-boosting models for HY bond return forecasting with macro
and market feature engineering.
`LSTM · GRU · GBM · Bond forecasting`

#### Portfolio Optimization
Mean-variance, Black-Litterman and Michaud resampling; efficient frontier and
allocation sensitivity.
`Mean-variance · Black-Litterman · Resampling`

#### Sentiment Analysis on Financial News
NLP pipeline for news sentiment extraction and signal construction.
`NLP · Sentiment · Text mining`

#### Stochastic Processes & Monte Carlo Simulations
GBM, Heston, Vasicek, CIR simulation with antithetic, control-variate and
importance-sampling variance reduction.
`Monte Carlo · Variance reduction · Stochastic processes`

---

## 🛠 Technical Stack

| Category | Tools |
|---|---|
| **Languages** | Python, C++20, SQL, R |
| **Quant Libraries** | QuantLib, NumPy, SciPy, pandas, statsmodels |
| **ML/DL** | scikit-learn, PyTorch, TensorFlow, LightGBM |
| **Engineering** | CMake, pybind11, Catch2, pytest, Ruff, mypy, clang-tidy, ASan/UBSan, GitHub Actions |
| **Visualisation** | Matplotlib, Plotly, Streamlit |

---

## 📄 Notes

Projects were developed independently for research purposes. Pricing models are
implemented from first principles where possible; flagship projects are
validated against independent benchmarks (e.g. analytic vs Monte Carlo) and
ship with unit tests and CI. Market data is sourced from public APIs
(yfinance, CBOE, QuantLib datasets, Elexon).

---

*Last updated: July 2026*
