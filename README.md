# Quantitative Finance — Personal Projects

> **Bathaix Philippe-Emmanuel Yao**
> MSc Financial Mathematics · London School of Economics 
> MSc Data Science · University of Essex · FRM
> 📍 London · [LinkedIn](https://www.linkedin.com/) · yaophilippeemmanuel@gmail.com

---

A curated portfolio of quantitative finance projects spanning **rates and IR
derivatives**, **systematic trading**, **commodities**, **volatility
modelling**, and **risk / XVA**. The emphasis throughout is on the things that
distinguish desk-ready work from a notebook: **point-in-time discipline,
realistic costs, model validation against independent benchmarks, and
production-grade code (tests + CI).**

A subset of these projects are built as standalone, installable packages with
unit tests and continuous integration — marked **🟢 production-grade** below.
The rest are research notebooks and prototypes.

---

## 🔑 Start Here — Flagship Projects

> Two standalone quantitative libraries plus a counterparty-risk engine. If
> you review only a few projects, review these.

| Project | Asset Class | What makes it desk-relevant | Status |
|---|---|---|---|
| [qf-rates — Rates Pricing & Risk Library](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/qf-rates) | Rates | C++20 library covering curves, swaps, Black-76/Bachelier, G2++, European and Bermudan swaptions, calibration, Monte Carlo, LSM, DV01, volatility scenarios and XVA; independently **validated against QuantLib and Monte Carlo** | 🟢 package + tests + CI |
| [Systematic Research — Point-in-Time Research Library](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/systematic-research) | Systematic | Python research framework with **point-in-time data**, survivorship-bias controls, realistic costs and market impact, walk-forward validation, PSR/DSR, capacity, attribution and deterministic experiment tracking | 🟢 package + tests + CI |
| [XVA Pricing Engine](./Rates%20and%20IR%20Derivatives/XVA%20Pricing%20Engine%20-%20Interest%20Rate%20Swaps) | Credit / Counterparty | CVA / DVA / FVA on IR swaps; exposure simulation under Hull-White, wrong-way risk, netting-set aggregation, SIMM capital | Engine + analysis |

---

## 📂 Full Project Index

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
cross-checked against an independent QuantLib implementation.

`C++20 · CMake · G2++ · Bachelier · Monte Carlo · LSM · DV01 · XVA · pybind11 · QuantLib`

#### XVA Pricing Engine — Interest Rate Swaps
CVA, DVA and FVA for vanilla IR swaps. Exposure (EPE/ENE) simulation under
Hull-White, wrong-way risk, netting-set aggregation, and SIMM capital.
`CVA · DVA · FVA · Wrong-way risk · Exposure simulation · SIMM`

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

### Risk & Credit

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
