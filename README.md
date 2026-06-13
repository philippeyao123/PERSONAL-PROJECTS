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

> Four production-grade projects, one per asset class. If you review only a
> few, review these.

| Project | Asset Class | What makes it desk-relevant | Status |
|---|---|---|---|
| [G2++ Swaption Engine](./Rates%20and%20IR%20Derivatives/Swaptions%20Engine) | Rates | Two-factor Gaussian model; semi-analytic European **validated vs Monte Carlo (<1%)**; Bermudan via Longstaff-Schwartz; calibration to vol surface (**sub-2bp RMSE**); bucketed DV01 / vega / scenario risk | 🟢 tests + CI |
| [Multi-Asset Alpha Factory](./Systematic%20Trading%20and%20Alpha/Alpha%20Factory) | Systematic | Cross-sectional factor pipeline with **point-in-time data**, **deflated Sharpe (Bailey-López de Prado)**, realistic costs, capacity analysis; includes a **critical replication of Time-Series Momentum** showing post-publication decay | 🟢 tests + CI |
| [Commodity Price Modelling](./Commodity/Commodity%20price%20modelling) | Commodities | Schwartz-Smith two-factor, Merton jump-diffusion, three-factor model, OU stat-arb with optimal trading bands; full pricing → signal → backtest pipeline | 🟢 modular `src/` |
| [XVA Pricing Engine](./Rates%20and%20IR%20Derivatives/XVA%20Pricing%20Engine%20-%20Interest%20Rate%20Swaps) | Credit / Counterparty | CVA / DVA / FVA on IR swaps; exposure simulation under Hull-White, wrong-way risk, netting-set aggregation, SIMM capital | Engine + analysis |

---

## 📂 Full Project Index

### 🟢 Rates & IR Derivatives

#### G2++ Swaption Engine *(production-grade: tests + CI)*
End-to-end rates-desk toolkit: calibrate a two-factor Gaussian (G2++) model to
a swaption volatility surface, price European swaptions semi-analytically and
Bermudan swaptions by Longstaff-Schwartz, and run a bump-and-revalue risk
engine (bucketed DV01, vega, parallel / steepener / flattener / vol scenarios).
The semi-analytic European price is validated against an independent
curve-consistent Monte Carlo to within MC noise; the LSM Bermudan is checked to
equal the European at a single exercise date and to exceed it once early
exercise is allowed.
`G2++ · Bachelier/normal vol · LSM · SABR · Calibration · DV01 · Vega`

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

#### Multi-Asset Alpha Factory *(production-grade: tests + CI)*
Cross-sectional systematic alpha platform built to refuse the three ways
backtests usually lie: look-ahead/survivorship bias (point-in-time loader,
delisted names retained), gross-Sharpe theatre (all results net of a
square-root market-impact cost model), and multiple-testing inflation (deflated
and probabilistic Sharpe ratios). Includes capacity analysis and a **critical
replication of Moskowitz-Ooi-Pedersen Time-Series Momentum** demonstrating
post-publication Sharpe decay.
`PIT data · Deflated Sharpe · IC/IR · Capacity · Walk-forward · TSMOM replication`

#### Commodity Signal Generator *(modular `src/` + tests)*
Carry and momentum signal engine across commodity futures with a vectorised
backtester, transaction costs, and a performance tearsheet.
`Carry · Momentum · Backtester · Tearsheet`

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

### 🟢 Commodities

#### Commodity Price Modelling *(modular `src/`)*
Schwartz-Smith two-factor and three-factor models, Merton jump-diffusion,
seasonality, and an OU stat-arb with optimal trading bands. Full pipeline from
price modelling through Monte Carlo to a calendar-spread strategy.
`Schwartz-Smith · Jump-diffusion · Three-factor · OU bands · Monte Carlo`

#### European Power Fair Value *(modular `src/`)*
ML fair-value model for European power: ingestion, feature engineering,
walk-forward forecasting, QA, and an automated morning/trading note with
AI-generated commentary.
`Power · LightGBM · Walk-forward · Feature engineering · Automated reporting`

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
| **Languages** | Python, SQL, R, C++ (basic) |
| **Quant Libraries** | QuantLib, NumPy, SciPy, pandas, statsmodels |
| **ML/DL** | scikit-learn, PyTorch, TensorFlow, LightGBM |
| **Engineering** | pytest, ruff, mypy, GitHub Actions (CI on flagship repos) |
| **Visualisation** | Matplotlib, Plotly, Streamlit |

---

## 📄 Notes

Projects were developed independently for research purposes. Pricing models are
implemented from first principles where possible; flagship projects are
validated against independent benchmarks (e.g. analytic vs Monte Carlo) and
ship with unit tests and CI. Market data is sourced from public APIs
(yfinance, CBOE, QuantLib datasets, Elexon).

---

*Last updated: June 2026*
