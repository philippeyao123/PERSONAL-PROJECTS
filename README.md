# Quantitative Finance — Personal Projects

> **Bathaix Philippe-Emmanuel Yao**  
> MSc Financial Mathematics · London School of Economics 
> MSc Data Science · University of Essex  · FRM Candidate  
> 📍 London · [LinkedIn](https://www.linkedin.com/) · yaophilippeemmanuel@gmail.com

---

This repository contains a curated portfolio of quantitative finance projects developed across my graduate studies. The work spans **derivatives pricing**, **volatility modelling**, **systematic trading**, **risk analytics**, and **market microstructure**, with a consistent focus on:

- Model selection, calibration trade-offs, and hedging intuition
- Market relevance and desk applicability
- Production-oriented code and clean numerical implementation

---

## 🔑 Recommended Starting Points

> If you review only a selection of projects, start here.

| Project | Core Focus | Relevant Desks |
|---|---|---|
| [Risk Forecasting Framework – VaR & Expected Shortfall](#risk-forecasting-var--expected-shortfall) | Historical · Parametric · GARCH · EVT · Backtesting | Risk, Portfolio Analytics |
| [Multi-Asset Portfolio Risk Terminal(Baraka Portfolio App)](#multi-asset-portfolio-risk-terminal) | Michaud Resampling · Monte Carlo · Stress Testing · VaR/CVaR | Risk, Portfolio Management |
| [Heston Volatility Model Calibration](#heston-calibration-with-smile-fitting) | Stochastic Vol · Calibration · Model Risk | Vol Desk, Quant Risk |
| [Interest Rate Risk Modelling – Hull-White](#hull-white-one-factor-model) | Yield Curve Dynamics · Bermudan Swaptions · IR Sensitivity | Rates Risk, XVA |

---

## 📂 Project Index

### Derivatives & Volatility Modelling

#### Heston Calibration with Smile Fitting
Calibrated the Heston stochastic volatility model to market-implied vol surfaces using least-squares optimisation on implied vols. Benchmarked against Black–Scholes and Local Volatility. Analysis of parameter stability and smile dynamics.  
`SV model · Calibration · Model risk`

#### FX Options Pricing Engine
Full FX option pricing stack: Garman–Kohlhagen analytical pricing, Dupire Local Volatility surface construction, Monte Carlo and finite-difference PDE pricing of barrier options. Analysis of vol smiles and interest-rate differential effects.  
`GK · Local Vol · Dupire · Barrier options · PDE · Monte Carlo`

#### FX Options Pricing with Bergomi Local–Stochastic Volatility
Extension of the FX pricer to a Bergomi LSV framework. Forward variance dynamics, mixing and leverage function calibration.  
`LSV · Bergomi · Forward variance`

#### RL-Based Hedging of FX Barrier Options (Bergomi + Dupire)
Reinforcement learning agent trained to dynamically hedge FX barrier options in a Bergomi/Dupire simulation environment. Comparison against delta-hedging benchmarks under transaction costs.  
`RL · Dynamic hedging · Transaction costs`

#### Autocallable Pricer
Pricing engine for autocallable structured products. Path-dependent payoff modelling, Monte Carlo with early-termination logic, Greeks computation via finite differences and AAD-inspired approaches.  
`Autocallables · Path-dependency · Structured products · Greeks`

#### Local Volatility Model (Dupire)
Standalone Dupire Local Vol surface construction from market option prices. Arbitrage-free interpolation, surface smoothing, and forward vol dynamics analysis.  
`Dupire · Local Vol surface · Arbitrage-free interpolation`

#### Machine Learning for Option Pricing
Neural network approximation of option prices and Greeks. Benchmarked against analytical and Monte Carlo baselines. Sensitivity to training data distribution and model architecture.  
`Neural networks · Option pricing · Greeks approximation`

#### Equity Implied Volatility
Implied vol extraction from equity option chains. Term structure and skew analysis. Interpolation methods and calendar spread arbitrage detection.  
`Implied vol · Skew · Term structure`

#### CMS Derivatives Pricing
CMS convexity adjustment pricing. Replication-based and model-based approaches.  
`CMS · Convexity adjustment · Rates`

#### CMS Spread Options Pricing
Pricing of CMS spread options via Gaussian copulas and Monte Carlo simulation of joint swap rate dynamics. Spread distribution and payoff sensitivity analysis.  
`CMS spread · Copula · Monte Carlo`

---

### Interest Rate Derivatives

#### Hull–White One-Factor Model
Calibration of the Hull–White model to the yield curve. Bermudan swaption pricing via PDE and Monte Carlo with LSM early exercise.  
`Hull-White · Bermudan swaptions · Yield curve calibration`

#### American Option Pricing
Comparison of binomial tree, Barone-Adesi & Whaley approximation, and Longstaff–Schwartz Monte Carlo (LSMC) for American option pricing.  
`American options · LSMC · Binomial tree · BAW`

---

### Systematic Trading

#### Statistical Arbitrage – Cointegration-Based Mean Reversion
Pair selection via Engle–Granger and Johansen cointegration tests. Z-score entry/exit signals. Full backtest with Sharpe ratio, drawdown, and turnover analysis.  
`Stat arb · Pairs trading · Cointegration · Backtesting`

#### Volatility Arbitrage – Implied vs Realised Volatility
Strategy exploiting the implied–realised vol spread. Systematic entry via vol surface mispricing, delta-hedged gamma positions, P&L attribution.  
`Vol arb · Gamma trading · Hedging`

#### Multi-Strategy Portfolio Backtester
Modular backtesting framework combining multiple systematic strategies. Portfolio-level risk aggregation, strategy weighting, and performance attribution.  
`Backtesting · Portfolio construction · Risk aggregation`

#### Alpha Research Notebook – Modular Backtesting Framework
Research-oriented alpha generation and signal testing framework. Modular design for rapid strategy iteration.  
`Signal research · Alpha generation · Modular backtesting`

---

### Market Microstructure & Execution

#### Market-Making Simulator
Simulation of a market-maker under inventory risk and adverse selection. Optimal bid–ask spread dynamics (Avellaneda–Stoikov framework), inventory management, and P&L analysis across different flow environments.  
`Market making · Inventory risk · Avellaneda-Stoikov · Microstructure`

#### High-Frequency Trading Analysis
Order flow analysis, spread decomposition, microstructure signals, and toxicity metrics from tick data.  
`HFT · Order flow · VPIN · Microstructure`

---

### Risk, Credit & XVA

#### Risk Forecasting – VaR & Expected Shortfall
Historical, Parametric and GARCH-based VaR/ES models for portfolio risk measurement. Backtesting using Kupiec and Christoffersen tests. Tail-risk behaviour analysis under volatility clustering and Extreme Value Theory for tail estimation.  
`VaR · ES · GARCH · EVT · Backtesting · Kupiec · Christoffersen`

#### Multi-Asset Portfolio Risk Terminal
Michaud Resampling optimisation (500 iterations) targeting maximum Sharpe under resampled uncertainty. Monte Carlo simulation (10,000 paths), historical stress testing (2008, COVID, Rate Hike, Stagflation), VaR/CVaR risk metrics, diversification ratio and factor attribution. Deployed as interactive dashboard.  
`Michaud Resampling · Monte Carlo · Stress Testing · VaR/CVaR · Streamlit`

#### XVA Pricing Engine – Interest Rate Swaps
End-to-end XVA framework for vanilla IR swaps: CVA, DVA, and FVA computation. Exposure simulation under Hull–White dynamics, wrong-way risk analysis, and netting set aggregation.  
`CVA · DVA · FVA · Counterparty risk · IR swaps · Wrong-way risk`

#### Merton Structural Credit Model
Firm-value model for credit risk. Distance-to-default estimation, PD extraction, and comparison against empirical default rates.  
`Merton model · Distance-to-default · Structural credit`

#### Credit Risk Modelling
Logistic regression and ML-based PD models. ROC/AUC/KS performance metrics.  
`PD modelling · Scorecard · Classification`

---

### Macro & Quantitative Research

#### Quantitative Forecasting for High-Yield Bonds
LSTM, GRU, and gradient boosting models for HY bond return forecasting. Feature engineering from macro and market data. MSc dissertation project.  
`LSTM · GRU · GBM · Bond forecasting · NLP`

#### Sentiment Analysis on Financial News
NLP pipeline for financial news sentiment extraction and signal construction. Correlation with equity returns.  
`NLP · Sentiment · Text mining`

#### Portfolio Optimization
Mean-variance, Black–Litterman, and Michaud resampling frameworks. Efficient frontier construction and allocation sensitivity analysis.  
`Mean-variance · Black-Litterman · Resampling · Allocation`

#### Stochastic Processes & Monte Carlo Simulations
GBM, Heston, Vasicek, and CIR process simulation. Variance reduction techniques: antithetic variates, control variates, importance sampling.  
`Monte Carlo · Variance reduction · Stochastic processes`

#### Baraka Portfolio App
Streamlit-based portfolio analytics dashboard. Interactive risk and return metrics, allocation visualisation.  
`Streamlit · Dashboard · Portfolio analytics`

---

## 🛠 Technical Stack

| Category | Tools |
|---|---|
| **Languages** | Python, SQL, R, C++ (basic) |
| **Quant Libraries** | QuantLib, NumPy, SciPy, pandas, statsmodels |
| **ML/DL** | scikit-learn, PyTorch, TensorFlow |
| **Visualisation** | Plotly, Matplotlib, Streamlit |
| **Infrastructure** | Jupyter, Git, GitHub Actions |

---

## 📄 Notes

All projects were developed independently for research and learning purposes. Pricing models are implemented from first principles where possible. Market data used for calibration is sourced from public APIs (Bloomberg snapshots, CBOE, QuantLib datasets).

---

*Last updated: April 2026*
