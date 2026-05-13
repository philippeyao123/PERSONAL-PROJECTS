# Unified Risk Management Pipeline

> **Empirical Calibration · ML Volatility Forecasting · Monte Carlo Simulation · Regime-Conditional VaR/ES**
>
> Candidate 37279 — LSE Financial Mathematics

---

## Overview

This notebook merges three independent projects into a single end-to-end quantitative risk pipeline:

| Source | Contribution |
|---|---|
| `RISK_MANAGEMENT_SIMULATION.ipynb` | Monte Carlo engines (GBM, Heston, Cholesky), credit/operational risk models, portfolio optimisation |
| `code_project_notebook_version.ipynb` / `.py` | ML volatility forecasting (LSTM, GRU, GBM), regime detection, feature engineering, macro integration |
| `CODEHOMEWORK2.R` | Random Forest / XGBoost volatility forecasting, VaR/ES on predictions, R-side benchmarks |

The central innovation is the **calibration bridge**: instead of hardcoding simulation parameters, all stochastic models (GBM, Heston) are parameterised from empirically estimated moments computed on the CRSP dataset, and those parameters vary by **market regime**.

---

## Data

**File:** `PROJECT_DATASET.csv`

- Source: CRSP daily stock data + FRED macroeconomic series
- Period: 2001-01-02 → 2023-12-29
- Tickers: `AAPL`, `C`, `F`
- Rows: 17,355 (after cleaning)
- Columns: `PERMNO`, `date`, `TICKER`, `PERMCO`, `PRC`, `VOL`, `RET`, `BID`, `ASK`, `cpiret`, `UNRATE`, `GDP`

---

## Pipeline Architecture

```
PROJECT_DATASET.csv  (CRSP + macro, 2001–2023)
        │
        ▼
Section 2 — Feature Engineering
   Log Returns · Rolling Volatility (20d) · MACD · RSI (14d)
   MA(20/50) · Bid-Ask Spread · Lagged Returns · Macro Lags
   Interaction Terms (RET × CPI, RET × UNRATE)
        │
        ▼
Section 3 — Regime Detection
   ├── Threshold: Bull (RET > 2%) / Bear (RET < -2%) / Crisis
   └── Markov Switching (3 regimes, switching variance)
        │
        ▼
Section 4 — Empirical Calibration  ◄──── KEY BRIDGE
   Per ticker × per regime:
   GBM  → μ (annualised drift), σ (annualised vol)
   Heston → κ (mean-reversion), θ (long-run var), ξ (vol-of-vol), ρ (price-vol corr)
        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  ▼
Section 5 — ML Forecasting                    Section 6 — Monte Carlo
   GBM (sklearn) · RF · LSTM · GRU               GBM paths (N=10,000)
   Target: rolling 20d volatility                Heston paths (full truncation)
   Features: base + macro (with/without)         Correlated multi-asset (Cholesky)
   CV: TimeSeriesSplit (3 folds)                 Calibrated from Section 4
   Metrics: RMSE, MAE, R², MAPE                         │
        │                                               ▼
        └──────────────────────────────► Section 7 — Risk Metrics
                                            VaR₉₅, VaR₉₉, ES₉₅, ES₉₉
                                            Sortino Ratio, Max Drawdown
                                                        │
                                         ┌──────────────┼──────────────┐
                                         ▼              ▼              ▼
                                    Section 8      Section 9      Section 10
                                 Regime-Cond.   Multi-Asset    Credit & OpRisk
                                   VaR/ES       Portfolio Opt   Merton / CVA / LDA
                                                (MV, RP, EW)
                                                        │
                                                        ▼
                                                   Section 11
                                                 Stress Testing
                                           (GFC · COVID · Rate Shock · Bull)
                                                        │
                                                        ▼
                                                   Section 12
                                              Consolidated Dashboard
```

---

## Sections

### 0 · Imports & Configuration
All dependencies loaded with a graceful fallback for TensorFlow (LSTM/GRU sections are gated behind `KERAS_AVAILABLE`).

### 1 · Data Loading
Reads `PROJECT_DATASET.csv`, parses dates, drops 4 rows with missing `VOL`. Prints shape, tickers, date range.

### 2 · Feature Engineering
Constructs the full feature set used by both the ML models and as inputs to regime detection:
- Log returns, 20-day rolling volatility
- MACD (EMA 12/26), RSI (14-day), bid-ask spread
- Lagged returns and macro variables (lags 1–3)
- Standardised macro features (`cpiret_norm`, `UNRATE_norm`, `GDP_norm`)
- Interaction terms: `RET × cpiret`, `RET × UNRATE`

### 3 · Market Regime Detection
Two parallel approaches:
- **Threshold-based**: classifies each daily observation as Bull / Bear / Crisis based on return magnitude
- **Markov Switching** (3 regimes, switching variance): fitted on pooled returns, states mapped to Bull/Bear/Crisis by average return per state

### 4 · Empirical Calibration *(central bridge)*
For each ticker × regime combination, estimates:
- **GBM**: `μ` (annualised mean return), `σ` (annualised volatility)
- **Heston**: `κ` via AR(1) on realised variance, `θ` (long-run variance), `ξ` (vol-of-vol), `ρ` (return/variance correlation)

These parameters replace the hardcoded values from the original simulation notebook.

### 5 · ML Volatility Forecasting
Target: 20-day rolling volatility. Evaluation via `TimeSeriesSplit` (3 folds).

| Model | Library | Notes |
|---|---|---|
| GBM | `sklearn.GradientBoostingRegressor` | 200 trees, depth 4, lr 0.05 |
| RF | `sklearn.RandomForestRegressor` | 200 trees, depth 6 |
| LSTM | `tensorflow.keras` | 64→32 units, dropout 0.2, early stopping |
| GRU | `tensorflow.keras` | Same architecture as LSTM |

Features: base technical indicators + macro variables (with and without macro ablation available).

### 6 · Monte Carlo Simulation
Three simulation engines, all calibrated from Section 4:

- **GBM**: standard log-normal diffusion, 10,000 paths × 252 steps
- **Heston**: full-truncation Euler discretisation for the variance process, correlated Brownians via Cholesky
- **Correlated multi-asset GBM**: empirical correlation matrix estimated from CRSP data, Cholesky decomposition for path generation

### 7 · Risk Metrics
Computed from the terminal distribution of each simulation:
- VaR₉₅, VaR₉₉ (quantile of loss distribution)
- ES₉₅, ES₉₉ (conditional tail expectation)
- Sortino ratio, maximum drawdown (empirical, from historical returns)

### 8 · Regime-Conditional Risk
Runs regime-specific GBM simulations (1-quarter horizon) using per-regime calibrated parameters. Outputs a heatmap of VaR₉₅ across tickers × regimes, making the risk differential between Bull, Crisis, and Bear states directly readable.

### 9 · Multi-Asset Portfolio Optimisation
Three portfolio strategies evaluated on correlated simulations:

| Strategy | Method |
|---|---|
| Mean-Variance | Maximise Sharpe ratio (SciPy SLSQP) |
| Risk Parity | Equal risk contribution per asset |
| Equal Weight | 1/3 each |

Correlation matrix is estimated empirically from CRSP daily returns.

### 10 · Credit Risk & Operational Risk
- **Merton model**: structural default probability using BSM d2 formula, `σ` from Section 4 calibration
- **Counterparty CVA**: Gaussian copula simulation over 3-counterparty portfolio, with illustrative exposures and LGD
- **LDA (Loss Distribution Approach)**: compound Poisson / Exponential severity for operational risk capital estimation

### 11 · Stress Testing
Five macro scenarios applied via `(μ_mult, σ_mult)` shocks on calibrated parameters:

| Scenario | μ multiplier | σ multiplier |
|---|---|---|
| Base Case | ×1.0 | ×1.0 |
| 2008 GFC | ×−3.0 | ×3.5 |
| COVID-19 (2020) | ×−2.0 | ×2.5 |
| Rate Shock (+200bp) | ×0.5 | ×1.4 |
| Bull Run | ×2.0 | ×0.8 |

### 12 · Consolidated Dashboard
Single-cell summary printing all key outputs: ML metrics, simulation VaR/ES, regime-conditional risk table, portfolio comparison, CVA, and operational risk capital.

---

## Dependencies

```
numpy pandas matplotlib seaborn scipy statsmodels scikit-learn
tensorflow   # optional – LSTM/GRU gated behind KERAS_AVAILABLE flag
```

Install:
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn tensorflow
```

---

## Usage

### Local
```bash
# 1. Place PROJECT_DATASET.csv in the working directory
# 2. Update DATA_PATH in Section 1 if needed (default: "PROJECT_DATASET.csv")
# 3. Run all cells top to bottom
jupyter notebook Unified_Risk_Pipeline.ipynb
```

### Google Colab
```python
# Mount Drive and update DATA_PATH in Section 1:
DATA_PATH = "/content/drive/MyDrive/PROJECT_DATASET.csv"
```

---

## Key Design Decisions

**Why is empirical calibration the bridge?**
The original simulation notebook used hardcoded parameters (`μ=0.05`, `σ=0.2`, etc.). Plugging in empirically estimated parameters anchors all downstream outputs — VaR, ES, stress tests, portfolio optimisation — to the actual risk profile of AAPL, C, and F over the 2001–2023 period.

**Why regime-conditional calibration?**
Risk is not stationary. A VaR₉₅ estimated under Bull market parameters is systematically underestimated for Bear/Crisis periods. Conditioning on regime produces a more realistic risk surface and is consistent with the Markov Switching approach from the ML homework.

**Why full-truncation Heston?**
The standard Euler scheme for the variance process can produce negative variances, which breaks the square-root diffusion assumption. Full truncation (`max(V, 0)`) is the standard fix for simulation robustness without moving to the more expensive QE scheme.

**LSTM/GRU vs tree-based models**
GBM and RF are always active. LSTM/GRU require TensorFlow and are gated behind `KERAS_AVAILABLE` so the notebook runs in any environment. In practice, for volatility forecasting on daily financial data, tree-based models tend to be competitive with RNNs at far lower computational cost.

---

## File Structure

```
├── Unified_Risk_Pipeline.ipynb   ← main notebook (this file)
├── PROJECT_DATASET.csv           ← CRSP data with macro variables (required)
├── DATASETFINALE.csv             ← CRSP data without macro (reference only)
└── README.md                     ← this file
```

---

*Built on: numpy · pandas · scikit-learn · statsmodels · scipy · tensorflow · matplotlib · seaborn*
