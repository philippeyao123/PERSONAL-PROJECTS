# Commodity Price Modeling & Trading

End-to-end quantitative pipeline for energy commodities (WTI, Brent, Henry Hub natural gas): stochastic price modeling, futures curve calibration via Kalman filtering, Monte Carlo simulation, and a backtested relative-value strategy. All on live market data.

```
pip install -r requirements.txt
python run_pipeline.py        # full pipeline, ~1 min, figures in outputs/
```

## What it does

**1. Data layer** (`src/data.py`)
Continuous front-month contracts (WTI, Brent, NG) since 2015, plus a panel of 9 WTI maturities (continuous front + 8 fixed contracts out to ~9 months) with exact time-to-maturity tracking. The negative WTI settlement of 20 April 2020 is bridged, as log-price models are undefined there.

**2. Seasonality** (`src/models/seasonality.py`)
Truncated Fourier series (K=2 harmonics) jointly fitted with a linear trend by OLS on log natural gas. The deseasonalised residual is the input to the stochastic layer — fitting mean reversion on raw seasonal prices biases the speed estimate upward.

**3. Schwartz (1997) one-factor** (`src/models/ou.py`)
Exponential OU on log spot, estimated by exact MLE (the discretised process is an AR(1), so the estimator reduces to closed-form OLS). On WTI since 2015: half-life ≈ 130 days, long-run level ≈ $63.

**4. Merton jump-diffusion** (`src/models/jumps.py`)
MLE via Poisson mixture of normals on WTI daily returns. The decomposition is the point: diffusive vol ≈ 32% annualised vs ≈ 50% unconditional — jumps (≈ 19/yr, mean −1%, std 8%) carry the tails that a Gaussian model entirely misses (see `outputs/03_jumps.png`, log scale).

**5. Schwartz–Smith (2000) two-factor** (`src/models/schwartz_smith.py`) — the core
State-space formulation: log spot = χ (short-term OU deviation) + ξ (long-term equilibrium, ABM), observed through the futures curve

&nbsp;&nbsp;&nbsp;&nbsp;ln F(t,T) = e^{−κτ} χ_t + ξ_t + A(τ) + ε

with the full risk-neutral A(τ) including risk premia and convexity terms. Parameters estimated by maximising the Kalman-filter likelihood (L-BFGS-B); the filter simultaneously extracts the latent factors. On 2024–2026 WTI data: κ ≈ 2.5 (100-day half-life), σ_χ ≈ 34%, σ_ξ ≈ 21%, curve RMSE ≈ 50bp.

A design point worth noting: a panel of *fixed* contracts has only long maturities early in the sample, where the χ-loading e^{−κτ} ≈ 0 — the short-term factor is then unidentified and the filter produces garbage. Adding the continuous front month (τ ≈ 2–6 weeks throughout) restores identification across the whole sample. This is the kind of issue constant-maturity series silently solve on a desk.

**6. Monte Carlo** (`run_pipeline.py`)
10,000 exact-discretisation paths of the two-factor spot under P over 1 year, from the latest filtered state — fan chart with 50/90% bands.

**7. WTI–Brent spread stat-arb** (`src/strategy.py`)
Rolling-OLS hedge ratio (120d, lagged — no look-ahead), z-score bands (entry ±2.0, exit ±0.5), execution at next close, 2bp proportional costs on both legs' turnover. Reported with Sharpe, max drawdown, hit rate, historical VaR/ES, and a parameter sensitivity grid.

**Honest result:** Sharpe ≈ 0.1 net across the grid over 2015–2026. The WTI–Brent convergence trade structurally weakened post-2020 (US export dynamics, Brent benchmark reform) and the backtest shows it. The grid demonstrating that no parameter corner rescues the strategy is the finding — a signed Sharpe from a single cherry-picked configuration would be the red flag.

## Structure

```
├── run_pipeline.py            # orchestrates everything
├── src/
│   ├── data.py                # yfinance loaders, term-structure panel
│   ├── strategy.py            # spread stat-arb backtest + risk metrics
│   └── models/
│       ├── ou.py              # Schwartz 1-factor, closed-form MLE
│       ├── seasonality.py     # Fourier + trend, OLS
│       ├── jumps.py           # Merton MJD, mixture MLE
│       └── schwartz_smith.py  # 2-factor Kalman MLE, curve pricing, MC
└── outputs/                   # figures + results.txt
```

## Possible extensions

Three-factor extension with stochastic convenience yield (Casassus–Collin-Dufresne); seasonality in the NG futures curve itself rather than spot only; regime-switching jump intensity; calendar-spread strategies driven by the filtered χ factor (signal: χ far from zero implies curve-shape mean reversion); transaction-cost-aware optimal bands (Leung–Li OU optimal switching).

## References

Schwartz (1997), *JF* — Schwartz & Smith (2000), *Management Science* — Merton (1976), *JFE* — Leung & Li (2015), *Optimal Mean Reversion Trading*.
