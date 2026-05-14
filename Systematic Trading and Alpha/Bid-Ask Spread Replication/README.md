# Bid-Ask Replication Pipeline
**Market Microstructure — Full Stack: Tick Classification → Spread Reconstruction → Feature Engineering → Modelling → Optimal Quoting**

Bathaix Philippe-Emmanuel Yao  
MSc Financial Mathematics · London School of Economics

---

## Overview

End-to-end market microstructure pipeline reconstructing bid-ask spreads from trades-only data and deploying an optimal market-making strategy. Built entirely from first principles — no ML libraries, no external quant dependencies.

The pipeline addresses a canonical problem in electronic trading: given only a stream of executed trades (no order book), can we infer the prevailing bid-ask spread, identify the microstructure drivers of that spread, and use the resulting model to post optimal two-sided quotes?

**Core themes:**
- Trades-only spread inference using classical and modern estimators
- Microstructure feature engineering grounded in theoretical models
- Time-series-safe predictive modelling with walk-forward cross-validation
- Optimal quoting under inventory risk and adverse selection (Avellaneda–Stoikov)

---

## Pipeline Architecture

```
Raw Tick Stream
      │
      ▼
[M1] Lee-Ready Classification          Buyer/seller-initiated trade labelling
      │
      ▼
[M2] Spread Reconstruction             Roll · Corwin-Schultz · Level-2 Proxy
      │
      ▼
[M3] Microstructure Features           VPIN · OFI · Kyle's λ · Realized Vol
      │
      ▼
[M4] Spread Modelling                  OLS · Ridge · GBM (walk-forward CV)
      │
      ▼
[M5] Optimal Quoting                   Avellaneda-Stoikov + VPIN adverse selection
      │
      ▼
[M6] Interactive Dashboard             Plotly · Full P&L attribution
```

---

## Module Detail

### M1 — Data Ingestion & Lee-Ready Classification
`module1_data.py`

Synthetic tick data generation under a regime-switching GBM with informed and uninformed traders (α = 25% informed fraction). Trade direction inferred via the Lee-Ready (1991) algorithm:

1. **Quote rule**: compare trade price to prevailing mid — above mid → buyer-initiated, below → seller-initiated
2. **Tick rule** (fallback when trade = mid): uptick/zero-uptick → buy, downtick/zero-downtick → sell

Validation against ground truth direction on synthetic data provides a clean accuracy benchmark.

> Lee, C.M.C. & Ready, M.J. (1991). *Inferring Trade Direction from Intraday Data.* Journal of Finance, 46(2).

---

### M2 — Bid-Ask Spread Reconstruction
`module2_reconstruction.py`

Three independent estimators, each operating on trade prices only:

| Estimator | Method | Reference |
|---|---|---|
| **Roll (1984)** | Serial covariance of price changes: `S = 2√(−Cov(Δp_t, Δp_{t-1}))` | Roll (1984) |
| **Corwin-Schultz (2012)** | High-low range decomposition over adjacent periods | Corwin & Schultz (2012) |
| **Level-2 Proxy** | `ask = max(buy-side prices)`, `bid = min(sell-side prices)` per window | — |

The Level-2 proxy mirrors the reconstruction approach used in production e-trading systems when full order book data is unavailable.

> Roll, R. (1984). *A Simple Implicit Measure of the Effective Bid-Ask Spread.* Journal of Finance, 39(4).  
> Corwin, S.A. & Schultz, P. (2012). *A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices.* Journal of Finance, 67(2).

---

### M3 — Microstructure Feature Engineering
`module3_features.py`

Four canonical signals with direct theoretical grounding:

**VPIN** (Volume-Synchronized Probability of Informed Trading)  
Volume-bucketed imbalance ratio, rolling over τ buckets. Elevated VPIN signals informed flow and predicts spread widening.

**OFI** (Order Flow Imbalance)  
`OFI = (buy_vol − sell_vol) / total_vol` per window. Predictor of short-horizon price impact.

**Kyle's Lambda**  
Price impact coefficient from rolling OLS: `Δp = λ · x + ε`. Higher λ → less liquid, more informed market.

**Realized Volatility**  
`RV = √(Σ (ln p_i/p_{i-1})²)`, annualized. Dominant spread predictor (ρ = +0.80 in-sample).

| Feature | Corr(spread) |
|---|---|
| Realized Volatility | +0.80 |
| Kyle's Lambda | +0.51 |
| OFI | +0.23 |
| VPIN | +0.07 |

> Easley, D., Lopez de Prado, M.M. & O'Hara, M. (2012). *Flow Toxicity and Liquidity in a High-Frequency World.* Review of Financial Studies, 25(5).  
> Kyle, A.S. (1985). *Continuous Auctions and Insider Trading.* Econometrica, 53(6).

---

### M4 — Spread Modelling
`module4_modelling.py`

Target: Level-2 reconstructed spread (realistic inference setting — not true spread).  
Features: lagged VPIN, OFI, Kyle's λ, Realized Vol (lag-1 to prevent look-ahead bias).

Three models, all implemented from scratch in NumPy:

- **OLS**: closed-form `β = (X'X)⁻¹X'y` with standardization
- **Ridge**: `β = (X'X + αI)⁻¹X'y`, regularization for collinear features
- **GBM**: gradient boosted regression trees, CART splitting, subsampling, learning rate schedule

**Walk-forward cross-validation** (expanding window, 5 folds) enforces strict temporal ordering — no shuffling, no leakage.

---

### M5 — Optimal Quoting (Avellaneda-Stoikov)
`module5_quoting.py`

The market-maker solves a stochastic control problem to maximize terminal wealth net of inventory penalty. Optimal quotes:

```
r(s, q, t)  = s − q · γ · σ² · (T − t)                           [reservation price]
δ*          = γ · σ² · (T − t) / 2 + (1/γ) · ln(1 + γ/k)        [half-spread]

bid_t = r_t − δ*,   ask_t = r_t + δ*
```

**Key design choices:**
- **Predicted spread as σ proxy**: Module 4 predictions feed directly as the volatility input, making the pipeline fully data-driven
- **VPIN-adjusted quoting**: adverse selection widening when informed flow is elevated (`spread × (1 + VPIN)`)
- **Inventory skew**: quotes shift asymmetrically when inventory accumulates to accelerate mean-reversion
- **Fill probability**: `P(fill | δ) = exp(−k · δ)` — Poisson arrival model

P&L attribution separates spread income from inventory carry.

> Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3).

---

### M6 — Interactive Dashboard
`module6_dashboard.py`

Single-file HTML dashboard (no server required). Charts:
- Spread estimators vs true spread (M2)
- VPIN time series with threshold overlay (M3)
- OFI bar chart + Kyle's λ dual-axis (M3)
- Walk-forward CV model comparison — R² per fold (M4)
- Out-of-sample predictions vs actual (M4)
- MTM P&L with spread income and inventory PnL attribution (M5)
- Inventory + VPIN risk monitor (M5)
- Quoted vs predicted vs true spread (M5)

---

## Running the Pipeline

```bash
# No external ML dependencies — pure NumPy/pandas
pip install numpy pandas

python main.py
# → console output for all 6 modules
# → bid_ask_dashboard.html (open in browser)
```

---

## Technical Stack

| Component | Implementation |
|---|---|
| Language | Python 3.10+ |
| Dependencies | NumPy, pandas only |
| ML models | Implemented from scratch (OLS, Ridge, CART/GBM) |
| Visualisation | Plotly (CDN, no install) |
| Data | Synthetic (regime-switching GBM, configurable) |

Zero dependency on scikit-learn, XGBoost, or any quant library.

---

## Key References

| Paper | Module |
|---|---|
| Lee & Ready (1991) — *Inferring Trade Direction from Intraday Data* | M1 |
| Roll (1984) — *A Simple Implicit Measure of the Effective Bid-Ask Spread* | M2 |
| Corwin & Schultz (2012) — *A Simple Way to Estimate Bid-Ask Spreads* | M2 |
| Easley, Lopez de Prado & O'Hara (2012) — *Flow Toxicity and Liquidity* | M3 |
| Kyle (1985) — *Continuous Auctions and Insider Trading* | M3 |
| Avellaneda & Stoikov (2008) — *High-frequency trading in a limit order book* | M5 |

---

*All models implemented from first principles. Part of a broader quantitative finance portfolio.*  
*[→ Full project index](https://github.com/philippeyao123/PERSONAL-PROJECTS)*
