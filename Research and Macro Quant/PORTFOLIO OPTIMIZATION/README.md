# Portfolio Optimization
---
> **Bathaix Philippe-Emmanuel Yao**

A quantitative portfolio construction framework covering four optimization paradigms, risk analytics,
and macro stress testing — driven by live price data via `yfinance`.

---

## Methods

| Method | Objective | Covariance Estimator |
|---|---|---|
| **Mean-Variance (Markowitz)** | $\max_w \text{Sharpe}(w)$ on the efficient frontier | Sample covariance |
| **Black-Litterman** | Bayesian posterior blend of CAPM equilibrium + investor views | Sample covariance |
| **Robust Optimization** | $\max_w \text{Sharpe}(w)$ with shrinkage covariance | Ledoit-Wolf shrinkage |
| **Monte Carlo** | Random portfolio sampling — visualises the feasible set | Sample covariance |
| **Scenario Analysis** | Re-optimise under bull / bear / stagflation return assumptions | Sample covariance |

---

## Features

**Data**
- Live download via `yfinance` with the updated `auto_adjust=True` API (fixes `Adj Close` → `Close` breaking change)
- Log-return computation for annualised statistics
- Normalised price performance chart and return correlation heatmap

**Optimization**
- `pypfopt.EfficientFrontier` with efficient frontier visualisation (Sharpe-coloured scatter)
- Black-Litterman with absolute views $Q$ and view matrix $P$; market equilibrium derived from market caps
- Ledoit-Wolf shrinkage for robust covariance estimation when $T/N$ is small
- Monte Carlo with $10^4$ random Dirichlet-sampled portfolios on the risk-return plane

**Risk Analytics**
- Per-asset annualised volatility, weight-vol contribution table
- Historical VaR and CVaR (Expected Shortfall) at 95% and 99% confidence
- Parametric (Gaussian) VaR for comparison
- Rolling Sharpe ratio and drawdown chart (63-day window)

**Allocation**
- Discrete share allocation via LP (`pypfopt.DiscreteAllocation`) for a configurable portfolio value

**Scenario Analysis**
- Bull market, bear market, and stagflation scenarios
- Optimal weights re-solved under each scenario's assumed return vector

---

## Mathematical Background

### Mean-Variance Optimisation

$$\max_w \frac{\mu^\top w - r_f}{\sqrt{w^\top \Sigma w}} \quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\; w \geq 0$$

### Black-Litterman Posterior

$$\tilde{\mu} = \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1} \left[(\tau\Sigma)^{-1}\pi + P^\top\Omega^{-1}Q\right]$$

where $\pi$ is the CAPM reverse-engineered equilibrium return, $P$ the view matrix, $Q$ the view returns, and $\Omega$ the view uncertainty diagonal.

### Ledoit-Wolf Shrinkage

$$\hat{\Sigma}_{LW} = (1 - \alpha)\hat{\Sigma}_{\text{sample}} + \alpha \hat{\mu}_F I$$

The optimal $\alpha$ minimises the expected Frobenius distance to the true covariance, producing more stable weights when $N$ is large relative to $T$.

### Conditional Value at Risk

$$\text{CVaR}_\alpha = -E[R \mid R \leq \text{VaR}_\alpha] = \frac{1}{1-\alpha}\int_{-\infty}^{\text{VaR}_\alpha} r\, f(r)\, dr$$

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
yfinance
pypfopt
cvxpy
scipy
```

```bash
pip install pandas numpy matplotlib seaborn yfinance pypfopt cvxpy scipy
```

---

## Usage

```bash
git clone https://github.com/philippeyao123/portfolio-optimization.git
cd portfolio-optimization
jupyter notebook Portfolio_Optimization.ipynb
```

Tickers, date range, risk-free rate, and portfolio value are configured at the top of Section 1 — no `input()` prompts.

**Default configuration:**

```python
TICKERS        = ['TSLA', 'BLK', 'NVDA', 'AAPL', 'MSFT']
START_DATE     = '2020-01-01'
END_DATE       = '2024-01-01'
TOTAL_VALUE    = 100_000       # USD
RISK_FREE_RATE = 0.04
```

---

## Results Summary (default config, indicative)

| Method | Ann. Return | Ann. Volatility | Sharpe Ratio |
|---|---|---|---|
| Mean-Variance | displayed in notebook | displayed | displayed |
| Black-Litterman | displayed | displayed | displayed |
| Robust (LW) | displayed | displayed | displayed |

*Run the notebook to generate all outputs with current market data.*

---

## References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization*. Financial Analysts Journal.
- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis.
- Martin, R. (2021). *PyPortfolioOpt Documentation*. https://pyportfolioopt.readthedocs.io
- Rockafellar, R.T. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk.

---

## Author

**Philippe-Emmanuel Yao Bathaix**  
MSc Financial Mathematics — London School of Economics  
FRM Part I | CFA Candidate

[LinkedIn](https://linkedin.com/in/) · [GitHub](https://github.com/)
