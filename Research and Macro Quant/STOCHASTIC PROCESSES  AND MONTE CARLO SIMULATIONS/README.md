# Stochastic Processes & Monte Carlo Simulation
---
> **Bathaix Philippe-Emmanuel Yao**

A modular Python framework for simulating asset price dynamics under three classical stochastic
process models, with a vectorised Monte Carlo engine, risk metrics, and interactive exploration
via `ipywidgets`.

---

## Models Implemented

| Process | Dynamics | Typical Use |
|---|---|---|
| **Geometric Brownian Motion** | $dS = \mu S\,dt + \sigma S\,dW$ | Equity prices, Black-Scholes baseline |
| **Merton Jump Diffusion** | GBM + compound Poisson jumps | Crash risk, fat-tailed returns |
| **Ornstein-Uhlenbeck** | $dX = \kappa(\theta - X)\,dt + \sigma\,dW$ | Interest rates, vol, stat-arb spreads |

---

## Features

**Simulation**
- Vectorised path generation — full $(M \times N)$ matrix in a single NumPy operation (~50× faster than Python loops)
- Antithetic variates variance reduction built into the base `StochasticProcess` class
- Exact discretisation for the OU process (no Euler-Maruyama bias)
- Jump-size aggregation via Poisson sampling with correct drift correction ($\lambda\kappa$ adjustment)

**Analytics**
- Terminal distribution statistics: mean, std, skewness, excess kurtosis
- Historical VaR and CVaR (Expected Shortfall) at arbitrary confidence levels
- Analytical benchmarks for GBM: $E[S(T)]$, $\text{Std}[S(T)]$, analytical VaR
- Convergence diagnostics: mean and VaR vs. number of paths $M$
- Analytical OU moments ($E[X(t)]$, $\text{Var}[X(t)]$) for MC validation

**Visualisation**
- Simulated paths with mean and 90% confidence band
- Terminal price histogram with analytical log-normal overlay (GBM) and VaR/CVaR markers
- GBM vs. Jump Diffusion risk metric comparison table
- Convergence plots (mean and VaR vs. $M$)
- Interactive dashboard via `ipywidgets`

---

## Mathematical Background

### Geometric Brownian Motion

The exact solution (Itô's lemma) is:

$$S(t) = S_0 \exp\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)t + \sigma W(t)\right]$$

where $W(t)$ is a standard Brownian motion. The terminal distribution $\log(S(T)/S_0)$ is Gaussian, giving a log-normal $S(T)$.

**Analytical moments:**
$$E[S(T)] = S_0 e^{\mu T}, \qquad \text{Std}[S(T)] = S_0 e^{\mu T}\sqrt{e^{\sigma^2 T} - 1}$$

### Merton Jump Diffusion

Jumps arrive as a Poisson process with intensity $\lambda$. Log-jump sizes are $\sim \mathcal{N}(\mu_J, \sigma_J^2)$. The drift is adjusted by $-\lambda\kappa$ (where $\kappa = e^{\mu_J + \sigma_J^2/2} - 1$) to keep $E[dS/S] = \mu\,dt$.

### Ornstein-Uhlenbeck

The exact conditional distribution is Gaussian:

$$X(t+dt) \mid X(t) \sim \mathcal{N}\!\left(X(t)e^{-\kappa dt} + \theta(1 - e^{-\kappa dt}),\; \frac{\sigma^2(1 - e^{-2\kappa dt})}{2\kappa}\right)$$

Long-run stationary distribution: $X \sim \mathcal{N}(\theta, \sigma^2 / 2\kappa)$.

---

## Requirements

```
numpy
scipy
matplotlib
ipywidgets
jupyter
```

```bash
pip install numpy scipy matplotlib ipywidgets jupyter
```

---

## Usage

```bash
git clone https://github.com/<your-username>/stochastic-processes-mc.git
cd stochastic-processes-mc
jupyter notebook Stochastic_Processes_Monte_Carlo.ipynb
```

Each section runs independently. The interactive dashboard (Section 5.4) requires Jupyter Notebook or JupyterLab with the `ipywidgets` extension enabled.

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy.
- Merton, R.C. (1976). *Option pricing when underlying stock returns are discontinuous*. Journal of Financial Economics.
- Vasicek, O. (1977). *An equilibrium characterization of the term structure*. Journal of Financial Economics.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

---

## Author

**Philippe-Emmanuel Yao Bathaix**  
MSc Financial Mathematics — London School of Economics  
FRM Part I | CFA Candidate

[LinkedIn](https://linkedin.com/in/) · [GitHub](https://github.com/)
