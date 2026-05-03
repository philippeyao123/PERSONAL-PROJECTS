# Derivatives Pricing Analytics

> **Bathaix Philippe-Emmanuel Yao**

A unified Python library for pricing interest rate and equity derivatives, structured as a single Jupyter notebook. Three independent pricing engines share a common infrastructure layer and are benchmarked against each other on real market data.

---

## Overview

| Module | Instruments | Methods |
|---|---|---|
| **I. Shared Infrastructure** | Yield curve, vol skew, swaption | Cubic spline bootstrapping, Bachelier (Normal) model |
| **II. CMS Derivatives** | CMS forwards, caplets, floorlets | Carr-Madan replication, Linear Mean-Reversion TSR |
| **III. CMS Spread Options** | Spread caplets, floorlets | Bivariate Gaussian copula, Breeden-Litzenberger, double Simpson |
| **IV. American Options** | American calls and puts | Binomial tree (CRR), BAW approximation, Longstaff-Schwartz LSMC |

All modules are calibrated to **EURIBOR 6M market data as of 1 February 2024** (IR products) and a standard equity example for the American options section.

---

## Contents

```
Derivatives_Pricing_Analytics.ipynb
│
├── I.  Shared Infrastructure
│     ├── zc_curve          — Zero-coupon curve with cubic spline interpolation
│     ├── vol_skew          — Normal vol skew (cubic interp / linear extrap)
│     └── swaption          — Bachelier swaption pricer (PV and forward price)
│
├── II. CMS Derivatives Pricing
│     ├── tsr_model         — Linear Mean-Reversion TSR (Hull-White 1F calibration)
│     └── replication_method — Carr-Madan static replication via OTM swaptions
│
├── III. CMS Spread Options Pricing
│     ├── math_utils        — Gaussian copula density + Simpson quadrature (1D/2D)
│     └── copula_method     — Full copula pricer (Breeden-Litzenberger + Radon-Nikodym)
│
└── IV. American Options Pricing
      ├── binomial_model    — CRR binomial tree (O(N) memory, backward induction)
      ├── baw_model         — Barone-Adesi & Whaley (Newton-Raphson / bisection hybrid)
      └── ls_mc             — Longstaff-Schwartz LSMC (degree-19 polynomial regression)
```

---

## Pricing Methodology

### II — CMS Derivatives

CMS products deliver a payoff $f(S(T_f))$ at $T_p$ where $S$ is a swap rate with fixed maturity. The PV under the $T_p$-forward measure is:

$$PV = B(0, T_p) \times E^{Q^{T_p}}\!\left[f\!\left(S(T_f)\right)\right]$$

We change measure to the annuity (level) measure $Q^{\text{LVL}}$, under which $S(t)$ is a martingale, and approximate the Radon-Nikodym derivative with the **Linear Mean-Reversion TSR** function:

$$g(s) = a \cdot s + b, \qquad a = \frac{B(0,T_p)(\gamma - \beta(T_f,T_p))}{B(0,T_N)\beta(T_f,T_N) + \text{LVL}(0)\cdot S(0)\cdot\gamma}$$

The resulting expectation is evaluated by **Carr-Madan static replication** — a strip of OTM payer and receiver swaptions weighted by $h''(k) = 2a$.

### III — CMS Spread Options

A CMS spread caplet with tenors $(T_1, T_2)$ and strike $K$ prices as a double integral:

$$PV = B(0,T_p) \iint \max(s_1 - s_2 - K,\, 0)\; c\!\left(F_1(s_1), F_2(s_2)\right) f_1(s_1)\, f_2(s_2)\; ds_1\, ds_2$$

where $c$ is the **bivariate Gaussian copula** density. The individual forward-measure PDFs $f_i$ are obtained via:

1. **Breeden-Litzenberger** — second-order finite differences on Bachelier swaption prices to extract the level-measure PDF
2. **Radon-Nikodym** change of measure using the TSR weight $g(s) = as + b$

The double integral is evaluated by **double composite Simpson quadrature** on a fixed grid (50 bps step, range $[-4\%, +8\%]$).

### IV — American Options

Three methods are implemented and benchmarked:

**Binomial Tree (CRR)** — reference pricer with $N = 10^4$ steps. Memory-efficient $O(N)$ backward induction; no $N \times N$ matrix allocated.

**Barone-Adesi & Whaley** — analytical approximation. The exercise frontier $S^*$ solves:

$$\phi(S^* - K) = V^{\text{BS-Eur}}(S^*) + A\left(\frac{S^*}{S^*}\right)^b$$

via a Newton-Raphson / bisection hybrid. Production-speed method ($O(1)$ per evaluation).

**Longstaff-Schwartz LSMC** — $6 \times 10^5$ paths, 100 time steps, degree-19 polynomial regression of discounted future values onto current spot to estimate the continuation value.

---

## Requirements

```
numpy
scipy
matplotlib
jupyter
```

Install with:

```bash
pip install numpy scipy matplotlib jupyter
```

---

## Usage

```bash
git clone https://github.com/<your-username>/derivatives-pricing-analytics.git
cd derivatives-pricing-analytics
jupyter notebook Derivatives_Pricing_Analytics.ipynb
```

Each section is self-contained and can be run independently. Sections II and III share the EURIBOR 6M market data defined at the top of Section II.

---

## Numerical Results

### CMS Derivatives (5Yx10Y EURIBOR 6M, paid 6Y)

| Instrument | Price |
|---|---|
| Swap Forward | ~2.52% |
| CMS Forward (convexity-adjusted) | ~2.57% |
| ATM CMS Caplet | displayed in bps |
| ATM CMS Floorlet | displayed in bps |

### CMS Spread Options (10Y–2Y, 5Y expiry, $\rho = 0.81$)

| Strike | Caplet (bps) | Floorlet (bps) |
|---|---|---|
| 0.20% | computed | computed |
| 0.30% | computed | computed |
| 0.40% | computed | computed |

### American Options ($S_0 = 120$, $T = 0.5$, $\sigma = 35\%$, $r = 3\%$, $q = 1\%$)

| Scenario | Binomial | BAW | LSMC |
|---|---|---|---|
| OTM Put (K = 95% Fwd) | displayed | displayed | displayed |
| OTM Call (K = 105% Fwd) | displayed | displayed | displayed |

*Run the notebook to generate all numerical outputs.*

---

## References

- Breeden, D. & Litzenberger, R. (1978). *Prices of State-Contingent Claims Implicit in Option Prices*. Journal of Business.
- Carr, P. & Madan, D. (1998). *Towards a Theory of Volatility Trading*. Volatility: New Estimation Techniques for Pricing Derivatives.
- Hull, J. & White, A. (1990). *Pricing Interest-Rate-Derivative Securities*. Review of Financial Studies.
- Barone-Adesi, G. & Whaley, R. (1987). *Efficient Analytic Approximation of American Option Values*. Journal of Finance.
- Longstaff, F. & Schwartz, E. (2001). *Valuing American Options by Simulation: A Simple Least-Squares Approach*. Review of Financial Studies.
- Sklar, A. (1959). *Fonctions de répartition à n dimensions et leurs marges*. Publications de l'Institut de Statistique de l'Université de Paris.

---

## Author

**Philippe-Emmanuel Yao Bathaix**  
MSc Financial Mathematics — London School of Economics  
FRM Part I | CFA Candidate

[LinkedIn](https://linkedin.com/in/) · [GitHub](https://github.com/)
