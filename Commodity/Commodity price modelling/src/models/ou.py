"""
Schwartz (1997) one-factor model — exponential Ornstein-Uhlenbeck on log spot.

    dX_t = kappa (mu - X_t) dt + sigma dW_t,   X = log S

Exact discretisation is an AR(1):
    X_{t+dt} = a + b X_t + eps,  b = e^{-kappa dt},  a = mu(1-b),
    Var(eps) = sigma^2 (1 - b^2) / (2 kappa)

so MLE reduces to an OLS regression of X_{t+1} on X_t — fast and exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OUParams:
    kappa: float      # mean-reversion speed (per year)
    mu: float         # long-run level of log price
    sigma: float      # volatility (per sqrt year)
    half_life_days: float
    r2: float

    def summary(self) -> str:
        return (f"kappa={self.kappa:.3f}/yr | mu(log)={self.mu:.3f} "
                f"(level≈{np.exp(self.mu):.2f}) | sigma={self.sigma:.3f} "
                f"| half-life={self.half_life_days:.0f} days | R²={self.r2:.3f}")


def fit_ou(log_price: pd.Series, dt: float = 1 / 252) -> OUParams:
    x = log_price.dropna().values
    y, xlag = x[1:], x[:-1]
    n = len(y)
    bx = np.cov(xlag, y, bias=True)[0, 1] / np.var(xlag)
    a = y.mean() - bx * xlag.mean()
    resid = y - (a + bx * xlag)
    sse = resid @ resid
    r2 = 1 - sse / ((y - y.mean()) @ (y - y.mean()))

    bx = min(max(bx, 1e-6), 1 - 1e-6)          # guard against non-stationary fit
    kappa = -np.log(bx) / dt
    mu = a / (1 - bx)
    sigma = np.sqrt((sse / n) * 2 * kappa / (1 - bx ** 2))
    half_life = np.log(2) / kappa * 252
    return OUParams(kappa, mu, sigma, half_life, r2)


def simulate_ou(params: OUParams, x0: float, n_days: int, n_paths: int,
                dt: float = 1 / 252, seed: int = 42) -> np.ndarray:
    """Exact simulation of the OU process. Returns array (n_days+1, n_paths) of log prices."""
    rng = np.random.default_rng(seed)
    b = np.exp(-params.kappa * dt)
    sd = params.sigma * np.sqrt((1 - b ** 2) / (2 * params.kappa))
    x = np.empty((n_days + 1, n_paths))
    x[0] = x0
    for t in range(n_days):
        x[t + 1] = params.mu * (1 - b) + b * x[t] + sd * rng.standard_normal(n_paths)
    return x
