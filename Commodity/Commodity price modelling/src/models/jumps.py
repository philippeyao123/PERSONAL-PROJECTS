"""
Merton (1976) jump-diffusion fitted to daily log returns by MLE.

    dS/S = (mu - lambda k) dt + sigma dW + (e^J - 1) dN,
    J ~ N(mu_J, sigma_J^2),  N ~ Poisson(lambda)

Over a small dt, the return density is a Poisson mixture of normals:
    f(r) = sum_{n>=0} P(N=n) * phi(r; mu dt + n mu_J, sigma^2 dt + n sigma_J^2)

We truncate the sum at n_max (jumps/day beyond ~5 are numerically irrelevant).
Useful for commodities, where supply shocks create fat tails that pure
diffusions cannot capture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm


@dataclass
class MertonParams:
    mu: float          # drift, annualised
    sigma: float       # diffusive vol, annualised
    lam: float         # jump intensity, jumps per year
    mu_j: float        # mean jump size (log)
    sigma_j: float     # jump size vol
    loglik: float
    n_obs: int

    def summary(self) -> str:
        return (f"sigma_diff={self.sigma:.1%} | lambda={self.lam:.1f} jumps/yr "
                f"| jump size: N({self.mu_j:+.2%}, {self.sigma_j:.2%}) "
                f"| logL={self.loglik:.1f}")


def _neg_loglik(theta: np.ndarray, r: np.ndarray, dt: float, n_max: int) -> float:
    mu, log_sig, log_lam, mu_j, log_sig_j = theta
    sig, lam, sig_j = np.exp(log_sig), np.exp(log_lam), np.exp(log_sig_j)
    n = np.arange(n_max + 1)
    log_pois = n * np.log(lam * dt + 1e-300) - (lam * dt) - np.cumsum(np.log(np.maximum(n, 1)))
    means = mu * dt + n * mu_j                       # (n_max+1,)
    stds = np.sqrt(sig ** 2 * dt + n * sig_j ** 2)
    comp = log_pois[None, :] + norm.logpdf(r[:, None], means[None, :], stds[None, :])
    return -logsumexp(comp, axis=1).sum()


def fit_merton(returns: pd.Series, dt: float = 1 / 252, n_max: int = 5) -> MertonParams:
    r = returns.dropna().values
    sig0 = r.std() * np.sqrt(252)
    x0 = np.array([r.mean() * 252, np.log(sig0 * 0.8), np.log(20.0),
                   0.0, np.log(0.03)])
    res = minimize(_neg_loglik, x0, args=(r, dt, n_max), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    mu, log_sig, log_lam, mu_j, log_sig_j = res.x
    return MertonParams(mu, np.exp(log_sig), np.exp(log_lam), mu_j,
                        np.exp(log_sig_j), -res.fun, len(r))


def merton_density(params: MertonParams, grid: np.ndarray,
                   dt: float = 1 / 252, n_max: int = 5) -> np.ndarray:
    n = np.arange(n_max + 1)
    log_pois = n * np.log(params.lam * dt) - params.lam * dt - np.cumsum(np.log(np.maximum(n, 1)))
    means = params.mu * dt + n * params.mu_j
    stds = np.sqrt(params.sigma ** 2 * dt + n * params.sigma_j ** 2)
    comp = np.exp(log_pois)[None, :] * norm.pdf(grid[:, None], means[None, :], stds[None, :])
    return comp.sum(axis=1)
