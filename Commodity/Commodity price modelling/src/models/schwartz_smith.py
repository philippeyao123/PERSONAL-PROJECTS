"""
Schwartz & Smith (2000) two-factor model, calibrated on the futures
term structure by maximum likelihood via Kalman filter.

State (log spot decomposition):
    ln S_t = chi_t + xi_t
    d chi_t = -kappa chi_t dt + sigma_chi dW_chi        (short-term deviation, OU)
    d xi_t  = mu_xi dt + sigma_xi dW_xi                 (long-term equilibrium, ABM)
    corr(dW_chi, dW_xi) = rho

Futures prices under Q:
    ln F(t, T) = e^{-kappa tau} chi_t + xi_t + A(tau),  tau = T - t
    A(tau) = mu_xi* tau - (1 - e^{-kappa tau}) lambda_chi / kappa
             + 0.5 * [ (1 - e^{-2 kappa tau}) sigma_chi^2 / (2 kappa)
                       + sigma_xi^2 tau
                       + 2 (1 - e^{-kappa tau}) rho sigma_chi sigma_xi / kappa ]

Measurement: observed log futures = model + iid N(0, s^2) error per contract.
Estimation: scipy L-BFGS-B on the Kalman log-likelihood, with the state
initialised diffusely. This is the canonical state-space treatment used on
commodity trading desks for curve modelling and risk decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SSParams:
    kappa: float
    sigma_chi: float
    lambda_chi: float    # short-term risk premium
    mu_xi: float         # P-drift of equilibrium factor
    mu_xi_star: float    # Q-drift of equilibrium factor
    sigma_xi: float
    rho: float
    s_err: float         # measurement error std (log price units)
    loglik: float

    def summary(self) -> str:
        hl = np.log(2) / self.kappa * 365
        return (f"kappa={self.kappa:.2f} (half-life {hl:.0f}d) | "
                f"sigma_chi={self.sigma_chi:.1%} | sigma_xi={self.sigma_xi:.1%} | "
                f"rho={self.rho:+.2f} | mu_xi*={self.mu_xi_star:+.2%} | "
                f"lambda_chi={self.lambda_chi:+.2%} | s_err={self.s_err * 1e4:.0f}bp | "
                f"logL={self.loglik:.0f}")


def _A(tau: np.ndarray, p: "SSParams | np.ndarray") -> np.ndarray:
    if isinstance(p, np.ndarray):
        kappa, s_chi, lam, _, mu_q, s_xi, rho = p[:7]
    else:
        kappa, s_chi, lam, mu_q, s_xi, rho = (p.kappa, p.sigma_chi, p.lambda_chi,
                                              p.mu_xi_star, p.sigma_xi, p.rho)
    e1 = 1 - np.exp(-kappa * tau)
    e2 = 1 - np.exp(-2 * kappa * tau)
    return (mu_q * tau - e1 * lam / kappa
            + 0.5 * (e2 * s_chi ** 2 / (2 * kappa) + s_xi ** 2 * tau
                     + 2 * e1 * rho * s_chi * s_xi / kappa))


def _kalman_loglik(theta: np.ndarray, y: np.ndarray, tau: np.ndarray,
                   dt: float, return_states: bool = False):
    kappa = np.exp(theta[0])
    s_chi = np.exp(theta[1])
    lam = theta[2]
    mu_p = theta[3]
    mu_q = theta[4]
    s_xi = np.exp(theta[5])
    rho = np.tanh(theta[6])
    s_err = np.exp(theta[7])
    p = np.array([kappa, s_chi, lam, mu_p, mu_q, s_xi, rho])

    T, m = y.shape
    # transition
    F = np.array([[np.exp(-kappa * dt), 0.0], [0.0, 1.0]])
    c = np.array([0.0, mu_p * dt])
    q11 = s_chi ** 2 * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
    q12 = rho * s_chi * s_xi * (1 - np.exp(-kappa * dt)) / kappa
    Q = np.array([[q11, q12], [q12, s_xi ** 2 * dt]])

    x = np.array([0.0, y[0].mean()])           # diffuse-ish init
    P = np.diag([0.5, 0.5])
    ll = 0.0
    states = np.empty((T, 2))

    for t in range(T):
        # predict
        x = F @ x + c
        P = F @ P @ F.T + Q
        # measurement
        H = np.column_stack([np.exp(-kappa * tau[t]), np.ones(m)])
        d = _A(tau[t], p)
        v = y[t] - (H @ x + d)
        S = H @ P @ H.T + s_err ** 2 * np.eye(m)
        try:
            S_inv = np.linalg.inv(S)
            _, logdet = np.linalg.slogdet(S)
        except np.linalg.LinAlgError:
            return 1e10
        ll += -0.5 * (m * np.log(2 * np.pi) + logdet + v @ S_inv @ v)
        # update
        K = P @ H.T @ S_inv
        x = x + K @ v
        P = (np.eye(2) - K @ H) @ P
        states[t] = x

    if return_states:
        return ll, states
    return -ll


def fit_schwartz_smith(log_f: pd.DataFrame, tau: pd.DataFrame,
                       dt: float = 1 / 252) -> tuple[SSParams, pd.DataFrame]:
    y = log_f.values
    tau_arr = tau.values
    # The likelihood has local optima in kappa (slow- vs fast-reverting
    # decompositions of the same curve), especially on short samples —
    # multi-start over kappa and keep the best optimum.
    best = None
    for kappa0 in (0.5, 1.5, 3.0):
        theta0 = np.array([np.log(kappa0), np.log(0.30), 0.0, 0.0, 0.0,
                           np.log(0.15), np.arctanh(0.3), np.log(0.01)])
        res = minimize(_kalman_loglik, theta0, args=(y, tau_arr, dt),
                       method="L-BFGS-B", options={"maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    res = best
    ll, states = _kalman_loglik(res.x, y, tau_arr, dt, return_states=True)
    t = res.x
    params = SSParams(np.exp(t[0]), np.exp(t[1]), t[2], t[3], t[4],
                      np.exp(t[5]), np.tanh(t[6]), np.exp(t[7]), ll)
    factors = pd.DataFrame(states, index=log_f.index, columns=["chi", "xi"])
    return params, factors


def model_curve(params: SSParams, chi: float, xi: float, tau: np.ndarray) -> np.ndarray:
    """Model-implied futures prices for maturities tau (years)."""
    p = np.array([params.kappa, params.sigma_chi, params.lambda_chi,
                  params.mu_xi, params.mu_xi_star, params.sigma_xi, params.rho])
    return np.exp(np.exp(-params.kappa * tau) * chi + xi + _A(tau, p))


def simulate_ss(params: SSParams, chi0: float, xi0: float, n_days: int,
                n_paths: int, dt: float = 1 / 252, seed: int = 7) -> np.ndarray:
    """Monte Carlo of the spot under P. Returns (n_days+1, n_paths) spot prices."""
    rng = np.random.default_rng(seed)
    b = np.exp(-params.kappa * dt)
    sd_chi = params.sigma_chi * np.sqrt((1 - b ** 2) / (2 * params.kappa))
    sd_xi = params.sigma_xi * np.sqrt(dt)
    chol = np.linalg.cholesky(np.array([[1.0, params.rho], [params.rho, 1.0]]))

    chi = np.full(n_paths, chi0)
    xi = np.full(n_paths, xi0)
    out = np.empty((n_days + 1, n_paths))
    out[0] = np.exp(chi + xi)
    for t in range(n_days):
        z = chol @ rng.standard_normal((2, n_paths))
        chi = b * chi + sd_chi * z[0]
        xi = xi + params.mu_xi * dt + sd_xi * z[1]
        out[t + 1] = np.exp(chi + xi)
    return out
