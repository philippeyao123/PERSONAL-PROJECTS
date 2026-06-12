"""
Schwartz (1997) Model 3 — three factors:

    d ln S = (mu - delta - 0.5 sigma1^2) dt + sigma1 dW1        (spot)
    d delta = kappa (alpha - delta) dt + sigma2 dW2             (convenience yield, OU)
    d r     = a (m - r) dt + sigma3 dW3                         (short rate, Vasicek)
    corr(dW1, dW2) = rho12 ; rate shocks assumed independent of (W1, W2).

Futures price (futures = E_Q[S_T], lognormal given the Gaussian affine state):

    ln F(t, T) = ln S_t - B_k(tau) delta_t + B_a(tau) r_t + A3(tau)
    B_k(tau) = (1 - e^{-kappa tau}) / kappa,  B_a likewise with a
    A3(tau) = -alpha_hat (tau - B_k) + m (tau - B_a)
              + 0.5 [ V_delta(tau) + V_r(tau)
                      - 2 rho12 sigma1 sigma2 / kappa (tau - B_k) ]
    V_delta(tau) = sigma2^2/kappa^2 (tau - 2 B_k + B_{2k}),  V_r likewise.

alpha_hat = alpha - lambda_delta / kappa is the risk-neutral mean of the
convenience yield; we estimate alpha (P) and alpha_hat (Q) separately, which
identifies the convenience-yield risk premium. Rate parameters (a, m, sigma3)
are pre-estimated by exact MLE on the 13-week T-bill (^IRX) and held fixed;
the rate risk premium is set to zero — a documented simplification, defensible
when the curve maturities are under one year.

Estimation: Kalman filter over the latent (ln S, delta), with the *observed*
r_t entering the measurement offset; MLE by L-BFGS-B as in the two-factor case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

from .ou import fit_ou, OUParams


def load_short_rate(start: str = "2024-01-01") -> pd.Series:
    r = yf.download("^IRX", start=start, progress=False)["Close"]
    if isinstance(r, pd.DataFrame):
        r = r.iloc[:, 0]
    return (r / 100.0).dropna().rename("r")


def fit_vasicek(r: pd.Series, dt: float = 1 / 252,
                kappa_floor: float = 0.2) -> OUParams:
    """
    Vasicek on the short rate. Over a ~2y sample policy rates are nearly
    unit-root, so the unconstrained AR(1) MLE gives kappa ~ 0 and an
    explosive long-run mean (m = a/(1-b) with b -> 1). Standard fix on a
    short sample: anchor the long-run mean at the sample mean and floor
    kappa — the rate factor only enters the curve through B_a(tau) r_t and
    small convexity terms at sub-1y maturities, so this is second-order.
    """
    raw = fit_ou(r, dt)
    if raw.kappa >= kappa_floor and abs(raw.mu - r.mean()) < 5 * r.std():
        return raw
    x = r.dropna().values
    resid_sd = np.std(np.diff(x))
    return OUParams(kappa=kappa_floor, mu=float(r.mean()),
                    sigma=resid_sd / np.sqrt(dt), half_life_days=np.log(2) / kappa_floor * 252,
                    r2=raw.r2)


@dataclass
class S3Params:
    mu: float          # P-drift of ln S
    kappa: float       # conv. yield mean-reversion
    alpha: float       # P-mean of conv. yield
    alpha_hat: float   # Q-mean of conv. yield
    sigma1: float
    sigma2: float
    rho12: float
    a: float           # Vasicek (fixed)
    m: float
    sigma3: float
    s_err: float
    loglik: float

    def summary(self) -> str:
        prem = (self.alpha - self.alpha_hat) * self.kappa
        return (f"kappa_delta={self.kappa:.2f} | alpha(P)={self.alpha:+.2%} | "
                f"alpha_hat(Q)={self.alpha_hat:+.2%} (lambda_delta={prem:+.2%}) | "
                f"sigma_S={self.sigma1:.1%} | sigma_delta={self.sigma2:.1%} | "
                f"rho={self.rho12:+.2f} | s_err={self.s_err*1e4:.0f}bp | "
                f"logL={self.loglik:.0f}")


def _B(k: float, tau: np.ndarray) -> np.ndarray:
    return (1 - np.exp(-k * tau)) / k


def _A3(tau: np.ndarray, kappa, alpha_hat, sigma1, sigma2, rho12,
        a, m, sigma3) -> np.ndarray:
    Bk, Ba = _B(kappa, tau), _B(a, tau)
    B2k, B2a = _B(2 * kappa, tau), _B(2 * a, tau)
    Vd = sigma2 ** 2 / kappa ** 2 * (tau - 2 * Bk + B2k)
    Vr = sigma3 ** 2 / a ** 2 * (tau - 2 * Ba + B2a)
    return (-alpha_hat * (tau - Bk) + m * (tau - Ba)
            + 0.5 * (Vd + Vr - 2 * rho12 * sigma1 * sigma2 / kappa * (tau - Bk)))


def _kalman(theta, y, tau, r_obs, vas: OUParams, dt, return_states=False):
    mu = theta[0]
    kappa = np.exp(theta[1])
    alpha = theta[2]
    alpha_hat = theta[3]
    s1 = np.exp(theta[4])
    s2 = np.exp(theta[5])
    rho = np.tanh(theta[6])
    s_err = np.exp(theta[7])
    a, m, s3 = vas.kappa, vas.mu, vas.sigma

    T, n = y.shape
    # transition x = (lnS, delta): Euler for lnS (delta enters drift), exact OU for delta
    e = np.exp(-kappa * dt)
    F = np.array([[1.0, -dt], [0.0, e]])
    c = np.array([(mu - 0.5 * s1 ** 2) * dt, alpha * (1 - e)])
    Q = np.array([[s1 ** 2 * dt, rho * s1 * s2 * dt],
                  [rho * s1 * s2 * dt, s2 ** 2 * (1 - e ** 2) / (2 * kappa)]])

    x = np.array([y[0].mean(), 0.0])
    P = np.diag([0.5, 0.1])
    ll = 0.0
    states = np.empty((T, 2))
    for t in range(T):
        x = F @ x + c
        P = F @ P @ F.T + Q
        Bk, Ba = _B(kappa, tau[t]), _B(a, tau[t])
        H = np.column_stack([np.ones(n), -Bk])
        d = Ba * r_obs[t] + _A3(tau[t], kappa, alpha_hat, s1, s2, rho, a, m, s3)
        v = y[t] - (H @ x + d)
        S = H @ P @ H.T + s_err ** 2 * np.eye(n)
        try:
            S_inv = np.linalg.inv(S)
            _, logdet = np.linalg.slogdet(S)
        except np.linalg.LinAlgError:
            return 1e10
        ll += -0.5 * (n * np.log(2 * np.pi) + logdet + v @ S_inv @ v)
        K = P @ H.T @ S_inv
        x = x + K @ v
        P = (np.eye(2) - K @ H) @ P
        states[t] = x
    if return_states:
        return ll, states
    return -ll


def fit_three_factor(log_f: pd.DataFrame, tau: pd.DataFrame,
                     dt: float = 1 / 252) -> tuple[S3Params, pd.DataFrame, OUParams]:
    rate = load_short_rate((log_f.index[0] - pd.Timedelta(days=400)).strftime("%Y-%m-%d"))
    vas = fit_vasicek(rate)
    r_aligned = rate.reindex(log_f.index).ffill().bfill().values

    y, tau_arr = log_f.values, tau.values
    theta0 = np.array([0.0, np.log(1.5), 0.02, 0.02, np.log(0.35),
                       np.log(0.30), np.arctanh(0.7), np.log(0.01)])
    res = minimize(_kalman, theta0, args=(y, tau_arr, r_aligned, vas, dt),
                   method="L-BFGS-B", options={"maxiter": 500})
    ll, states = _kalman(res.x, y, tau_arr, r_aligned, vas, dt, return_states=True)
    t = res.x
    p = S3Params(t[0], np.exp(t[1]), t[2], t[3], np.exp(t[4]), np.exp(t[5]),
                 np.tanh(t[6]), vas.kappa, vas.mu, vas.sigma, np.exp(t[7]), ll)
    factors = pd.DataFrame(states, index=log_f.index, columns=["lnS", "delta"])
    return p, factors, vas


def model_curve_3f(p: S3Params, lnS: float, delta: float, r: float,
                   tau: np.ndarray) -> np.ndarray:
    Bk, Ba = _B(p.kappa, tau), _B(p.a, tau)
    A = _A3(tau, p.kappa, p.alpha_hat, p.sigma1, p.sigma2, p.rho12,
            p.a, p.m, p.sigma3)
    return np.exp(lnS - Bk * delta + Ba * r + A)
