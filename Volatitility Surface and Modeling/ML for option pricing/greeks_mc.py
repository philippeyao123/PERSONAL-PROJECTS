"""Greeks approximation (finite differences on the ML pricing surface)
and Monte Carlo benchmark.

Greeks
------
Any fitted pricing model defines a surface C_ml(S, K, T, r, sigma_atm).
Delta and vega are extracted by central finite differences, bumping the
input and REBUILDING all S-dependent features (log-moneyness, BS anchor):

    delta = [C(S(1+h)) - C(S(1-h))] / (2 h S)
    vega  = [C(sigma+dv) - C(sigma-dv)] / (2 dv)

Benchmark: analytical Black-Scholes greeks evaluated at the per-option
implied vol (the market-standard greeks). A key empirical finding is that
tree ensembles produce piecewise-constant surfaces, hence noisy/zero FD
greeks — smoothness, not RMSE, is where neural approximators earn their
keep for risk purposes.

Monte Carlo
-----------
GBM terminal-value MC with antithetic variates:
    S_T = S * exp((r - 0.5 sigma^2) T + sigma sqrt(T) Z)
Under flat (ATM) vol this converges to BS(ATM) — it benchmarks the
*pricing engine*, and its runtime motivates ML as a fast approximator.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from ml_option_pricing import bs_call


# ------------------------------------------------------ analytical greeks
def bs_delta(S, K, T, r, sigma):
    T = np.maximum(np.asarray(T, float), 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def bs_vega(S, K, T, r, sigma):
    T = np.maximum(np.asarray(T, float), 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


# ------------------------------------------------- ML surface re-pricing
def _features_at(df: pd.DataFrame, S: np.ndarray, atm_iv: np.ndarray,
                 feats: list[str]) -> np.ndarray:
    """Rebuild the clean feature matrix for bumped inputs."""
    K, T, r = df["strike"].values, df["T"].values, df["r"].values
    cols = {
        "log_moneyness": np.log(S / K),
        "sqrt_T": np.sqrt(T),
        "atm_iv": atm_iv,
        "r": r,
        "bs_atm_norm": bs_call(S, K, T, r, atm_iv) / K,
    }
    return np.column_stack([cols[f] for f in feats])


def ml_price(df, model, feats, S=None, atm_iv=None, mode="residual"):
    """Price (in $) from a fitted model at possibly bumped (S, atm_iv)."""
    S = df["S"].values if S is None else S
    atm_iv = df["atm_iv"].values if atm_iv is None else atm_iv
    K, T, r = df["strike"].values, df["T"].values, df["r"].values
    X = _features_at(df, S, atm_iv, feats)
    pred = model.predict(X)
    if mode == "residual":
        return bs_call(S, K, T, r, atm_iv) + K * pred
    return K * pred


def fd_greeks(df, model, feats, mode="residual",
              h_rel: float = 0.005, dv: float = 0.005):
    """Central finite-difference delta and vega from the ML surface."""
    S = df["S"].values
    up = ml_price(df, model, feats, S=S * (1 + h_rel), mode=mode)
    dn = ml_price(df, model, feats, S=S * (1 - h_rel), mode=mode)
    delta = (up - dn) / (2 * h_rel * S)

    iv = df["atm_iv"].values
    up = ml_price(df, model, feats, atm_iv=iv + dv, mode=mode)
    dn = ml_price(df, model, feats, atm_iv=iv - dv, mode=mode)
    vega = (up - dn) / (2 * dv)
    return delta, vega


def greeks_report(df_test, fitted: dict, feats: list[str]) -> pd.DataFrame:
    """MAE of FD greeks vs analytical BS greeks (per-option IV)."""
    S, K, T, r, iv = (df_test[c].values for c in
                      ["S", "strike", "T", "r", "iv"])
    d_ref, v_ref = bs_delta(S, K, T, r, iv), bs_vega(S, K, T, r, iv)
    rows = []
    for name, (model, mode) in fitted.items():
        d, v = fd_greeks(df_test, model, feats, mode=mode)
        rows.append(dict(model=name,
                         delta_mae=float(np.mean(np.abs(d - d_ref))),
                         vega_mae=float(np.mean(np.abs(v - v_ref))),
                         delta_in_01=float(np.mean((d >= -0.01) & (d <= 1.01)))))
    return pd.DataFrame(rows).sort_values("delta_mae")


# ----------------------------------------------------------- Monte Carlo
def mc_call(S, K, T, r, sigma, n_paths: int = 100_000, seed: int = 42):
    """GBM terminal MC with antithetic variates. Vectorised over options.
    Returns (price, standard_error)."""
    rng = np.random.default_rng(seed)
    S, K, T, sigma = map(np.atleast_1d, (S, K, T, sigma))
    n = len(S)
    Z = rng.standard_normal((n_paths // 2, n))
    Z = np.vstack([Z, -Z])                                  # antithetic
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    disc = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    return disc.mean(0), disc.std(0, ddof=1) / np.sqrt(n_paths)


def mc_benchmark(df_test: pd.DataFrame, fitted_best, feats,
                 mode="residual", n_paths: int = 100_000) -> dict:
    """Accuracy and runtime: MC(ATM vol) vs analytic BS vs ML inference."""
    S, K, T, r, iv = (df_test[c].values for c in
                      ["S", "strike", "T", "r", "atm_iv"])
    t0 = time.perf_counter()
    px_mc, se = mc_call(S, K, T, r, iv, n_paths=n_paths)
    t_mc = time.perf_counter() - t0

    t0 = time.perf_counter()
    px_bs = bs_call(S, K, T, r, iv)
    t_bs = time.perf_counter() - t0

    t0 = time.perf_counter()
    px_ml = ml_price(df_test, fitted_best, feats, mode=mode)
    t_ml = time.perf_counter() - t0

    mid = df_test["mid"].values
    rmse = lambda p: float(np.sqrt(np.mean((p - mid) ** 2)))
    return dict(
        rmse_mc=rmse(px_mc), rmse_bs=rmse(px_bs), rmse_ml=rmse(px_ml),
        mc_vs_bs_maxdiff=float(np.max(np.abs(px_mc - px_bs))),
        mc_mean_se=float(se.mean()),
        t_mc_ms=1e3 * t_mc, t_bs_ms=1e3 * t_bs, t_ml_ms=1e3 * t_ml,
        n_paths=n_paths, n_options=len(mid))
