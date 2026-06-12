"""
Deterministic seasonality — truncated Fourier series on log price.

    s(t) = c + sum_{k=1..K} [ a_k cos(2 pi k t) + b_k sin(2 pi k t) ],  t in years

Standard for natural gas / power, where winter demand creates a strong
annual cycle. Fitted jointly with a linear trend by OLS; the residual
(deseasonalised log price) is what stochastic models should be fitted on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SeasonalityFit:
    coefs: np.ndarray
    K: int
    fitted: pd.Series       # trend + seasonal component
    seasonal: pd.Series     # seasonal component only (zero-mean)
    residual: pd.Series     # deseasonalised, detrended log price
    r2_seasonal: float      # share of detrended variance explained by seasonality


def _design(t_years: np.ndarray, K: int) -> np.ndarray:
    cols = [np.ones_like(t_years), t_years]
    for k in range(1, K + 1):
        cols += [np.cos(2 * np.pi * k * t_years), np.sin(2 * np.pi * k * t_years)]
    return np.column_stack(cols)


def fit_seasonality(log_price: pd.Series, K: int = 2) -> SeasonalityFit:
    s = log_price.dropna()
    t = np.array([(d - s.index[0]).days / 365.25 for d in s.index])
    X = _design(t, K)
    beta, *_ = np.linalg.lstsq(X, s.values, rcond=None)
    fitted = X @ beta

    seasonal = X[:, 2:] @ beta[2:]
    trend = X[:, :2] @ beta[:2]
    detrended = s.values - trend
    resid = s.values - fitted
    r2 = 1 - np.var(resid) / np.var(detrended)

    return SeasonalityFit(
        coefs=beta, K=K,
        fitted=pd.Series(fitted, index=s.index),
        seasonal=pd.Series(seasonal - seasonal.mean(), index=s.index),
        residual=pd.Series(resid, index=s.index),
        r2_seasonal=r2,
    )
