"""Analytic European swaption pricing: Black-76 (lognormal) and Bachelier
(normal). Post-2008, EUR/GBP rates desks quote in *normal* (bp) vol because
rates can be negative, so Bachelier is the primary convention here; Black is
retained for comparison and for positive-rate regimes.

All formulas price a swaption as an option on the forward swap rate, scaled by
the annuity (the natural numeraire under the annuity measure).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_swaption(
    forward: float, strike: float, expiry: float, vol: float,
    annuity: float, payer: bool = True,
) -> float:
    """Black-76 swaption price (lognormal vol)."""
    if expiry <= 0 or vol <= 0:
        intrinsic = max(forward - strike, 0.0) if payer else max(strike - forward, 0.0)
        return annuity * intrinsic
    if forward <= 0 or strike <= 0:
        raise ValueError("Black model requires positive forward and strike")
    sqrt_t = np.sqrt(expiry)
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    if payer:
        price = forward * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        price = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
    return annuity * price


def bachelier_swaption(
    forward: float, strike: float, expiry: float, vol: float,
    annuity: float, payer: bool = True,
) -> float:
    """Bachelier (normal) swaption price. `vol` is absolute (e.g. bp/year)."""
    if expiry <= 0 or vol <= 0:
        intrinsic = max(forward - strike, 0.0) if payer else max(strike - forward, 0.0)
        return annuity * intrinsic
    sqrt_t = np.sqrt(expiry)
    d = (forward - strike) / (vol * sqrt_t)
    if payer:
        price = (forward - strike) * norm.cdf(d) + vol * sqrt_t * norm.pdf(d)
    else:
        price = (strike - forward) * norm.cdf(-d) + vol * sqrt_t * norm.pdf(d)
    return annuity * price


def implied_normal_vol(
    price: float, forward: float, strike: float, expiry: float,
    annuity: float, payer: bool = True,
) -> float:
    """Invert the Bachelier formula for normal implied vol (bisection)."""
    intrinsic = annuity * (max(forward - strike, 0.0) if payer
                           else max(strike - forward, 0.0))
    if price <= intrinsic + 1e-14:
        return 0.0
    lo, hi = 1e-6, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        p = bachelier_swaption(forward, strike, expiry, mid, annuity, payer)
        if p > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)
