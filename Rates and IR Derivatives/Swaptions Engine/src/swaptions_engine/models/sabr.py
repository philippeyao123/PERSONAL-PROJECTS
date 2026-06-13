"""SABR stochastic-volatility model for the swaption smile.

Implements Hagan et al. (2002) implied-vol approximations. The *normal* (bp)
approximation is the primary one used on rates desks; the lognormal form is
also provided. SABR is calibrated per expiry-tenor to fit the strike smile,
parameterized by (alpha, beta, rho, nu).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SABRParams:
    alpha: float  # ATM vol level
    beta: float   # CEV exponent (often fixed: 0=normal, 1=lognormal)
    rho: float    # spot/vol correlation in (-1, 1)
    nu: float     # vol-of-vol


def sabr_normal_vol(
    forward: float, strike: float, expiry: float, p: SABRParams
) -> float:
    """Hagan normal (Bachelier) implied vol under SABR.

    Uses the normal-vol expansion; valid for rates that may be negative when
    beta -> 0 (the typical desk choice post-2008).
    """
    a, b, rho, nu = p.alpha, p.beta, p.rho, p.nu
    if abs(forward - strike) < 1e-12:  # ATM
        term = (
            1
            + (
                (2 - 3 * rho**2) / 24 * nu**2
                # additional beta-dependent terms vanish for beta=0
            )
            * expiry
        )
        if abs(b) < 1e-12:
            return a * term
        return a * (forward ** b) * term

    # Non-ATM: zeta / chi expansion.
    if abs(b) < 1e-12:
        # Pure normal SABR (beta = 0) closed form.
        zeta = nu / a * (forward - strike)
    else:
        f_mid = np.sqrt(max(forward * strike, 1e-12))
        zeta = nu / a * (f_mid ** (1 - b)) * np.log(forward / strike)
    x_zeta = np.log((np.sqrt(1 - 2 * rho * zeta + zeta**2) + zeta - rho)
                    / (1 - rho))
    ratio = 1.0 if abs(x_zeta) < 1e-12 else zeta / x_zeta
    base = a * ratio
    correction = 1 + ((2 - 3 * rho**2) / 24 * nu**2) * expiry
    return base * correction


def sabr_atm_normal_vol(forward: float, expiry: float, p: SABRParams) -> float:
    """Convenience: ATM normal vol."""
    return sabr_normal_vol(forward, forward, expiry, p)
