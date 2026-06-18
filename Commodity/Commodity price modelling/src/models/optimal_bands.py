"""
Leung & Li (2015) — optimal entry/exit for OU trading with costs.

Process:   dX = kappa (theta - X) dt + sigma dW
Problem:   choose entry time nu and exit time tau to maximise
           E[ e^{-r tau} (X_tau - c_s) - e^{-r nu} (X_nu + c_b) ]

The value functions are built from the fundamental solutions of the
OU resolvent ODE  (sigma^2/2) u'' + kappa (theta - x) u' - r u = 0:

    F(x) = int_0^inf  u^{r/kappa - 1} exp(  beta (x - theta) u - u^2/2 ) du
    G(x) = int_0^inf  u^{r/kappa - 1} exp( -beta (x - theta) u - u^2/2 ) du
    beta = sqrt(2 kappa) / sigma

F is increasing, G decreasing. Optimal thresholds (Thm 4.2 / 4.5):

    exit b*:   F(b) = (b - c_s) F'(b)
    entry d*:  G(d) [ V'(d) - 1 ] = G'(d) [ V(d) - d - c_b ],
               with V(x) = (b* - c_s) F(x) / F(b*) for x < b*.

Both are smooth one-dimensional root-finding problems (quad + brentq).
The point of using this over ad-hoc z-score bands: the thresholds respond
correctly to kappa, sigma, costs and the discount rate — e.g. wider bands
when costs rise or mean reversion slows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


@dataclass
class OptimalBands:
    entry: float    # d* : buy when X <= d*
    exit: float     # b* : sell when X >= b*
    theta: float
    value_at_entry: float

    def summary(self) -> str:
        return (f"entry d*={self.entry:+.4f} | exit b*={self.exit:+.4f} | "
                f"theta={self.theta:+.4f} | expected value per round-trip "
                f"(at entry)={self.value_at_entry:.4f}")


class _OUFunctions:
    def __init__(self, kappa: float, theta: float, sigma: float, r: float):
        self.k, self.th, self.s, self.r = kappa, theta, sigma, r
        self.beta = np.sqrt(2 * kappa) / sigma
        self.p = r / kappa

    def _integrand(self, u, x, sign):
        return u ** (self.p - 1) * np.exp(sign * self.beta * (x - self.th) * u
                                          - 0.5 * u ** 2)

    def F(self, x):
        return quad(self._integrand, 0, np.inf, args=(x, +1), limit=200)[0]

    def G(self, x):
        return quad(self._integrand, 0, np.inf, args=(x, -1), limit=200)[0]

    def dF(self, x):
        f = lambda u: u ** self.p * np.exp(self.beta * (x - self.th) * u - 0.5 * u ** 2)
        return self.beta * quad(f, 0, np.inf, limit=200)[0]

    def dG(self, x):
        f = lambda u: u ** self.p * np.exp(-self.beta * (x - self.th) * u - 0.5 * u ** 2)
        return -self.beta * quad(f, 0, np.inf, limit=200)[0]


def optimal_bands(kappa: float, theta: float, sigma: float,
                  r: float = 0.04, c_buy: float = 0.0, c_sell: float = 0.0,
                  span: float = 6.0) -> OptimalBands:
    """
    Solve for (d*, b*). `span` controls the search interval in units of the
    stationary std sigma/sqrt(2 kappa).
    """
    fn = _OUFunctions(kappa, theta, sigma, r)
    sd = sigma / np.sqrt(2 * kappa)

    # ---- exit threshold: F(b) - (b - c_s) F'(b) = 0, root above theta + c_s
    g_exit = lambda b: fn.F(b) - (b - c_sell) * fn.dF(b)
    lo = max(theta, c_sell) + 1e-6
    hi = theta + span * sd
    while g_exit(hi) > 0 and hi < theta + 20 * sd:
        hi += sd
    b_star = brentq(g_exit, lo, hi, xtol=1e-8)

    V = lambda x: (b_star - c_sell) * fn.F(x) / fn.F(b_star) if x < b_star else x - c_sell
    dV = lambda x: (b_star - c_sell) * fn.dF(x) / fn.F(b_star) if x < b_star else 1.0

    # ---- entry threshold: G(d)(V'(d)-1) - G'(d)(V(d)-d-c_b) = 0, below theta
    g_entry = lambda d: fn.G(d) * (dV(d) - 1) - fn.dG(d) * (V(d) - d - c_buy)
    lo, hi = theta - span * sd, b_star - 1e-6
    while g_entry(lo) < 0 and lo > theta - 20 * sd:
        lo -= sd
    d_star = brentq(g_entry, lo, hi, xtol=1e-8)

    return OptimalBands(d_star, b_star, theta, V(d_star) - d_star - c_buy)
