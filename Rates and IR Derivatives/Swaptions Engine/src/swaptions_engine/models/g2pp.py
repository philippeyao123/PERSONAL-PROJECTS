"""G2++ : two-additive-factor Gaussian short-rate model.

    r(t) = x(t) + y(t) + phi(t)
    dx = -a x dt + sigma dW1
    dy = -b y dt + eta  dW2
    dW1 dW2 = rho dt

phi(t) is fitted to the initial discount curve so the model is calibrated to
today's term structure by construction. G2++ is the workhorse for Bermudan
swaptions on rates desks because two factors capture curve-shape risk that a
one-factor Hull-White cannot, while staying analytically tractable for
European swaptions (via Jamshidian-style decomposition / a 1-D integral).

References: Brigo & Mercurio, "Interest Rate Models", Ch. 4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from swaptions_engine.curve.yield_curve import Curve
from swaptions_engine.instruments.swap import VanillaSwap


@dataclass
class G2ppParams:
    a: float      # mean reversion of x
    sigma: float  # vol of x
    b: float      # mean reversion of y
    eta: float    # vol of y
    rho: float    # correlation between the two factors

    def validate(self) -> None:
        if self.a <= 0 or self.b <= 0:
            raise ValueError("mean reversions must be positive")
        if self.sigma < 0 or self.eta < 0:
            raise ValueError("vols must be non-negative")
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")


class G2pp:
    """G2++ model fitted to an initial curve."""

    def __init__(self, curve: Curve, params: G2ppParams) -> None:
        params.validate()
        self.curve = curve
        self.p = params

    # ---- analytic building blocks (Brigo-Mercurio notation) ----
    def _V(self, t: float, T: float) -> float:
        """Variance term V(t,T) used in the bond reconstitution formula."""
        a, b, s, eta, rho = (self.p.a, self.p.b, self.p.sigma,
                             self.p.eta, self.p.rho)
        tau = T - t
        Va = (s**2 / a**2) * (tau + (2 / a) * np.exp(-a * tau)
                              - (1 / (2 * a)) * np.exp(-2 * a * tau) - 3 / (2 * a))
        Vb = (eta**2 / b**2) * (tau + (2 / b) * np.exp(-b * tau)
                                - (1 / (2 * b)) * np.exp(-2 * b * tau) - 3 / (2 * b))
        Vab = (2 * rho * s * eta / (a * b)) * (
            tau + (np.exp(-a * tau) - 1) / a + (np.exp(-b * tau) - 1) / b
            - (np.exp(-(a + b) * tau) - 1) / (a + b)
        )
        return Va + Vb + Vab

    def discount_bond(self, t: float, T: float, x: float, y: float) -> float:
        """P(t,T) given factor values x,y at t (model-implied bond price)."""
        a, b = self.p.a, self.p.b
        pT = float(self.curve.discount(T))
        pt = float(self.curve.discount(t))
        A = (pT / pt) * np.exp(
            0.5 * (self._V(t, T) - self._V(0, T) + self._V(0, t))
        )
        Bx = (1 - np.exp(-a * (T - t))) / a
        By = (1 - np.exp(-b * (T - t))) / b
        return A * np.exp(-Bx * x - By * y)

    def european_swaption(
        self, swaption_underlying: VanillaSwap, payer: bool = True,
        strike: float | None = None,
    ) -> float:
        """Price a European swaption via Brigo-Mercurio's 1-D integral (G2++).

        Works under the T-forward measure: x(T) is Gaussian, and conditional on
        x the exercise condition on y is a single threshold y*(x). The price is
        a 1-D integral over x of a Black-Scholes-like expression. See Brigo &
        Mercurio (2006), eq. 4.31.
        """
        swap = swaption_underlying
        T = swap.start
        K = swap.fixed_rate if strike is None else strike
        a, b, s, eta, rho = (self.p.a, self.p.b, self.p.sigma,
                             self.p.eta, self.p.rho)
        w = 1.0 if payer else -1.0

        pay_times = swap.pay_times
        # Cashflows of the underlying "coupon bond" struck at K: c_i = K*tau,
        # last one + notional. The payer swaption is a put on this bond struck
        # at par (1), receiver is a call.
        c = np.full(len(pay_times), swap.accrual * K)
        c[-1] += 1.0

        # T-forward moments of x(T), y(T)  (Brigo-Mercurio 4.31 mu_x, mu_y).
        mu_x = -self._M_x(0.0, T, T)
        mu_y = -self._M_y(0.0, T, T)
        sig_x = s * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a))
        sig_y = eta * np.sqrt((1 - np.exp(-2 * b * T)) / (2 * b))
        rho_xy = (rho * s * eta) / ((a + b) * sig_x * sig_y) * \
                 (1 - np.exp(-(a + b) * T))
        rho_xy = float(np.clip(rho_xy, -0.999, 0.999))

        A_i = np.array([self._bond_coef_A(T, ti) for ti in pay_times])
        Ba = np.array([self._Bx(T, ti) for ti in pay_times])  # coeff on x
        Bb = np.array([self._By(T, ti) for ti in pay_times])  # coeff on y

        def integrand(x: float) -> float:
            mu_cond = mu_y + rho_xy * (sig_y / sig_x) * (x - mu_x)
            std_cond = sig_y * np.sqrt(1 - rho_xy**2)

            # lambda_i(x) = c_i A_i exp(-Ba_i x)
            lam = c * A_i * np.exp(-Ba * x)
            # Solve sum_i lam_i exp(-Bb_i y) = 1 for y* (decreasing in y).
            y_star = _solve_exercise(np.ones_like(lam), lam, Bb)

            h1 = (y_star - mu_cond) / std_cond
            # sum_i lam_i exp(-Bb_i mu_cond + 0.5 Bb_i^2 std_cond^2) * N(...)
            kappa = -Bb * (mu_cond - 0.5 * Bb * std_cond**2)
            terms = lam * np.exp(kappa)
            h2 = h1 + Bb * std_cond
            inner = norm.cdf(-w * h1) - np.sum(
                terms * norm.cdf(-w * h2)
            )
            phi_x = np.exp(-0.5 * ((x - mu_x) / sig_x) ** 2) / (
                sig_x * np.sqrt(2 * np.pi)
            )
            return phi_x * w * inner

        lo, hi = mu_x - 8 * sig_x, mu_x + 8 * sig_x
        price, _ = quad(integrand, lo, hi, limit=120)
        p0 = float(self.curve.discount(T))
        return p0 * max(price, 0.0)

    def _M_x(self, s_: float, t: float, T: float) -> float:
        """T-forward drift adjustment for x (Brigo-Mercurio 4.31)."""
        a, b, sig, eta, rho = (self.p.a, self.p.b, self.p.sigma,
                               self.p.eta, self.p.rho)
        return (
            (sig**2 / a**2 + rho * sig * eta / (a * b))
            * (1 - np.exp(-a * (t - s_)))
            - (sig**2 / (2 * a**2))
            * (np.exp(-a * (T - t)) - np.exp(-a * (T + t - 2 * s_)))
            - (rho * sig * eta / (b * (a + b)))
            * (np.exp(-b * (T - t)) - np.exp(-b * T - a * t + (a + b) * s_))
        )

    def _M_y(self, s_: float, t: float, T: float) -> float:
        """T-forward drift adjustment for y (symmetric to _M_x)."""
        a, b, sig, eta, rho = (self.p.a, self.p.b, self.p.sigma,
                               self.p.eta, self.p.rho)
        return (
            (eta**2 / b**2 + rho * sig * eta / (a * b))
            * (1 - np.exp(-b * (t - s_)))
            - (eta**2 / (2 * b**2))
            * (np.exp(-b * (T - t)) - np.exp(-b * (T + t - 2 * s_)))
            - (rho * sig * eta / (a * (a + b)))
            * (np.exp(-a * (T - t)) - np.exp(-a * T - b * t + (a + b) * s_))
        )

    # bond price P(T, ti) = A_coef * exp(-Bx x - By y); helpers:
    def _Bx(self, t: float, T: float) -> float:
        return (1 - np.exp(-self.p.a * (T - t))) / self.p.a

    def _By(self, t: float, T: float) -> float:
        return (1 - np.exp(-self.p.b * (T - t))) / self.p.b

    def _bond_coef_A(self, t: float, T: float) -> float:
        pT, pt = float(self.curve.discount(T)), float(self.curve.discount(t))
        return (pT / pt) * np.exp(
            0.5 * (self._V(t, T) - self._V(0, T) + self._V(0, t))
        )


def _solve_exercise(c: np.ndarray, lam: np.ndarray, kappa: np.ndarray) -> float:
    """Solve sum_i c_i lam_i exp(-kappa_i y) = 1 for y (monotone in y)."""
    def f(y: float) -> float:
        return float(np.sum(c * lam * np.exp(-kappa * y)) - 1.0)
    lo, hi = -1.0, 1.0
    # expand bracket
    for _ in range(60):
        if f(lo) * f(hi) < 0:
            break
        lo *= 1.5
        hi *= 1.5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)
