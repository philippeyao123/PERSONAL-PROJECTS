"""G2++ Monte Carlo simulation of the two factors and the discount/numeraire.

Simulates (x, y) on a time grid under the risk-neutral measure, accumulating
the money-market numeraire so that swaption payoffs can be priced by
discounted expectation. This MC engine serves two purposes:

    1. an independent check on the semi-analytic European price, and
    2. the path generator for the Longstaff-Schwartz Bermudan pricer.
"""
from __future__ import annotations

import numpy as np

from swaptions_engine.models.g2pp import G2pp


class G2ppSimulator:
    """Exact-moment Euler simulation of G2++ factors on a fixed grid."""

    def __init__(self, model: G2pp, seed: int = 0) -> None:
        self.model = model
        self.rng = np.random.default_rng(seed)

    def simulate(
        self, times: np.ndarray, n_paths: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate factors on `times`.

        Returns
        -------
        x, y : np.ndarray, shape (n_paths, len(times))
            Factor paths.
        numeraire : np.ndarray, shape (n_paths, len(times))
            Money-market account B(t) = exp(int_0^t r ds), approximated by
            trapezoidal integration of r = x + y + phi, with phi implied by
            the curve so that E[1/B(t)] matches P(0,t).
        """
        times = np.asarray(times, dtype=float)
        a, b = self.model.p.a, self.model.p.b
        s, eta, rho = self.model.p.sigma, self.model.p.eta, self.model.p.rho
        n_steps = len(times)

        x = np.zeros((n_paths, n_steps))
        y = np.zeros((n_paths, n_steps))

        for k in range(1, n_steps):
            dt = times[k] - times[k - 1]
            # Exact conditional moments of OU increments.
            ex = np.exp(-a * dt)
            ey = np.exp(-b * dt)
            var_x = s**2 / (2 * a) * (1 - ex**2)
            var_y = eta**2 / (2 * b) * (1 - ey**2)
            cov_xy = (rho * s * eta / (a + b)) * (1 - np.exp(-(a + b) * dt))
            std_x = np.sqrt(var_x)
            std_y = np.sqrt(var_y)
            corr = np.clip(cov_xy / (std_x * std_y + 1e-30), -0.999, 0.999)

            z1 = self.rng.standard_normal(n_paths)
            z2 = self.rng.standard_normal(n_paths)
            w1 = z1
            w2 = corr * z1 + np.sqrt(1 - corr**2) * z2

            x[:, k] = x[:, k - 1] * ex + std_x * w1
            y[:, k] = y[:, k - 1] * ey + std_y * w2

        # phi(t): deterministic shift so model fits the curve. The short rate
        # is r = x + y + phi; the numeraire is built so discounting is
        # curve-consistent. We use phi(t) = f(0,t) + convexity adjustment.
        phi = self._phi(times)
        r = x + y + phi[None, :]
        # Trapezoidal integral of r for the numeraire.
        integ = np.zeros((n_paths, n_steps))
        for k in range(1, n_steps):
            dt = times[k] - times[k - 1]
            integ[:, k] = integ[:, k - 1] + 0.5 * dt * (r[:, k] + r[:, k - 1])
        numeraire = np.exp(integ)
        return x, y, numeraire

    def _phi(self, times: np.ndarray) -> np.ndarray:
        """Deterministic shift phi(t) = f^M(0,t) + G2++ convexity term."""
        a, b = self.model.p.a, self.model.p.b
        s, eta, rho = self.model.p.sigma, self.model.p.eta, self.model.p.rho
        # Instantaneous forward from the curve via finite difference.
        h = 1e-4
        t_safe = np.maximum(times, h)
        df_m = self.model.curve.discount(t_safe - h)
        df_p = self.model.curve.discount(t_safe + h)
        fwd = -(np.log(df_p) - np.log(df_m)) / (2 * h)
        conv = (
            (s**2 / (2 * a**2)) * (1 - np.exp(-a * times)) ** 2
            + (eta**2 / (2 * b**2)) * (1 - np.exp(-b * times)) ** 2
            + (rho * s * eta / (a * b))
            * (1 - np.exp(-a * times))
            * (1 - np.exp(-b * times))
        )
        return fwd + conv
