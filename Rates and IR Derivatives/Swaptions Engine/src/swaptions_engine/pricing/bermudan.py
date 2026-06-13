"""Bermudan swaption pricing via Longstaff-Schwartz (LSM).

A Bermudan swaption lets the holder enter the underlying swap at any of several
exercise dates. LSM estimates the optimal exercise policy by regressing the
discounted continuation value on basis functions of the state (the two G2++
factors) at each exercise date, working backwards.

Standard practice for an unbiased *price* estimate: estimate the exercise
policy on one set of paths, then evaluate the policy on an independent set
(in-sample regression overstates value). We do this two-pass split here.
"""
from __future__ import annotations

import numpy as np

from swaptions_engine.instruments.swap import Swaption
from swaptions_engine.models.g2pp import G2pp
from swaptions_engine.pricing.montecarlo import G2ppSimulator


def _basis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Polynomial basis in the two factors (degree 2 + cross term)."""
    ones = np.ones_like(x)
    return np.column_stack([ones, x, y, x**2, y**2, x * y])


class BermudanLSM:
    """Longstaff-Schwartz pricer for Bermudan swaptions under G2++."""

    def __init__(self, model: G2pp, steps_per_year: int = 12) -> None:
        self.model = model
        self.steps_per_year = steps_per_year

    def _swap_npv_at(self, swaption: Swaption, t: float,
                     x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """PV (per path) of entering the underlying swap at exercise time t.

        Only coupons with pay time > t remain. Payer value =
        (1 - P(t, t_N)) - K * annuity, all reconstructed from G2++ bonds.
        """
        swap = swaption.underlying
        m = self.model
        remaining = swap.pay_times[swap.pay_times > t + 1e-9]
        if len(remaining) == 0:
            return np.zeros_like(x)
        ann = np.zeros_like(x)
        for ti in remaining:
            ann += swap.accrual * _g2_bond(m, t, ti, x, y)
        p_last = _g2_bond(m, t, remaining[-1], x, y)
        payer_val = (1.0 - p_last) - swap.fixed_rate * ann
        return payer_val if swap.pay_fixed else -payer_val

    def price(
        self, swaption: Swaption, n_paths: int = 50_000, seed: int = 0,
    ) -> dict[str, float]:
        """Price the Bermudan. Returns price and a standard error."""
        ex_times = np.sort(swaption.exercise_times)
        T_max = ex_times[-1]

        # Time grid containing all exercise dates.
        n_steps = max(int(T_max * self.steps_per_year), len(ex_times))
        grid = np.unique(np.concatenate([
            np.linspace(0, T_max, n_steps + 1), ex_times
        ]))
        ex_idx = [int(np.argmin(np.abs(grid - t))) for t in ex_times]

        # Two independent path sets: one to learn the policy, one to price.
        train = self._simulate(grid, n_paths, seed)
        test = self._simulate(grid, n_paths, seed + 10_000)

        coefs = self._backward_regression(swaption, grid, ex_idx, *train)
        price, se = self._evaluate_policy(swaption, grid, ex_idx, coefs, *test)
        # European lower bound (most-valuable single exercise) for sanity.
        return {"price": price, "std_error": se}

    def _simulate(self, grid, n_paths, seed):
        sim = G2ppSimulator(self.model, seed=seed)
        x, y, num = sim.simulate(grid, n_paths)
        return x, y, num

    def _backward_regression(self, swaption, grid, ex_idx, x, y, num):
        """Estimate continuation-value regression coefficients backwards."""
        n_paths = x.shape[0]
        last = ex_idx[-1]
        # Cashflow if we hold to the last exercise and exercise optimally there.
        cf = np.maximum(self._swap_npv_at(swaption, grid[last], x[:, last],
                                          y[:, last]), 0.0)
        cf_time = np.full(n_paths, last)
        coefs: dict[int, np.ndarray] = {}

        for k in reversed(ex_idx[:-1]):
            t = grid[k]
            exercise = self._swap_npv_at(swaption, t, x[:, k], y[:, k])
            itm = exercise > 0
            # Discount existing cashflow back to t.
            disc = num[:, k] / num[np.arange(n_paths), cf_time]
            y_reg = cf * disc
            if itm.sum() > 10:
                X = _basis(x[itm, k], y[itm, k])
                beta, *_ = np.linalg.lstsq(X, y_reg[itm], rcond=None)
                coefs[k] = beta
                cont = X @ beta
                do_ex = exercise[itm] > cont
                idx = np.where(itm)[0][do_ex]
                cf[idx] = exercise[idx]
                cf_time[idx] = k
            else:
                coefs[k] = np.zeros(6)
        return coefs

    def _evaluate_policy(self, swaption, grid, ex_idx, coefs, x, y, num):
        """Apply the learned policy to independent paths for an unbiased price."""
        n_paths = x.shape[0]
        exercised = np.zeros(n_paths, dtype=bool)
        payoff = np.zeros(n_paths)

        for k in ex_idx:
            t = grid[k]
            exercise = self._swap_npv_at(swaption, t, x[:, k], y[:, k])
            itm = (exercise > 0) & (~exercised)
            if k == ex_idx[-1]:
                do = itm  # always exercise ITM at final date
            else:
                beta = coefs.get(k, np.zeros(6))
                cont = _basis(x[:, k], y[:, k]) @ beta
                do = itm & (exercise > cont)
            idx = np.where(do)[0]
            payoff[idx] = exercise[idx] / num[idx, k]  # discount to t0
            exercised[idx] = True

        price = float(payoff.mean())
        se = float(payoff.std() / np.sqrt(n_paths))
        return price, se


def _g2_bond(m: G2pp, t: float, T: float, x: np.ndarray, y: np.ndarray):
    return m._bond_coef_A(t, T) * np.exp(-m._Bx(t, T) * x - m._By(t, T) * y)
