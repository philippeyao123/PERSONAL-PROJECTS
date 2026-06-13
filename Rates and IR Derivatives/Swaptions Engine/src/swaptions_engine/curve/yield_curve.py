"""Yield curve: discount factors, zero rates, forward rates.

A curve is defined by a set of pillar times and zero rates; discount factors
are interpolated log-linearly (equivalently, piecewise-constant instantaneous
forwards), which is the market-standard, arbitrage-free-on-the-pillars choice
for a single-curve framework.

This is deliberately single-curve (no OIS/LIBOR basis). The dual-curve
extension is a documented next step; the pricing/risk layers consume the
`Curve` interface and would not change.
"""
from __future__ import annotations

import numpy as np


class Curve:
    """A zero-coupon yield curve with log-linear DF interpolation.

    Parameters
    ----------
    pillar_times : np.ndarray
        Year fractions of the curve pillars (ascending, > 0).
    zero_rates : np.ndarray
        Continuously-compounded zero rates at each pillar.
    """

    def __init__(self, pillar_times: np.ndarray, zero_rates: np.ndarray) -> None:
        pillar_times = np.asarray(pillar_times, dtype=float)
        zero_rates = np.asarray(zero_rates, dtype=float)
        if pillar_times.ndim != 1 or pillar_times.shape != zero_rates.shape:
            raise ValueError("pillar_times and zero_rates must be 1-D, same len")
        if np.any(np.diff(pillar_times) <= 0):
            raise ValueError("pillar_times must be strictly increasing")
        if np.any(pillar_times <= 0):
            raise ValueError("pillar_times must be positive")
        self.t = pillar_times
        self.r = zero_rates
        # log DF at pillars = -r*t ; we interpolate this linearly in t.
        self._log_df = -self.r * self.t

    def discount(self, t: float | np.ndarray) -> np.ndarray:
        """Discount factor P(0, t)."""
        t = np.asarray(t, dtype=float)
        # Linear interpolation of log-DF, flat extrapolation of the zero rate.
        log_df = np.interp(t, self.t, self._log_df,
                           left=-self.r[0] * t if np.ndim(t) == 0 else None)
        # np.interp clamps at the ends; handle short/long extrapolation via rate.
        log_df = np.where(t < self.t[0], -self.r[0] * t, log_df)
        log_df = np.where(t > self.t[-1], -self.r[-1] * t, log_df)
        return np.exp(log_df)

    def zero_rate(self, t: float | np.ndarray) -> np.ndarray:
        """Continuously-compounded zero rate at t."""
        t = np.asarray(t, dtype=float)
        df = self.discount(t)
        return np.where(t > 0, -np.log(df) / np.maximum(t, 1e-12), self.r[0])

    def forward_rate(self, t1: float, t2: float) -> float:
        """Simply-compounded forward rate between t1 and t2."""
        if t2 <= t1:
            raise ValueError("t2 must exceed t1")
        df1, df2 = float(self.discount(t1)), float(self.discount(t2))
        return (df1 / df2 - 1.0) / (t2 - t1)

    def shifted(self, bump: float) -> Curve:
        """Return a parallel-shifted curve (all zero rates + bump)."""
        return Curve(self.t.copy(), self.r + bump)

    def bump_pillar(self, i: int, bump: float) -> Curve:
        """Return a curve with a single pillar's zero rate bumped."""
        r = self.r.copy()
        r[i] += bump
        return Curve(self.t.copy(), r)

    @classmethod
    def flat(cls, rate: float, max_t: float = 30.0) -> Curve:
        """A flat curve at `rate` (useful for tests / demos)."""
        t = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
        t = t[t <= max_t]
        return cls(t, np.full_like(t, rate, dtype=float))
