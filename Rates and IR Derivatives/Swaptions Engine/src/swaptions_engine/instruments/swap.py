"""Interest-rate swap and swaption instruments.

A `VanillaSwap` is a fixed-for-floating IRS defined by its schedule. The key
quantities the pricing layer needs are the annuity (PV of a 1bp fixed leg) and
the forward swap rate; both are computed off a `Curve`.

A `Swaption` is the option to enter a swap at expiry. European and Bermudan
exercise styles are supported (the exercise dates drive the LSM pricer).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from swaptions_engine.curve.yield_curve import Curve


@dataclass
class VanillaSwap:
    """Fixed-for-floating interest-rate swap.

    Attributes
    ----------
    start : float
        Swap start (year fraction from valuation date).
    tenor : float
        Swap length in years.
    fixed_rate : float
        Fixed leg rate (the strike when embedded in a swaption).
    freq : int
        Fixed-leg payments per year (e.g. 2 for semi-annual).
    notional : float
    pay_fixed : bool
        True = payer swap (pay fixed, receive float).
    """

    start: float
    tenor: float
    fixed_rate: float
    freq: int = 2
    notional: float = 1.0
    pay_fixed: bool = True
    pay_times: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = int(round(self.tenor * self.freq))
        dt = 1.0 / self.freq
        self.pay_times = self.start + dt * np.arange(1, n + 1)

    @property
    def accrual(self) -> float:
        return 1.0 / self.freq

    def annuity(self, curve: Curve, as_of: float | None = None) -> float:
        """PV of the fixed leg's unit-coupon (the level / annuity factor).

        If `as_of` is given, discounting is to that date (forward annuity);
        otherwise to today.
        """
        dfs = curve.discount(self.pay_times)
        ann = self.accrual * float(np.sum(dfs))
        if as_of is not None and as_of > 0:
            ann /= float(curve.discount(as_of))
        return ann

    def forward_swap_rate(self, curve: Curve) -> float:
        """Par forward swap rate = (P(start) - P(end)) / annuity."""
        p_start = float(curve.discount(self.start))
        p_end = float(curve.discount(self.pay_times[-1]))
        ann = self.annuity(curve)
        return (p_start - p_end) / ann

    def npv(self, curve: Curve) -> float:
        """Present value of the swap to the fixed-rate payer/receiver."""
        ann = self.annuity(curve)
        fwd = self.forward_swap_rate(curve)
        pv_payer = self.notional * ann * (fwd - self.fixed_rate)
        return pv_payer if self.pay_fixed else -pv_payer


@dataclass
class Swaption:
    """Option to enter `underlying` swap. European or Bermudan.

    For European, `exercise_times` is just the swap start. For Bermudan, it is
    the set of allowed exercise dates (each a coupon date of the underlying).
    """

    underlying: VanillaSwap
    exercise_times: np.ndarray
    style: str = "european"  # "european" | "bermudan"

    def __post_init__(self) -> None:
        self.exercise_times = np.atleast_1d(
            np.asarray(self.exercise_times, dtype=float)
        )
        if self.style not in {"european", "bermudan"}:
            raise ValueError("style must be 'european' or 'bermudan'")
        if self.style == "european" and len(self.exercise_times) != 1:
            raise ValueError("european swaption needs exactly one exercise time")

    @property
    def expiry(self) -> float:
        return float(self.exercise_times[0])
