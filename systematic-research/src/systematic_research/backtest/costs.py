"""Transaction-cost and market-impact models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from systematic_research.exceptions import DataValidationError


@dataclass(frozen=True)
class LinearCosts:
    commission_bps: float = 0.5
    half_spread_bps: float = 1.0
    slippage_bps: float = 0.5

    def __post_init__(self) -> None:
        if min(self.commission_bps, self.half_spread_bps, self.slippage_bps) < 0:
            raise DataValidationError("linear cost parameters cannot be negative")

    @property
    def rate(self) -> float:
        return (self.commission_bps + self.half_spread_bps + self.slippage_bps) / 10_000.0

    def cost(self, absolute_weight_change: pd.Series) -> pd.Series:
        return absolute_weight_change * self.rate


@dataclass(frozen=True)
class SquareRootImpact:
    coefficient: float = 0.10
    max_participation: float = 0.20

    def __post_init__(self) -> None:
        if self.coefficient < 0 or not 0 < self.max_participation <= 1:
            raise DataValidationError("impact coefficient or participation limit is invalid")

    def cost(
        self,
        absolute_weight_change: pd.Series,
        volatility: pd.Series,
        adv: pd.Series,
        capital: float,
    ) -> pd.Series:
        """Return impact as portfolio-return drag."""
        if capital <= 0 or (adv <= 0).any() or (volatility < 0).any():
            raise DataValidationError("impact requires positive capital/ADV and non-negative vol")
        traded_notional = absolute_weight_change * capital
        participation = traded_notional / adv
        impact_rate = self.coefficient * volatility * np.sqrt(participation.clip(lower=0.0))
        return absolute_weight_change * impact_rate

    def participation(
        self, absolute_weight_change: pd.Series, adv: pd.Series, capital: float
    ) -> pd.Series:
        if capital <= 0 or (adv <= 0).any():
            raise DataValidationError("participation requires positive capital and ADV")
        return absolute_weight_change * capital / adv
