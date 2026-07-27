"""Momentum, carry and value features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from systematic_research.data.schema import validate_market_data
from systematic_research.exceptions import DataValidationError
from systematic_research.features.base import Feature


def _feature_frame(
    ordered: pd.DataFrame,
    values: pd.Series,
    availability: pd.Series,
    name: str,
) -> pd.DataFrame:
    result = ordered[["date", "asset"]].copy()
    result["feature"] = values
    result["available_at"] = pd.to_datetime(availability, utc=True)
    result["feature_name"] = name
    return result.dropna(subset=["feature", "available_at"]).reset_index(drop=True)


@dataclass(frozen=True)
class Momentum(Feature):
    name: str = "momentum"

    def compute(self, market_data: pd.DataFrame) -> pd.DataFrame:
        ordered = validate_market_data(market_data).sort_values(["asset", "date"])
        grouped_price = ordered.groupby("asset", sort=False)["price"]
        grouped_available = ordered.groupby("asset", sort=False)["available_at"]
        recent = grouped_price.shift(self.lag)
        old = grouped_price.shift(self.lag + self.lookback)
        values = recent / old - 1.0
        recent_availability = grouped_available.shift(self.lag)
        old_availability = grouped_available.shift(self.lag + self.lookback)
        availability = pd.concat([recent_availability, old_availability], axis=1).max(axis=1)
        return _feature_frame(ordered, values, availability, self.name)


@dataclass(frozen=True)
class Carry(Feature):
    name: str = "carry"

    def compute(self, market_data: pd.DataFrame) -> pd.DataFrame:
        if "carry" not in market_data:
            raise DataValidationError("carry feature requires a carry column")
        ordered = validate_market_data(market_data).sort_values(["asset", "date"])
        values = ordered.groupby("asset", sort=False)["carry"].shift(self.lag)
        availability = ordered.groupby("asset", sort=False)["available_at"].shift(self.lag)
        return _feature_frame(ordered, values, availability, self.name)


@dataclass(frozen=True)
class Value(Feature):
    name: str = "value"

    def compute(self, market_data: pd.DataFrame) -> pd.DataFrame:
        if "value" not in market_data:
            raise DataValidationError("value feature requires a value column")
        ordered = validate_market_data(market_data).sort_values(["asset", "date"])
        raw = ordered.groupby("asset", sort=False)["value"].shift(self.lag)
        values = -raw.replace([np.inf, -np.inf], np.nan)
        availability = ordered.groupby("asset", sort=False)["available_at"].shift(self.lag)
        return _feature_frame(ordered, values, availability, self.name)
