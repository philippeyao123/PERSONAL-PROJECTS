"""Canonical long-form market data schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from systematic_research.exceptions import DataValidationError


@dataclass(frozen=True)
class MarketDataSchema:
    """Column names required by the research pipeline."""

    date: str = "date"
    asset: str = "asset"
    price: str = "price"
    volume: str = "volume"
    available_at: str = "available_at"
    optional: FrozenSet[str] = frozenset(
        {"return", "signal", "sector", "adv", "volatility", "carry", "value"}
    )

    @property
    def required(self) -> FrozenSet[str]:
        return frozenset({self.date, self.asset, self.price, self.volume, self.available_at})


DEFAULT_MARKET_DATA_SCHEMA = MarketDataSchema()


def validate_market_data(
    frame: pd.DataFrame,
    schema: MarketDataSchema = DEFAULT_MARKET_DATA_SCHEMA,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Validate and canonicalize market data without filling missing observations."""
    missing = schema.required.difference(frame.columns)
    if missing:
        raise DataValidationError(f"market data is missing required columns: {sorted(missing)}")
    result = frame.copy() if copy else frame
    for column in (schema.date, schema.available_at):
        if not is_datetime64_any_dtype(result[column]):
            try:
                result[column] = pd.to_datetime(result[column], utc=True)
            except (TypeError, ValueError) as error:
                raise DataValidationError(f"{column} must contain valid timestamps") from error
        elif getattr(result[column].dt, "tz", None) is None:
            result[column] = result[column].dt.tz_localize("UTC")
    if not is_numeric_dtype(result[schema.price]) or not is_numeric_dtype(result[schema.volume]):
        raise DataValidationError("price and volume columns must be numeric")
    if not np.isfinite(result[schema.price]).all() or (result[schema.price] <= 0).any():
        raise DataValidationError("prices must be finite and strictly positive")
    if not np.isfinite(result[schema.volume].dropna()).all() or (result[schema.volume] < 0).any():
        raise DataValidationError("volumes must be finite and non-negative")
    if result[schema.asset].isna().any() or (result[schema.asset].astype(str).str.len() == 0).any():
        raise DataValidationError("asset identifiers cannot be missing or empty")
    if result.duplicated([schema.date, schema.asset]).any():
        raise DataValidationError("duplicate date/asset observations are not allowed")
    if (result[schema.available_at] < result[schema.date]).any():
        raise DataValidationError("available_at cannot precede the observation date")
    return result.sort_values([schema.date, schema.asset], kind="stable").reset_index(drop=True)
