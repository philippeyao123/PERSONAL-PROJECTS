"""Availability-aware data access and historical universe membership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import pandas as pd

from systematic_research.data.schema import (
    DEFAULT_MARKET_DATA_SCHEMA,
    MarketDataSchema,
    validate_market_data,
)
from systematic_research.exceptions import DataValidationError, LeakageError


class PointInTimeData:
    """Immutable market data view that enforces historical availability."""

    def __init__(
        self,
        frame: pd.DataFrame,
        schema: MarketDataSchema = DEFAULT_MARKET_DATA_SCHEMA,
    ) -> None:
        self._schema = schema
        self._frame = validate_market_data(frame, schema)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    def as_of(self, timestamp: Union[pd.Timestamp, str]) -> pd.DataFrame:
        cutoff = pd.Timestamp(timestamp)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        mask = (self._frame[self._schema.date] <= cutoff) & (
            self._frame[self._schema.available_at] <= cutoff
        )
        return self._frame.loc[mask].copy()

    def assert_available(self, decisions: pd.DataFrame) -> None:
        required = {"date", "asset", "available_at"}
        if not required.issubset(decisions.columns):
            raise DataValidationError("decisions require date, asset and available_at")
        dates = pd.to_datetime(decisions["date"], utc=True)
        availability = pd.to_datetime(decisions["available_at"], utc=True)
        if (availability > dates).any():
            examples = decisions.loc[availability > dates, ["date", "asset", "available_at"]]
            raise LeakageError(
                f"future information used by decisions:\n{examples.head().to_string()}"
            )


@dataclass(frozen=True)
class UniverseMembership:
    asset: str
    valid_from: pd.Timestamp
    valid_to: Optional[pd.Timestamp] = None
    delisted: bool = False


class UniverseHistory:
    """Time-varying universe that retains entrants, exits and delisted assets."""

    def __init__(self, memberships: Iterable[UniverseMembership]) -> None:
        self._memberships: List[UniverseMembership] = list(memberships)
        if not self._memberships:
            raise DataValidationError("universe history cannot be empty")
        for member in self._memberships:
            if member.valid_to is not None and member.valid_to < member.valid_from:
                raise DataValidationError(f"invalid membership interval for {member.asset}")

    def members_at(self, timestamp: Union[pd.Timestamp, str]) -> List[str]:
        date = pd.Timestamp(timestamp)
        return sorted(
            member.asset
            for member in self._memberships
            if member.valid_from <= date and (member.valid_to is None or date <= member.valid_to)
        )

    @property
    def all_assets(self) -> List[str]:
        return sorted({member.asset for member in self._memberships})
