from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_research.data.calendar import align_to_calendar
from systematic_research.data.point_in_time import (
    PointInTimeData,
    UniverseHistory,
    UniverseMembership,
)
from systematic_research.data.returns import compute_returns, roll_adjusted_futures_returns
from systematic_research.data.schema import validate_market_data
from systematic_research.exceptions import DataValidationError


def test_schema_and_point_in_time_filter(market_data: pd.DataFrame) -> None:
    delayed = market_data.copy()
    delayed.loc[0, "available_at"] = delayed.loc[0, "date"] + pd.Timedelta(days=3)
    data = PointInTimeData(delayed)
    cutoff = delayed.loc[0, "date"] + pd.Timedelta(days=1)
    visible = data.as_of(cutoff)
    assert not ((visible["date"] == delayed.loc[0, "date"]) & (visible["asset"] == "A")).any()


def test_schema_rejects_duplicates_and_invalid_prices(market_data: pd.DataFrame) -> None:
    duplicate = pd.concat([market_data, market_data.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_market_data(duplicate)
    invalid = market_data.copy()
    invalid.loc[0, "price"] = -1.0
    with pytest.raises(DataValidationError, match="positive"):
        validate_market_data(invalid)


def test_returns_are_per_asset_and_futures_roll_safe(market_data: pd.DataFrame) -> None:
    returns = compute_returns(market_data)
    first_rows = market_data.groupby("asset", sort=False).head(1).index
    assert returns.loc[first_rows].isna().all()
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4),
            "asset": ["F"] * 4,
            "contract": ["H", "H", "M", "M"],
            "price": [100.0, 101.0, 120.0, 121.2],
        }
    )
    adjusted = roll_adjusted_futures_returns(frame)
    assert np.isnan(adjusted.iloc[2])
    assert adjusted.iloc[3] == pytest.approx(0.01)


def test_calendar_never_forward_fills_prices(market_data: pd.DataFrame) -> None:
    calendar = pd.bdate_range("2020-01-01", periods=15, tz="UTC")
    aligned = align_to_calendar(market_data, calendar)
    assert aligned["price"].isna().any()
    with pytest.raises(DataValidationError, match="forward-fill"):
        align_to_calendar(market_data, calendar, allow_price_fill=True)


def test_universe_retains_delisted_assets() -> None:
    history = UniverseHistory(
        [
            UniverseMembership("LIVE", pd.Timestamp("2020-01-01")),
            UniverseMembership(
                "DELISTED",
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-06-30"),
                True,
            ),
        ]
    )
    assert "DELISTED" in history.members_at("2020-05-01")
    assert "DELISTED" not in history.members_at("2021-01-01")
    assert "DELISTED" in history.all_assets
