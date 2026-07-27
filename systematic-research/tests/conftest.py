from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=12, tz="UTC")
    rows = []
    for asset_index, asset in enumerate(["A", "B", "C"]):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "price": 100.0 * (1.0 + 0.01 * asset_index + 0.005 * index),
                    "volume": 100_000.0,
                    "adv": 10_000_000.0,
                    "volatility": 0.01,
                    "carry": 0.01 * asset_index,
                    "value": 1.0 + 0.1 * asset_index,
                    "available_at": date,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def return_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=5, tz="UTC")
    rows = []
    for asset, values in {
        "A": [0.0, 0.01, 0.02, -0.01, 0.01],
        "B": [0.0, -0.01, 0, 0.02, 0],
    }.items():
        for date, value in zip(dates, values):
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "return": value,
                    "adv": 10_000_000.0,
                    "volatility": 0.01,
                }
            )
    return pd.DataFrame(rows)
