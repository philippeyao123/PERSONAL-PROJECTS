"""Synthetic point-in-time market with entries, exits and a delisting."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from systematic_research.data.point_in_time import UniverseHistory, UniverseMembership


def synthetic_market(
    *,
    seed: int = 42,
    periods: int = 1_000,
    assets: int = 12,
) -> Tuple[pd.DataFrame, UniverseHistory]:
    generator = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=periods, tz="UTC")
    asset_names = [f"ASSET_{index:02d}" for index in range(assets)]
    market_factor = generator.normal(0.0002, 0.008, periods)
    rows = []
    memberships = []
    for index, asset in enumerate(asset_names):
        start_index = 0 if index < assets - 2 else 100
        end_index = periods - 1 if index != 1 else periods - 120
        memberships.append(
            UniverseMembership(
                asset,
                dates[start_index],
                dates[end_index],
                delisted=index == 1,
            )
        )
        idiosyncratic = generator.normal(0.0, 0.009 + index * 0.0002, periods)
        trend = (
            pd.Series(market_factor + idiosyncratic).rolling(20, min_periods=1).mean().to_numpy()
        )
        returns = 0.35 * market_factor + idiosyncratic + 0.08 * np.roll(trend, 1)
        returns[0] = 0.0
        price = 100.0 * np.cumprod(1.0 + returns)
        volume = generator.lognormal(13.0, 0.35, periods)
        volatility = pd.Series(returns).rolling(20, min_periods=5).std().fillna(0.01).to_numpy()
        carry = 0.01 * np.sin(np.arange(periods) / 40.0 + index) + generator.normal(
            0, 0.002, periods
        )
        value = price / pd.Series(price).rolling(252, min_periods=1).mean().to_numpy()
        for position in range(start_index, end_index + 1):
            rows.append(
                {
                    "date": dates[position],
                    "asset": asset,
                    "price": price[position],
                    "volume": volume[position],
                    "adv": volume[position] * price[position],
                    "volatility": max(volatility[position], 0.001),
                    "carry": carry[position],
                    "value": value[position],
                    "sector": f"SECTOR_{index % 3}",
                    "available_at": dates[position],
                }
            )
    market_price = 100.0 * np.cumprod(1.0 + market_factor)
    for position, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "asset": "MARKET",
                "price": market_price[position],
                "volume": 20_000_000.0,
                "adv": 2_000_000_000.0,
                "volatility": 0.008,
                "carry": 0.0,
                "value": 1.0,
                "sector": "MARKET",
                "available_at": date,
            }
        )
    memberships.append(UniverseMembership("MARKET", dates[0], dates[-1]))
    return pd.DataFrame(rows), UniverseHistory(memberships)
