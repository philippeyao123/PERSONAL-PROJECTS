from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_research.examples.stat_arb import fit_pair_in_sample
from systematic_research.examples.tsmom import time_series_momentum_positions


def test_tsmom_uses_past_returns() -> None:
    dates = pd.bdate_range("2020-01-01", periods=30)
    frame = pd.DataFrame({"date": dates, "asset": "A", "return": 0.01})
    positions = time_series_momentum_positions(
        frame, lookback=5, volatility_window=5, target_volatility=0.1
    )
    assert (positions["target_weight"] > 0).all()


def test_pair_model_is_fit_only_in_sample() -> None:
    generator = np.random.default_rng(2)
    dates = pd.bdate_range("2020-01-01", periods=100)
    y = np.cumsum(generator.normal(size=100)) + 100
    x = 2.0 * y + generator.normal(scale=0.5, size=100)
    prices = pd.DataFrame({"X": x, "Y": y}, index=dates)
    train_end = dates[69]
    baseline = fit_pair_in_sample(prices, "X", "Y", train_end)
    prices.loc[dates[70] :, "X"] *= 100
    changed = fit_pair_in_sample(prices, "X", "Y", train_end)
    assert baseline == changed
