from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_research.exceptions import LeakageError
from systematic_research.features.factors import Carry, Momentum, Value
from systematic_research.signals.pipeline import SignalPipeline
from systematic_research.signals.transforms import cross_sectional_rank, rolling_zscore


def test_features_declare_lag_and_availability(market_data: pd.DataFrame) -> None:
    for feature in [
        Momentum(lookback=3, lag=1),
        Carry(lookback=1, lag=1),
        Value(lookback=1, lag=1),
    ]:
        output = feature.compute(market_data)
        assert {"date", "asset", "feature", "available_at"}.issubset(output)
        assert (output["available_at"] <= output["date"]).all()


def test_pipeline_weights_have_zero_net_and_declared_gross(market_data: pd.DataFrame) -> None:
    result = SignalPipeline(Momentum(lookback=3, lag=1)).run(market_data)
    gross = result.groupby("date")["target_weight"].apply(lambda values: values.abs().sum())
    net = result.groupby("date")["target_weight"].sum()
    assert np.allclose(gross, 1.0)
    assert (net.abs() < 1e-12).all()


def test_rolling_zscore_parameters_use_only_past() -> None:
    dates = pd.bdate_range("2020-01-01", periods=8, tz="UTC")
    base = pd.DataFrame(
        {
            "date": dates,
            "asset": "A",
            "feature": range(8),
            "available_at": dates,
        }
    )
    changed = base.copy()
    changed.loc[7, "feature"] = 1_000_000
    original_score = rolling_zscore(base, 3, 3).loc[6, "score"]
    changed_score = rolling_zscore(changed, 3, 3).loc[6, "score"]
    assert original_score == changed_score


def test_cross_sectional_rank_is_bounded(market_data: pd.DataFrame) -> None:
    features = Momentum(lookback=2, lag=1).compute(market_data)
    ranked = cross_sectional_rank(features)
    assert ranked["score"].between(-1, 1).all()


def test_future_availability_is_rejected(market_data: pd.DataFrame) -> None:
    contaminated = market_data.copy()
    contaminated["available_at"] = contaminated["date"] + pd.Timedelta(days=30)
    with pytest.raises(LeakageError):
        SignalPipeline(Momentum(lookback=2, lag=1)).run(contaminated)
