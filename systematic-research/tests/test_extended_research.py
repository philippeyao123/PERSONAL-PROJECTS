from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_research.backtest.attribution import (
    period_attribution,
    signal_attribution,
)
from systematic_research.backtest.costs import LinearCosts
from systematic_research.backtest.engine import BacktestResult, VectorizedBacktester
from systematic_research.data.calendar import align_to_calendar
from systematic_research.data.returns import compute_returns, excess_returns
from systematic_research.data.schema import validate_market_data
from systematic_research.exceptions import DataValidationError
from systematic_research.portfolio import enforce_constraints, equal_weight_from_scores
from systematic_research.signals.transforms import (
    cross_sectional_rank,
    rolling_zscore,
    winsorize_past,
)
from systematic_research.statistics import (
    deflated_sharpe_ratio,
    information_coefficient,
    probabilistic_sharpe_ratio,
    return_moments,
)


def _backtest_result() -> tuple[pd.DataFrame, BacktestResult]:
    dates = pd.date_range("2023-01-01", periods=12, freq="D")
    returns = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "asset": ["A", "B"] * len(dates),
            "return": [0.01, -0.005] * len(dates),
        }
    )
    targets = returns[["date", "asset"]].copy()
    targets["target_weight"] = [0.5, -0.5] * len(dates)
    targets["available_at"] = targets["date"]
    result = VectorizedBacktester(
        execution_lag=1,
        linear_costs=LinearCosts(commission_bps=0.5, half_spread_bps=0.5, slippage_bps=0.0),
    ).run(returns, targets)
    return returns, result


def test_period_and_signal_attribution_reconcile() -> None:
    _, result = _backtest_result()
    monthly = period_attribution(result, frequency="M")
    assert monthly["contribution"].notna().all()
    assert monthly["contribution"].sum() == pytest.approx(result.daily["gross_return"].sum())

    daily_gross = result.daily.set_index("date")["gross_return"]
    attributed = signal_attribution(
        {"momentum": daily_gross * 0.6, "value": daily_gross * 0.4},
        daily_gross,
    )
    pd.testing.assert_series_equal(attributed.sum(axis=1), daily_gross, check_names=False)


def test_signal_transforms_and_validation() -> None:
    frame = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2024-01-01", periods=5), 3),
            "asset": list("ABC") * 5,
            "feature": [0.0, 1.0, 100.0] * 5,
            "available_at": np.repeat(pd.date_range("2024-01-01", periods=5), 3),
        }
    )
    winsorized = winsorize_past(frame, window=2, lower=0.1, upper=0.9)
    assert "feature" in winsorized

    ranked = cross_sectional_rank(frame)
    assert ranked["score"].between(-1.0, 1.0).all()

    rolling = rolling_zscore(frame, window=3, min_periods=2)
    assert rolling.groupby("asset")["score"].head(1).isna().all()

    with pytest.raises(DataValidationError):
        winsorize_past(frame, window=2, lower=0.9, upper=0.1)
    with pytest.raises(DataValidationError):
        rolling_zscore(frame, window=1)


def test_data_helpers_and_calendar_guards() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset": ["A", "A", "A"],
            "price": [100.0, 110.0, 121.0],
        }
    )
    simple = compute_returns(market, kind="simple")
    log_returns = compute_returns(market, kind="log")
    assert simple.iloc[1] == pytest.approx(0.1)
    assert log_returns.iloc[1] == pytest.approx(np.log(1.1))
    assert excess_returns(simple, 0.001).iloc[1] == pytest.approx(0.099)

    aligned = align_to_calendar(
        market,
        pd.DatetimeIndex(pd.date_range("2024-01-01", periods=4)),
        allow_price_fill=False,
    )
    assert len(aligned) == 4
    with pytest.raises(DataValidationError):
        align_to_calendar(
            market,
            pd.DatetimeIndex(pd.date_range("2024-01-01", periods=4)),
            allow_price_fill=True,
        )
    with pytest.raises(DataValidationError):
        compute_returns(market, kind="unsupported")  # type: ignore[arg-type]


def test_portfolio_constraints_and_schema_failures() -> None:
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "asset": list("ABCD"),
            "score": [-2.0, -1.0, 1.0, 2.0],
        }
    )
    portfolio = enforce_constraints(
        equal_weight_from_scores(signals),
        gross_limit=1.0,
        net_limit=0.1,
        concentration_limit=0.3,
    )
    assert portfolio["target_weight"].abs().sum() <= 1.0 + 1e-12
    assert abs(portfolio["target_weight"].sum()) <= 0.1 + 1e-12

    invalid_market = pd.DataFrame(
        {
            "date": ["not-a-date"],
            "asset": [""],
            "price": [-1.0],
            "volume": [-1.0],
            "available_at": ["2023-12-31"],
        }
    )
    with pytest.raises(DataValidationError):
        validate_market_data(invalid_market)


def test_research_statistics_cover_edge_cases() -> None:
    rng = np.random.default_rng(17)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    observed_sharpe = float(returns.mean() / returns.std(ddof=1))
    skewness, kurtosis = return_moments(returns)
    psr = probabilistic_sharpe_ratio(
        observed_sharpe,
        benchmark_sharpe=0.0,
        observations=len(returns),
        skewness=skewness,
        kurtosis=kurtosis,
    )
    dsr = deflated_sharpe_ratio(
        observed_sharpe,
        pd.Series(rng.normal(0.0, 0.2, 20)),
        observations=len(returns),
        skewness=skewness,
        kurtosis=kurtosis,
    )
    assert 0.0 <= psr <= 1.0
    assert 0.0 <= dsr <= 1.0

    panel = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2020-01-01", periods=5), 4),
            "score": np.tile([-1.0, -0.5, 0.5, 1.0], 5),
            "forward_return": np.tile([-0.02, -0.01, 0.01, 0.02], 5),
        }
    )
    assert information_coefficient(panel).eq(1.0).all()
