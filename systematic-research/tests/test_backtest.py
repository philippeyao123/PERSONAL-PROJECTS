from __future__ import annotations

import pandas as pd
import pytest

from systematic_research.backtest.costs import LinearCosts, SquareRootImpact
from systematic_research.backtest.engine import VectorizedBacktester
from systematic_research.exceptions import LeakageError


def _targets(return_frame: pd.DataFrame) -> pd.DataFrame:
    first_date = return_frame["date"].min()
    return pd.DataFrame(
        {
            "date": [first_date, first_date],
            "asset": ["A", "B"],
            "target_weight": [0.5, -0.5],
            "available_at": [first_date, first_date],
        }
    )


def test_execution_lag_and_pnl_reconcile(return_frame: pd.DataFrame) -> None:
    target = _targets(return_frame)
    zero_costs = LinearCosts(0, 0, 0)
    result_t0 = VectorizedBacktester(
        execution_lag=0, linear_costs=zero_costs, impact=SquareRootImpact(0)
    ).run(return_frame, target)
    result_t1 = VectorizedBacktester(
        execution_lag=1, linear_costs=zero_costs, impact=SquareRootImpact(0)
    ).run(return_frame, target)
    assert result_t0.execution_lag == 0
    assert result_t1.execution_lag == 1
    assert result_t1.daily.iloc[0]["gross_return"] == 0.0
    result_t1.assert_reconciled()
    assert (
        result_t0.positions.iloc[0]["executed_weight"]
        != result_t1.positions.iloc[0]["executed_weight"]
    )


def test_costs_reconcile_and_impact_increases_with_capital(return_frame: pd.DataFrame) -> None:
    target = _targets(return_frame)
    small = VectorizedBacktester(capital=1_000_000).run(return_frame, target)
    large = VectorizedBacktester(capital=100_000_000).run(return_frame, target)
    assert small.daily["linear_cost"].sum() > 0
    assert large.daily["impact_cost"].sum() > small.daily["impact_cost"].sum()
    assert (small.daily["net_return"] <= small.daily["gross_return"] + 1e-15).all()


def test_backtest_rejects_future_target(return_frame: pd.DataFrame) -> None:
    target = _targets(return_frame)
    target["available_at"] = target["date"] + pd.Timedelta(days=1)
    with pytest.raises(LeakageError):
        VectorizedBacktester().run(return_frame, target)
