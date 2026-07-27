from __future__ import annotations

import pandas as pd
import pytest

from systematic_research.risk import (
    annualized_volatility,
    expected_shortfall,
    max_drawdown,
    performance_summary,
    value_at_risk,
)
from systematic_research.statistics import (
    deflated_sharpe_ratio,
    information_coefficient,
    probabilistic_sharpe_ratio,
)


def test_known_drawdown_and_tail_risk() -> None:
    returns = pd.Series([0.10, -0.20, 0.05, -0.10])
    assert max_drawdown(returns) == pytest.approx(-0.244)
    assert value_at_risk(returns, 0.75) >= 0
    assert expected_shortfall(returns, 0.75) >= value_at_risk(returns, 0.75)
    assert annualized_volatility(returns) > 0
    assert "sharpe" in performance_summary(returns)


def test_psr_and_dsr_are_probabilities() -> None:
    psr = probabilistic_sharpe_ratio(1.0, 0.0, 252)
    dsr = deflated_sharpe_ratio(1.0, pd.Series([0.1, 0.3, 0.5, 0.7]), 252)
    assert 0 <= psr <= 1
    assert 0 <= dsr <= 1
    assert psr > 0.99


def test_cross_sectional_information_coefficient() -> None:
    dates = [pd.Timestamp("2020-01-01")] * 4
    frame = pd.DataFrame(
        {
            "date": dates,
            "asset": list("ABCD"),
            "score": [1, 2, 3, 4],
            "forward_return": [0.1, 0.2, 0.3, 0.4],
        }
    )
    ic = information_coefficient(frame)
    assert ic.iloc[0] == pytest.approx(1.0)
