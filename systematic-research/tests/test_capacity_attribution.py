from __future__ import annotations

import pandas as pd

from systematic_research.backtest.attribution import attribution_by
from systematic_research.backtest.engine import VectorizedBacktester
from systematic_research.capacity import capacity_curve


def test_capacity_and_attribution(return_frame: pd.DataFrame) -> None:
    first = return_frame["date"].min()
    targets = pd.DataFrame(
        {
            "date": [first, first],
            "asset": ["A", "B"],
            "target_weight": [0.5, -0.5],
            "available_at": [first, first],
        }
    )
    result = VectorizedBacktester().run(return_frame, targets)
    attribution = attribution_by(result, "asset")
    assert attribution["contribution"].sum() == result.daily["gross_return"].sum()
    curve = capacity_curve(result, [1_000_000, 10_000_000, 100_000_000])
    assert curve["annualized_impact"].is_monotonic_increasing
    assert curve["max_participation"].is_monotonic_increasing
