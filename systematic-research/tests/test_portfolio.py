from __future__ import annotations

import pandas as pd

from systematic_research.portfolio import enforce_constraints, equal_weight_from_scores


def test_constraints_hold_after_clipping() -> None:
    date = pd.Timestamp("2020-01-01", tz="UTC")
    weights = pd.DataFrame(
        {
            "date": [date] * 4,
            "asset": list("ABCD"),
            "target_weight": [0.8, 0.3, -0.1, -0.1],
        }
    )
    constrained = enforce_constraints(
        weights,
        gross_limit=1.0,
        net_limit=0.05,
        concentration_limit=0.3,
    )
    assert constrained["target_weight"].abs().sum() <= 1.0 + 1e-12
    assert abs(constrained["target_weight"].sum()) <= 0.05 + 1e-12
    assert constrained["target_weight"].abs().max() <= 0.3 + 1e-12


def test_equal_weight_tails() -> None:
    date = pd.Timestamp("2020-01-01", tz="UTC")
    scores = pd.DataFrame({"date": [date] * 10, "asset": range(10), "score": range(10)})
    weights = equal_weight_from_scores(scores)
    assert abs(weights["target_weight"].sum()) < 1e-12
    assert abs(weights["target_weight"]).sum() == 1.0
