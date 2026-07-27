from __future__ import annotations

import pandas as pd

from systematic_research.validation.walk_forward import nested_select, walk_forward_splits


def test_rolling_and_expanding_folds_are_strictly_separated() -> None:
    dates = pd.bdate_range("2020-01-01", periods=100)
    for window in ["rolling", "expanding"]:
        folds = walk_forward_splits(
            dates,
            train_periods=40,
            validation_periods=10,
            test_periods=10,
            step_periods=10,
            window=window,
            purge_periods=2,
            embargo_periods=3,
        )
        assert folds
        for fold in folds:
            fold.assert_separated()
            assert len(fold.train) >= 38
            assert fold.test.min() > fold.validation.max()


def test_nested_selection_never_uses_test_to_choose() -> None:
    dates = pd.bdate_range("2020-01-01", periods=70)
    fold = walk_forward_splits(
        dates,
        train_periods=40,
        validation_periods=10,
        test_periods=10,
        step_periods=10,
    )[0]
    calls = []

    def evaluator(candidate: int, evaluation_dates: pd.DatetimeIndex) -> float:
        calls.append((candidate, evaluation_dates))
        if evaluation_dates.equals(fold.validation):
            return float(candidate)
        return float(-candidate)

    winner, test_score, _ = nested_select([1, 2, 3], fold, evaluator)
    assert winner == 3
    assert test_score == -3
    test_calls = [
        candidate for candidate, evaluation_dates in calls if evaluation_dates.equals(fold.test)
    ]
    assert test_calls == [3]
