"""Rolling/expanding train-validation-test splits with purge and embargo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, TypeVar

import pandas as pd

from systematic_research.exceptions import DataValidationError

T = TypeVar("T")


@dataclass(frozen=True)
class WalkForwardFold:
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def assert_separated(self) -> None:
        if self.train.intersection(self.validation).size:
            raise AssertionError("train and validation overlap")
        if self.train.intersection(self.test).size:
            raise AssertionError("train and test overlap")
        if self.validation.intersection(self.test).size:
            raise AssertionError("validation and test overlap")
        if not (self.train.max() < self.validation.min() < self.test.min()):
            raise AssertionError("folds are not chronologically ordered")


def walk_forward_splits(
    dates: Iterable[pd.Timestamp],
    *,
    train_periods: int,
    validation_periods: int,
    test_periods: int,
    step_periods: int,
    window: str = "rolling",
    purge_periods: int = 0,
    embargo_periods: int = 0,
) -> List[WalkForwardFold]:
    unique = pd.DatetimeIndex(dates).unique().sort_values()
    if window not in {"rolling", "expanding"}:
        raise DataValidationError("window must be rolling or expanding")
    if min(train_periods, validation_periods, test_periods, step_periods) <= 0:
        raise DataValidationError("window sizes must be positive")
    if purge_periods < 0 or embargo_periods < 0 or purge_periods >= train_periods:
        raise DataValidationError("purge/embargo parameters are invalid")
    folds = []
    cursor = 0
    while True:
        train_start = 0 if window == "expanding" else cursor
        raw_train_end = cursor + train_periods
        validation_start = raw_train_end
        validation_end = validation_start + validation_periods
        test_start = validation_end + embargo_periods
        test_end = test_start + test_periods
        if test_end > len(unique):
            break
        fold = WalkForwardFold(
            train=unique[train_start : raw_train_end - purge_periods],
            validation=unique[validation_start:validation_end],
            test=unique[test_start:test_end],
        )
        fold.assert_separated()
        folds.append(fold)
        cursor += step_periods
    return folds


def nested_select(
    candidates: Sequence[T],
    fold: WalkForwardFold,
    evaluator: Callable[[T, pd.DatetimeIndex], float],
) -> Tuple[T, float, Dict[str, float]]:
    """Choose only on validation, then evaluate the winner once on test."""
    if not candidates:
        raise DataValidationError("nested selection requires candidates")
    validation_scores = {
        repr(candidate): evaluator(candidate, fold.validation) for candidate in candidates
    }
    winner = max(candidates, key=lambda candidate: validation_scores[repr(candidate)])
    test_score = evaluator(winner, fold.test)
    return winner, test_score, validation_scores


def assert_no_label_overlap(
    train_labels: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    *,
    start_column: str = "label_start",
    end_column: str = "label_end",
) -> None:
    required = {start_column, end_column}
    if not required.issubset(train_labels.columns):
        raise DataValidationError(f"labels require {sorted(required)}")
    if test_dates.empty:
        return
    overlap = (train_labels[start_column] <= test_dates.max()) & (
        train_labels[end_column] >= test_dates.min()
    )
    if overlap.any():
        raise DataValidationError("purging failed: train labels overlap the test interval")
