"""Feature-to-score-to-position pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from systematic_research.exceptions import DataValidationError, LeakageError
from systematic_research.features.base import Feature
from systematic_research.signals.transforms import cross_sectional_rank, rolling_zscore


def score_to_weights(scores: pd.DataFrame, gross_target: float = 1.0) -> pd.DataFrame:
    """Demean scores cross-sectionally and normalize absolute exposure."""
    if gross_target <= 0 or not {"date", "asset", "score", "available_at"}.issubset(scores.columns):
        raise DataValidationError("scores are incomplete or gross_target is invalid")
    result = scores.copy()
    result["score"] = result["score"] - result.groupby("date")["score"].transform("mean")
    gross = result.groupby("date")["score"].transform(lambda values: values.abs().sum())
    result["target_weight"] = result["score"].div(gross.where(gross > 0)).fillna(0.0) * gross_target
    return result


@dataclass(frozen=True)
class SignalPipeline:
    feature: Feature
    normalization: Literal["cross_sectional_rank", "rolling_zscore"] = "cross_sectional_rank"
    normalization_window: int = 252
    gross_target: float = 1.0

    def run(self, market_data: pd.DataFrame) -> pd.DataFrame:
        features = self.feature.compute(market_data)
        dates = pd.to_datetime(features["date"], utc=True)
        availability = pd.to_datetime(features["available_at"], utc=True)
        if (availability > dates).any():
            raise LeakageError("feature is not available at its decision timestamp")
        if self.normalization == "cross_sectional_rank":
            scores = cross_sectional_rank(features)
        elif self.normalization == "rolling_zscore":
            scores = rolling_zscore(features, self.normalization_window)
        else:
            raise DataValidationError(f"unknown normalization: {self.normalization}")
        return score_to_weights(scores.dropna(subset=["score"]), self.gross_target)
