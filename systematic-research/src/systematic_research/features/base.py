"""Common feature interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from systematic_research.exceptions import ConfigurationError


@dataclass(frozen=True)
class Feature(ABC):
    """A feature must declare the observations it consumes and its release lag."""

    lookback: int
    lag: int = 1
    name: str = "feature"

    def __post_init__(self) -> None:
        if self.lookback < 1 or self.lag < 0:
            raise ConfigurationError("feature lookback must be positive and lag non-negative")

    @abstractmethod
    def compute(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Return date, asset, feature and available_at columns."""
