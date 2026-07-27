"""Point-in-time systematic research toolkit."""

from systematic_research.backtest.engine import BacktestResult, VectorizedBacktester
from systematic_research.config import ExperimentConfig
from systematic_research.data.schema import MarketDataSchema, validate_market_data
from systematic_research.features.base import Feature

__all__ = [
    "BacktestResult",
    "ExperimentConfig",
    "Feature",
    "MarketDataSchema",
    "VectorizedBacktester",
    "validate_market_data",
]

__version__ = "0.1.0"
