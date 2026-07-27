"""Serializable experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml

from systematic_research.exceptions import ConfigurationError
from systematic_research.tracking import stable_hash


@dataclass(frozen=True)
class DataConfig:
    frequency: str = "D"
    return_kind: str = "simple"
    benchmark_asset: str = "MARKET"

    def __post_init__(self) -> None:
        if self.return_kind not in {"simple", "log"}:
            raise ConfigurationError("return_kind must be 'simple' or 'log'")


@dataclass(frozen=True)
class SignalConfig:
    feature: str = "momentum"
    lookback: int = 63
    lag: int = 1
    normalization: str = "cross_sectional_rank"

    def __post_init__(self) -> None:
        if self.lookback < 2 or self.lag < 0:
            raise ConfigurationError("feature lookback must be >= 2 and lag must be >= 0")


@dataclass(frozen=True)
class PortfolioConfig:
    gross_limit: float = 1.0
    net_limit: float = 0.10
    concentration_limit: float = 0.15
    volatility_target: float = 0.10

    def __post_init__(self) -> None:
        if not (
            self.gross_limit > 0
            and self.net_limit >= 0
            and 0 < self.concentration_limit <= self.gross_limit
            and self.volatility_target > 0
        ):
            raise ConfigurationError("invalid portfolio limits")


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float = 0.5
    half_spread_bps: float = 1.0
    slippage_bps: float = 0.5
    impact_coefficient: float = 0.10
    capital: float = 10_000_000.0

    def __post_init__(self) -> None:
        if (
            min(
                self.commission_bps,
                self.half_spread_bps,
                self.slippage_bps,
                self.impact_coefficient,
                self.capital,
            )
            < 0
        ):
            raise ConfigurationError("cost inputs cannot be negative")


@dataclass(frozen=True)
class ValidationConfig:
    train_periods: int = 504
    validation_periods: int = 126
    test_periods: int = 126
    step_periods: int = 126
    window: str = "rolling"
    purge_periods: int = 0
    embargo_periods: int = 0

    def __post_init__(self) -> None:
        if self.window not in {"rolling", "expanding"}:
            raise ConfigurationError("window must be 'rolling' or 'expanding'")
        if (
            min(
                self.train_periods,
                self.validation_periods,
                self.test_periods,
                self.step_periods,
            )
            <= 0
        ):
            raise ConfigurationError("walk-forward window sizes must be positive")
        if self.purge_periods < 0 or self.embargo_periods < 0:
            raise ConfigurationError("purge and embargo must be non-negative")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "flagship_momentum"
    seed: int = 42
    periods_per_year: int = 252
    execution_lag: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self) -> None:
        if not self.name or self.periods_per_year <= 0 or self.execution_lag < 0:
            raise ConfigurationError("name, periods_per_year and execution_lag are invalid")

    @property
    def experiment_id(self) -> str:
        """Stable hash of every economically relevant input."""
        return stable_hash(asdict(self))[:12]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        return cls(
            name=str(raw.get("name", "flagship_momentum")),
            seed=int(raw.get("seed", 42)),
            periods_per_year=int(raw.get("periods_per_year", 252)),
            execution_lag=int(raw.get("execution_lag", 1)),
            data=DataConfig(**dict(raw.get("data", {}))),
            signal=SignalConfig(**dict(raw.get("signal", {}))),
            portfolio=PortfolioConfig(**dict(raw.get("portfolio", {}))),
            costs=CostConfig(**dict(raw.get("costs", {}))),
            validation=ValidationConfig(**dict(raw.get("validation", {}))),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise ConfigurationError("configuration root must be a mapping")
        return cls.from_mapping(raw)

    def to_yaml(self, path: Union[str, Path]) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=True)
