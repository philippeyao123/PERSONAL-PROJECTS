"""Auditable vectorized portfolio accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from systematic_research.backtest.costs import LinearCosts, SquareRootImpact
from systematic_research.exceptions import DataValidationError, LeakageError


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    positions: pd.DataFrame
    execution_lag: int
    capital: float

    def assert_reconciled(self, tolerance: float = 1e-12) -> None:
        contribution = self.positions.groupby("date")["gross_contribution"].sum()
        aligned = self.daily.set_index("date")["gross_return"]
        if not np.allclose(
            contribution.reindex(aligned.index, fill_value=0.0), aligned, atol=tolerance
        ):
            raise AssertionError("asset contributions do not reconcile with gross P&L")
        expected_net = (
            self.daily["gross_return"]
            + self.daily["cash_return"]
            - self.daily["linear_cost"]
            - self.daily["impact_cost"]
        )
        if not np.allclose(expected_net, self.daily["net_return"], atol=tolerance):
            raise AssertionError("gross/cost/net P&L does not reconcile")


class VectorizedBacktester:
    """Apply availability-safe targets to future returns with explicit execution lag."""

    def __init__(
        self,
        *,
        execution_lag: int = 1,
        linear_costs: Optional[LinearCosts] = None,
        impact: Optional[SquareRootImpact] = None,
        capital: float = 10_000_000.0,
    ) -> None:
        if execution_lag < 0 or capital <= 0:
            raise DataValidationError("execution lag and capital are invalid")
        self.execution_lag = execution_lag
        self.linear_costs = linear_costs or LinearCosts()
        self.impact = impact or SquareRootImpact()
        self.capital = capital

    def run(
        self,
        returns: pd.DataFrame,
        targets: pd.DataFrame,
        *,
        cash_returns: Optional[pd.Series] = None,
    ) -> BacktestResult:
        required_returns = {"date", "asset", "return"}
        required_targets = {"date", "asset", "target_weight", "available_at"}
        if not required_returns.issubset(returns.columns):
            raise DataValidationError(f"returns require {sorted(required_returns)}")
        if not required_targets.issubset(targets.columns):
            raise DataValidationError(f"targets require {sorted(required_targets)}")
        market = returns.copy()
        desired = targets.copy()
        market["date"] = pd.to_datetime(market["date"], utc=True)
        desired["date"] = pd.to_datetime(desired["date"], utc=True)
        desired["available_at"] = pd.to_datetime(desired["available_at"], utc=True)
        if (desired["available_at"] > desired["date"]).any():
            raise LeakageError("a target uses information unavailable on its decision date")
        if (
            market.duplicated(["date", "asset"]).any()
            or desired.duplicated(["date", "asset"]).any()
        ):
            raise DataValidationError("returns and targets require unique date/asset rows")
        market = market.sort_values(["asset", "date"], kind="stable")
        desired = desired[["date", "asset", "target_weight", "available_at"]]
        merged = market.merge(desired, on=["date", "asset"], how="left", validate="one_to_one")
        grouped = merged.groupby("asset", sort=False)
        merged["desired_weight"] = grouped["target_weight"].ffill().fillna(0.0)
        merged["executed_weight"] = (
            merged.groupby("asset", sort=False)["desired_weight"]
            .shift(self.execution_lag)
            .fillna(0.0)
        )
        merged["weight_change"] = merged["executed_weight"] - merged.groupby("asset", sort=False)[
            "executed_weight"
        ].shift(1).fillna(0.0)
        merged["absolute_weight_change"] = merged["weight_change"].abs()
        merged["gross_contribution"] = merged["executed_weight"] * merged["return"].fillna(0.0)
        merged["linear_cost_contribution"] = self.linear_costs.cost(
            merged["absolute_weight_change"]
        )
        if {"adv", "volatility"}.issubset(merged.columns):
            if merged[["adv", "volatility"]].isna().any().any():
                raise DataValidationError(
                    "ADV and volatility cannot be missing when impact is used"
                )
            merged["impact_cost_contribution"] = self.impact.cost(
                merged["absolute_weight_change"],
                merged["volatility"],
                merged["adv"],
                self.capital,
            )
            merged["participation"] = self.impact.participation(
                merged["absolute_weight_change"], merged["adv"], self.capital
            )
        else:
            merged["impact_cost_contribution"] = 0.0
            merged["participation"] = 0.0
        by_date = merged.groupby("date", sort=True)
        daily = pd.DataFrame(
            {
                "gross_return": by_date["gross_contribution"].sum(),
                "linear_cost": by_date["linear_cost_contribution"].sum(),
                "impact_cost": by_date["impact_cost_contribution"].sum(),
                "turnover": 0.5 * by_date["absolute_weight_change"].sum(),
                "gross_exposure": by_date["executed_weight"].apply(
                    lambda values: values.abs().sum()
                ),
                "net_exposure": by_date["executed_weight"].sum(),
                "max_participation": by_date["participation"].max(),
            }
        )
        daily["cash_weight"] = 1.0 - daily["net_exposure"]
        if cash_returns is None:
            daily["cash_return"] = 0.0
        else:
            cash = cash_returns.copy()
            cash.index = pd.to_datetime(cash.index, utc=True)
            daily["cash_return"] = daily["cash_weight"] * cash.reindex(daily.index).fillna(0.0)
        daily["net_return"] = (
            daily["gross_return"]
            + daily["cash_return"]
            - daily["linear_cost"]
            - daily["impact_cost"]
        )
        daily = daily.reset_index()
        result = BacktestResult(daily, merged, self.execution_lag, self.capital)
        result.assert_reconciled()
        return result
