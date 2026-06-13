"""Transaction cost model.

Total cost per rebalance = spread cost + market impact + borrow cost on shorts.

Market impact uses the square-root law (Almgren et al.): cost in bps scales
with sqrt(participation), where participation = traded notional / ADV. This is
the standard reduced-form impact model used across the buy-side. The headline
number a systematic desk cares about is *net* Sharpe after these costs, not
the gross backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostParams:
    half_spread_bps: float = 2.0      # half bid-ask spread, in bps
    impact_coef_bps: float = 10.0     # impact at 100% ADV participation, bps
    adv_participation: float = 0.05   # assumed fraction of ADV traded
    borrow_bps_annual: float = 50.0   # annual borrow cost on short notional


class TransactionCostModel:
    """Computes per-rebalance cost as a fraction of gross book value."""

    def __init__(self, params: CostParams | None = None) -> None:
        self.p = params or CostParams()

    def turnover(self, w_prev: pd.Series, w_new: pd.Series) -> float:
        """One-way turnover = 0.5 * sum |w_new - w_prev| over the union."""
        idx = w_prev.index.union(w_new.index)
        a = w_prev.reindex(idx).fillna(0.0)
        b = w_new.reindex(idx).fillna(0.0)
        return 0.5 * (b - a).abs().sum()

    def cost(
        self, w_prev: pd.Series, w_new: pd.Series, periods_per_year: int = 252,
        holding_periods: int = 21,
    ) -> float:
        """Return total cost as a fraction of book value for this rebalance.

        Spread + impact are charged on traded notional; borrow is charged on
        short notional, pro-rated for the holding period.
        """
        idx = w_prev.index.union(w_new.index)
        a = w_prev.reindex(idx).fillna(0.0)
        b = w_new.reindex(idx).fillna(0.0)
        traded = (b - a).abs()  # per-name traded notional fraction

        spread_cost = (self.p.half_spread_bps / 1e4) * traded.sum()

        impact_bps = self.p.impact_coef_bps * np.sqrt(self.p.adv_participation)
        impact_cost = (impact_bps / 1e4) * traded.sum()

        short_notional = b.clip(upper=0).abs().sum()
        borrow_cost = (
            (self.p.borrow_bps_annual / 1e4)
            * short_notional
            * (holding_periods / periods_per_year)
        )

        return spread_cost + impact_cost + borrow_cost
