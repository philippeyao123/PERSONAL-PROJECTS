"""Walk-forward backtest engine.

Strict temporal hygiene:
    - At each rebalance date `t`, factors only see data up to and including `t`.
    - Weights set at `t` earn the *forward* return from `t` to the next
      rebalance. No same-bar look-ahead.
    - Costs are charged at `t` based on the change from the previous book.

The engine returns a BacktestResult carrying the net/gross return streams,
weight history, turnover, and the rolling factor IC used for IC-weighting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_factory.backtest.costs import TransactionCostModel
from alpha_factory.data.loader import PanelData
from alpha_factory.factors.combiner import FactorCombiner
from alpha_factory.factors.library import Factor
from alpha_factory.portfolio.construction import (
    apply_position_limits,
    quantile_long_short,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    weights: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    factor_ic: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def equity_curve(self) -> pd.Series:
        return (1 + self.net_returns).cumprod()


class WalkForwardBacktest:
    """Run a periodic-rebalance walk-forward backtest of a factor portfolio."""

    def __init__(
        self,
        factors: list[Factor],
        combiner: FactorCombiner | None = None,
        cost_model: TransactionCostModel | None = None,
        rebalance_freq: int = 21,
        ic_window: int = 6,
        quantile: float = 0.1,
        max_weight: float = 0.05,
        warmup: int = 252,
    ) -> None:
        self.factors = factors
        self.combiner = combiner or FactorCombiner(method="equal")
        self.cost_model = cost_model or TransactionCostModel()
        self.rebalance_freq = rebalance_freq
        self.ic_window = ic_window
        self.quantile = quantile
        self.max_weight = max_weight
        self.warmup = warmup

    def run(self, panel: PanelData) -> BacktestResult:
        dates = panel.dates
        rebal_idx = list(range(self.warmup, len(dates) - self.rebalance_freq,
                               self.rebalance_freq))
        if not rebal_idx:
            raise ValueError("not enough data for a single rebalance")

        sectors = None
        if "sector" in panel.metadata:
            sectors = panel.metadata["sector"]

        gross_r, net_r, turn = {}, {}, {}
        weights_hist: dict[pd.Timestamp, pd.Series] = {}
        ic_records: list[dict] = []
        w_prev = pd.Series(dtype=float)
        recent_ic: dict[str, list[float]] = {f.name: [] for f in self.factors}

        for i in rebal_idx:
            t = dates[i]
            t_next = dates[i + self.rebalance_freq]

            # Forward return realized over the holding period (per asset).
            fwd = panel.prices.loc[t_next] / panel.prices.loc[t] - 1.0

            # 1. Compute each factor's cross-sectional scores as of t.
            factor_scores = {}
            for f in self.factors:
                s = f.compute(panel, t)
                if not s.empty:
                    factor_scores[f.name] = s
                    # Record this factor's IC vs forward return (for weighting
                    # and diagnostics). Uses only info available after t_next,
                    # so it lags one period in the live weighting below.
                    common = s.index.intersection(fwd.dropna().index)
                    if len(common) > 10:
                        ic = s[common].corr(fwd[common], method="spearman")
                        recent_ic[f.name].append(ic)
                        ic_records.append({"date": t, "factor": f.name, "ic": ic})

            if not factor_scores:
                continue

            # 2. IC weights from the trailing window (excludes current period).
            ic_weights = {
                name: float(np.nanmean(vals[-self.ic_window - 1:-1]))
                if len(vals) > 1 else 0.0
                for name, vals in recent_ic.items()
            }

            # 3. Combine into one signal.
            signal = self.combiner.combine(
                factor_scores, sectors=sectors, ic_weights=ic_weights
            )
            if signal.empty:
                continue

            # 4. Construct the book.
            w_new = quantile_long_short(signal, quantile=self.quantile)
            w_new = apply_position_limits(w_new, self.max_weight)

            # 5. Cost & turnover for moving from w_prev to w_new.
            to = self.cost_model.turnover(w_prev, w_new)
            cost = self.cost_model.cost(
                w_prev, w_new, holding_periods=self.rebalance_freq
            )

            # 6. Portfolio return over the holding period.
            common = w_new.index.intersection(fwd.dropna().index)
            port_ret = float((w_new[common] * fwd[common]).sum())

            gross_r[t_next] = port_ret
            net_r[t_next] = port_ret - cost
            turn[t_next] = to
            weights_hist[t] = w_new
            w_prev = w_new

        ic_df = (pd.DataFrame(ic_records).pivot(index="date", columns="factor",
                                                values="ic")
                 if ic_records else pd.DataFrame())

        result = BacktestResult(
            gross_returns=pd.Series(gross_r).sort_index(),
            net_returns=pd.Series(net_r).sort_index(),
            turnover=pd.Series(turn).sort_index(),
            weights=weights_hist,
            factor_ic=ic_df,
        )
        logger.info("Backtest done: %d rebalances, mean turnover %.1f%%",
                    len(net_r), 100 * result.turnover.mean())
        return result
