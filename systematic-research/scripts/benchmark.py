"""Small deterministic end-to-end benchmark."""

from __future__ import annotations

from time import perf_counter

from systematic_research.backtest.engine import VectorizedBacktester
from systematic_research.data.returns import compute_returns
from systematic_research.examples.synthetic import synthetic_market
from systematic_research.features.factors import Momentum
from systematic_research.portfolio import enforce_constraints
from systematic_research.signals.pipeline import SignalPipeline


def main() -> None:
    market, _ = synthetic_market(seed=42)
    market["return"] = compute_returns(market)
    tradable = market.loc[market["asset"] != "MARKET"].copy()

    started = perf_counter()
    targets = SignalPipeline(Momentum(lookback=63, lag=1)).run(tradable)
    targets = enforce_constraints(targets, gross_limit=1.0, net_limit=0.1, concentration_limit=0.15)
    returns = tradable[["date", "asset", "return", "adv", "volatility", "sector"]].copy()
    result = VectorizedBacktester(execution_lag=1).run(returns, targets)
    elapsed = perf_counter() - started
    print(
        f"rows={len(tradable)} days={len(result.daily)} "
        f"seconds={elapsed:.4f} rows_per_second={len(tradable) / elapsed:.0f}"
    )


if __name__ == "__main__":
    main()
