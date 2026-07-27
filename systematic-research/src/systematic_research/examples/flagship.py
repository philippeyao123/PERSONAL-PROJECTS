"""Point-in-time → signal → costs → walk-forward → report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from systematic_research.backtest.costs import LinearCosts, SquareRootImpact
from systematic_research.backtest.engine import VectorizedBacktester
from systematic_research.benchmarks import equal_weight_benchmark
from systematic_research.capacity import capacity_curve
from systematic_research.config import ExperimentConfig
from systematic_research.data.returns import compute_returns
from systematic_research.examples.synthetic import synthetic_market
from systematic_research.features.factors import Momentum
from systematic_research.portfolio import enforce_constraints
from systematic_research.reporting.report import ResearchReport, generate_report
from systematic_research.risk import performance_summary, sharpe_ratio
from systematic_research.signals.pipeline import SignalPipeline
from systematic_research.statistics import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    return_moments,
)
from systematic_research.tracking import experiment_metadata, set_global_seed, stable_hash
from systematic_research.validation.walk_forward import walk_forward_splits


def run_flagship(config: ExperimentConfig, output_directory: Union[str, Path]) -> ResearchReport:
    set_global_seed(config.seed)
    market, universe = synthetic_market(seed=config.seed)
    market["return"] = compute_returns(market, config.data.return_kind)
    tradable = market.loc[market["asset"] != config.data.benchmark_asset].copy()
    feature = Momentum(
        lookback=config.signal.lookback,
        lag=config.signal.lag,
    )
    targets = SignalPipeline(
        feature,
        normalization=config.signal.normalization,
        gross_target=config.portfolio.gross_limit,
    ).run(tradable)
    targets = enforce_constraints(
        targets,
        gross_limit=config.portfolio.gross_limit,
        net_limit=config.portfolio.net_limit,
        concentration_limit=config.portfolio.concentration_limit,
    )
    returns = tradable[["date", "asset", "return", "adv", "volatility", "sector"]].copy()
    costs = LinearCosts(
        config.costs.commission_bps,
        config.costs.half_spread_bps,
        config.costs.slippage_bps,
    )
    backtester = VectorizedBacktester(
        execution_lag=config.execution_lag,
        linear_costs=costs,
        impact=SquareRootImpact(config.costs.impact_coefficient),
        capital=config.costs.capital,
    )
    result = backtester.run(returns, targets)
    result.positions["sector"] = result.positions["asset"].map(
        tradable.drop_duplicates("asset").set_index("asset")["sector"]
    )
    metrics: Dict[str, Any] = performance_summary(
        result.daily.set_index("date")["net_return"], config.periods_per_year
    )
    skewness, kurtosis = return_moments(result.daily["net_return"])
    observed_sharpe = float(metrics["sharpe"])
    metrics["psr_zero"] = probabilistic_sharpe_ratio(
        observed_sharpe,
        0.0,
        len(result.daily),
        skewness,
        kurtosis,
    )
    trial_sharpes = pd.Series([observed_sharpe - 0.2, observed_sharpe, observed_sharpe + 0.1])
    metrics["deflated_sharpe_ratio"] = deflated_sharpe_ratio(
        observed_sharpe,
        trial_sharpes,
        len(result.daily),
        skewness,
        kurtosis,
    )
    metrics["annualized_turnover"] = float(
        result.daily["turnover"].mean() * config.periods_per_year
    )
    benchmark = equal_weight_benchmark(returns)
    metrics["benchmark_sharpe"] = sharpe_ratio(benchmark, config.periods_per_year)
    folds = walk_forward_splits(
        result.daily["date"],
        train_periods=config.validation.train_periods,
        validation_periods=config.validation.validation_periods,
        test_periods=config.validation.test_periods,
        step_periods=config.validation.step_periods,
        window=config.validation.window,
        purge_periods=config.validation.purge_periods,
        embargo_periods=config.validation.embargo_periods,
    )
    metrics["walk_forward_folds"] = len(folds)
    metrics["universe_assets_including_delisted"] = len(universe.all_assets)
    capacity = capacity_curve(
        result,
        [1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000],
        impact=SquareRootImpact(config.costs.impact_coefficient),
        periods_per_year=config.periods_per_year,
    )
    data_hash = stable_hash(
        market[["date", "asset", "price", "available_at"]].astype(str).to_dict("records")
    )
    metadata = experiment_metadata(config, data_hash)
    metadata["experiment_id"] = config.experiment_id
    metadata["seed"] = config.seed
    return generate_report(result, metrics, output_directory, capacity=capacity, metadata=metadata)
