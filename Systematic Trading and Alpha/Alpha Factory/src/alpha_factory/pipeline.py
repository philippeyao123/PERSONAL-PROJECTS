"""End-to-end pipeline: data -> factors -> backtest -> diagnostics -> report.

Run as a script::

    python -m alpha_factory.pipeline

or import `run_pipeline` for programmatic use / notebooks.
"""
from __future__ import annotations

import logging

from alpha_factory.backtest.costs import CostParams, TransactionCostModel
from alpha_factory.backtest.engine import BacktestResult, WalkForwardBacktest
from alpha_factory.data.loader import PanelData, make_synthetic_panel
from alpha_factory.diagnostics.metrics import (
    capacity_analysis,
    deflated_sharpe_ratio,
    information_coefficient,
    performance_stats,
    probabilistic_sharpe_ratio,
)
from alpha_factory.factors.combiner import FactorCombiner
from alpha_factory.factors.library import (
    Amihud,
    FundamentalValue,
    LowVolatility,
    Momentum,
    ShortTermReversal,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("alpha_factory.pipeline")


def default_factors() -> list:
    return [
        Momentum(lookback=252, skip=21),
        ShortTermReversal(lookback=21),
        LowVolatility(lookback=126),
        FundamentalValue(field="earnings_yield"),
        FundamentalValue(field="book_to_price"),
        Amihud(lookback=63),
    ]


def run_pipeline(
    panel: PanelData | None = None,
    n_trials: int = 50,
    combine_method: str = "ic",
) -> dict:
    """Run the full pipeline and return a results dict.

    `n_trials` feeds the deflated Sharpe: be honest about how many factor /
    parameter configurations you actually explored. Inflating this number
    deflates your own Sharpe — which is the point: it keeps you honest.
    """
    if panel is None:
        logger.info("No panel supplied; generating synthetic data.")
        panel = make_synthetic_panel()

    factors = default_factors()
    combiner = FactorCombiner(method=combine_method)
    cost_model = TransactionCostModel(CostParams())

    bt = WalkForwardBacktest(
        factors=factors, combiner=combiner, cost_model=cost_model,
        rebalance_freq=21, quantile=0.1, max_weight=0.05, warmup=252,
    )
    result: BacktestResult = bt.run(panel)

    ppy = 252 / bt.rebalance_freq  # rebalances per year

    gross = performance_stats(result.gross_returns, result.turnover, ppy)
    net = performance_stats(result.net_returns, result.turnover, ppy)
    psr = probabilistic_sharpe_ratio(result.net_returns, 0.0, ppy)
    dsr = deflated_sharpe_ratio(result.net_returns, n_trials=n_trials,
                                periods_per_year=ppy)
    ic_summary = information_coefficient(result.factor_ic)
    capacity = capacity_analysis(
        gross.sharpe, net.mean_turnover, net.ann_return,
        periods_per_year=ppy,
    )

    return {
        "result": result,
        "gross_stats": gross,
        "net_stats": net,
        "psr": psr,
        "dsr": dsr,
        "ic_summary": ic_summary,
        "capacity": capacity,
        "n_trials": n_trials,
    }


def print_report(out: dict) -> None:
    net, gross, dsr = out["net_stats"], out["gross_stats"], out["dsr"]
    print("\n" + "=" * 64)
    print("MULTI-ASSET ALPHA FACTORY — BACKTEST REPORT")
    print("=" * 64)
    print(f"\n{'Metric':<22}{'Gross':>12}{'Net':>12}")
    print("-" * 46)
    for label, g, n in [
        ("Ann. return", gross.ann_return, net.ann_return),
        ("Ann. vol", gross.ann_vol, net.ann_vol),
        ("Sharpe", gross.sharpe, net.sharpe),
        ("Sortino", gross.sortino, net.sortino),
        ("Max drawdown", gross.max_drawdown, net.max_drawdown),
        ("Calmar", gross.calmar, net.calmar),
        ("Skew", gross.skew, net.skew),
        ("Excess kurtosis", gross.kurtosis, net.kurtosis),
        ("Hit rate", gross.hit_rate, net.hit_rate),
    ]:
        print(f"{label:<22}{g:>12.3f}{n:>12.3f}")
    print(f"{'Mean turnover':<22}{net.mean_turnover:>12.3f}"
          f"{'(per rebal)':>12}")

    print("\n--- STATISTICAL RIGOR " + "-" * 42)
    print(f"Probabilistic Sharpe Ratio (net, vs 0): {out['psr']:.3f}")
    print(f"Trials assumed for deflation           : {out['n_trials']}")
    print(f"Expected max Sharpe under null         : {dsr['expected_max_sr']:.3f}")
    print(f"Deflated Sharpe Ratio (net)            : {dsr['dsr']:.3f}")
    verdict = ("SURVIVES deflation (DSR > 0.95)" if dsr["dsr"] and dsr["dsr"] > 0.95
               else "does NOT clear DSR>0.95 — treat as inconclusive")
    print(f"Verdict                                : {verdict}")

    print("\n--- FACTOR IC SUMMARY " + "-" * 42)
    if not out["ic_summary"].empty:
        print(out["ic_summary"].round(4).to_string())

    print("\n--- CAPACITY (net return crosses zero) " + "-" * 25)
    cap = out["capacity"]
    breach = cap[cap["net_return"] <= 0]
    if not breach.empty:
        aum = breach.iloc[0]["aum"]
        print(f"Estimated capacity: ~${aum/1e9:.1f}B AUM")
    else:
        print("No zero-crossing within tested AUM grid (<= $100B)")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    out = run_pipeline()
    print_report(out)
