"""
Extension 3: Volatility Arbitrage
====================================
Systematic strategy exploiting the implied-realised vol spread.

The Volatility Risk Premium (VRP):
  VRP = IV - RV (typically positive, ~2-4 vol points)

Strategy:
  - Short vol when VRP is wide (sell overpriced options)
  - Long vol when VRP is compressed or negative (buy cheap protection)
  - Delta-hedged: P&L comes from gamma × (RV² - IV²)

Implementation:
  - Synthetic IV from GARCH forecast as proxy
  - Realised vol from rolling returns
  - Z-score of the VRP spread for entry/exit
  - Comparison with constant short vol (benchmark)

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from stat_arb_engine import PerformanceAnalytics


@dataclass
class VolArbConfig:
    """Volatility arbitrage parameters."""
    rv_window: int = 21          # Realised vol window (1 month)
    iv_window: int = 21          # Implied vol proxy window
    z_window: int = 63           # Z-score lookback (3 months)
    entry_z: float = 1.0         # VRP z-score entry
    exit_z: float = 0.3          # VRP z-score exit
    gamma_scale: float = 100.0   # Notional scaling


class GARCHProxy:
    """Simple GARCH(1,1) for implied vol proxy."""

    @staticmethod
    def fit(returns: np.ndarray, omega=1e-6, alpha=0.10, beta=0.85):
        """Fit GARCH(1,1) and return conditional variance series."""
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

        return np.sqrt(sigma2) * np.sqrt(252)  # Annualised vol


class VolatilityArbitrage:
    """Volatility arbitrage strategy engine."""

    def __init__(self, config: VolArbConfig = None):
        self.config = config or VolArbConfig()

    def compute_vol_surfaces(self, returns: np.ndarray) -> dict:
        """Compute realised vol, implied vol proxy, and VRP."""
        c = self.config
        n = len(returns)

        # Realised vol (rolling)
        rv = np.zeros(n)
        for i in range(c.rv_window, n):
            rv[i] = np.std(returns[i - c.rv_window:i]) * np.sqrt(252)

        # Implied vol proxy via GARCH
        iv = GARCHProxy.fit(returns)

        # Add typical risk premium (~2 vol points)
        iv = iv + 0.02

        # VRP
        vrp = iv - rv

        # Z-score of VRP
        vrp_z = np.zeros(n)
        for i in range(c.z_window, n):
            w = vrp[i - c.z_window:i]
            mu, sigma = np.mean(w), np.std(w)
            vrp_z[i] = (vrp[i] - mu) / sigma if sigma > 0 else 0

        return {
            'realised_vol': rv,
            'implied_vol': iv,
            'vrp': vrp,
            'vrp_zscore': vrp_z,
        }

    def backtest(self, returns: np.ndarray) -> pd.DataFrame:
        """Run vol arb backtest."""
        c = self.config
        vols = self.compute_vol_surfaces(returns)
        n = len(returns)

        rv = vols['realised_vol']
        iv = vols['implied_vol']
        vrp = vols['vrp']
        vrp_z = vols['vrp_zscore']

        positions = np.zeros(n)  # +1 = short vol, -1 = long vol
        pnl = np.zeros(n)

        start = max(c.rv_window, c.z_window) + 1

        for i in range(start, n):
            prev = positions[i - 1]

            # Entry/exit based on VRP z-score
            if prev == 0:
                if vrp_z[i] > c.entry_z:
                    positions[i] = 1    # Short vol (VRP wide, sell options)
                elif vrp_z[i] < -c.entry_z:
                    positions[i] = -1   # Long vol (VRP compressed, buy protection)
            else:
                if abs(vrp_z[i]) < c.exit_z:
                    positions[i] = 0
                else:
                    positions[i] = prev

            # P&L: gamma-like payoff
            # Short vol profits when RV < IV, loses when RV > IV
            # Daily P&L ≈ (IV² - RV²) / 2 × position × scale
            if positions[i - 1] != 0 and iv[i] > 0:
                daily_gamma_pnl = (iv[i - 1] ** 2 - returns[i] ** 2 * 252) / 2
                pnl[i] = positions[i - 1] * daily_gamma_pnl * c.gamma_scale

        # Benchmark: constant short vol
        pnl_bench = np.zeros(n)
        for i in range(start, n):
            if iv[i] > 0:
                daily_gamma = (iv[i - 1] ** 2 - returns[i] ** 2 * 252) / 2
                pnl_bench[i] = daily_gamma * c.gamma_scale

        results = pd.DataFrame({
            'returns': returns,
            'realised_vol': rv,
            'implied_vol': iv,
            'vrp': vrp,
            'vrp_zscore': vrp_z,
            'position': positions,
            'pnl': pnl,
            'pnl_benchmark': pnl_bench,
        })
        results['cum_pnl'] = results['pnl'].cumsum()
        results['cum_bench'] = results['pnl_benchmark'].cumsum()

        return results


def generate_returns_with_vol_clustering(n=1500, seed=42):
    """Generate returns with realistic vol clustering."""
    rng = np.random.default_rng(seed)

    # GARCH-like process
    omega = 1e-5
    alpha = 0.10
    beta = 0.85
    sigma2 = np.zeros(n)
    sigma2[0] = 0.02 ** 2
    returns = np.zeros(n)

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))

    return returns


if __name__ == "__main__":
    print("Volatility Arbitrage: IV vs RV Spread Strategy")
    print("=" * 55)

    returns = generate_returns_with_vol_clustering(1500)

    engine = VolatilityArbitrage()
    results = engine.backtest(returns)

    strat_stats = PerformanceAnalytics.compute(results['pnl'].values, 'Vol Arb')
    bench_stats = PerformanceAnalytics.compute(results['pnl_benchmark'].values, 'Constant Short Vol')

    print(f"\n{'Metric':<20} {'Vol Arb':>12} {'Short Vol':>12}")
    print("-" * 46)
    for key in ['total_return', 'sharpe', 'max_drawdown', 'win_rate', 'profit_factor']:
        sv = strat_stats[key]
        bv = bench_stats[key]
        fmt = '.2f' if key != 'win_rate' else '.1%'
        print(f"{key:<20} {sv:>12{fmt}} {bv:>12{fmt}}")

    # VRP statistics
    vrp = results['vrp'].values[63:]
    print(f"\nVOLATILITY RISK PREMIUM")
    print(f"  Mean VRP:         {np.mean(vrp):.2%}")
    print(f"  Median VRP:       {np.median(vrp):.2%}")
    print(f"  VRP > 0:          {np.mean(vrp > 0):.0%} of time")
    print(f"  Mean RV:          {np.mean(results['realised_vol'].values[63:]):.2%}")
    print(f"  Mean IV:          {np.mean(results['implied_vol'].values[63:]):.2%}")

    # Position analysis
    pos = results['position'].values
    print(f"\nPOSITION ANALYSIS")
    print(f"  Short vol:  {np.mean(pos == 1):.0%} of time")
    print(f"  Long vol:   {np.mean(pos == -1):.0%} of time")
    print(f"  Flat:       {np.mean(pos == 0):.0%} of time")
