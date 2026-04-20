"""
Extension 1: Regime-Aware Mean Reversion
==========================================
Adjusts trading thresholds dynamically based on market regime.

Regimes detected via:
  - Rolling volatility (high/low vol states)
  - Spread half-life (fast/slow mean reversion)
  - Hurst exponent (trending vs mean-reverting)

In high-vol regimes: wider entry thresholds, tighter stops.
In low-vol regimes:  tighter entry, wider stops (stronger mean reversion).

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict
from stat_arb_engine import DataGenerator, PerformanceAnalytics


@dataclass
class RegimeConfig:
    """Regime-dependent trading parameters."""
    entry_z: float
    exit_z: float
    stop_loss_z: float
    max_holding: int
    label: str


LOW_VOL = RegimeConfig(entry_z=1.2, exit_z=0.3, stop_loss_z=3.5, max_holding=80, label='LOW_VOL')
HIGH_VOL = RegimeConfig(entry_z=2.0, exit_z=0.8, stop_loss_z=3.0, max_holding=30, label='HIGH_VOL')
TRENDING = RegimeConfig(entry_z=2.5, exit_z=1.0, stop_loss_z=2.5, max_holding=15, label='TRENDING')


class RegimeDetector:
    """Detect market regime from spread characteristics."""

    @staticmethod
    def rolling_vol_regime(spreads: np.ndarray, window: int = 60, threshold: float = None):
        """Classify into HIGH_VOL / LOW_VOL based on rolling volatility."""
        n = len(spreads)
        regimes = np.full(n, 'LOW_VOL', dtype='U10')
        vols = np.zeros(n)

        for i in range(window, n):
            vols[i] = np.std(spreads[i - window:i])

        if threshold is None:
            threshold = np.median(vols[window:])

        for i in range(window, n):
            regimes[i] = 'HIGH_VOL' if vols[i] > threshold else 'LOW_VOL'

        return regimes, vols

    @staticmethod
    def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
        """
        Estimate Hurst exponent.
        H < 0.5 → mean-reverting, H = 0.5 → random walk, H > 0.5 → trending.
        """
        lags = range(2, min(max_lag, len(series) // 4))
        tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]

        if len(tau) < 2 or any(t <= 0 for t in tau):
            return 0.5

        log_lags = np.log(list(lags))
        log_tau = np.log(tau)
        poly = np.polyfit(log_lags, log_tau, 1)
        return poly[0]

    @staticmethod
    def rolling_hurst(spreads: np.ndarray, window: int = 100) -> np.ndarray:
        """Rolling Hurst exponent."""
        n = len(spreads)
        hursts = np.full(n, 0.5)
        for i in range(window, n):
            hursts[i] = RegimeDetector.hurst_exponent(spreads[i - window:i])
        return hursts

    @staticmethod
    def half_life(spreads: np.ndarray, window: int = 100) -> np.ndarray:
        """Rolling half-life of mean reversion."""
        n = len(spreads)
        hl = np.full(n, np.nan)
        for i in range(window, n):
            s = spreads[i - window:i]
            delta = np.diff(s)
            lagged = s[:-1]
            if np.var(lagged) > 0:
                X = np.column_stack([np.ones(len(lagged)), lagged])
                beta = np.linalg.lstsq(X, delta, rcond=None)[0]
                if beta[1] < 0:
                    hl[i] = -np.log(2) / beta[1]
                else:
                    hl[i] = np.inf
        return hl


class RegimeAwareBacktester:
    """Backtester that adapts parameters to the current regime."""

    def __init__(self, window: int = 100, tc_bps: float = 5.0):
        self.window = window
        self.tc_bps = tc_bps

    def backtest(self, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        n = len(x)
        positions = np.zeros(n)
        spreads = np.zeros(n)
        zscores = np.zeros(n)
        pnl = np.zeros(n)
        regime_labels = np.full(n, 'LOW_VOL', dtype='U10')
        holding = np.zeros(n)

        for i in range(self.window, n):
            # Rolling OLS
            X_w = x[i - self.window:i]
            Y_w = y[i - self.window:i]
            X_mat = np.column_stack([np.ones(self.window), X_w])
            beta = np.linalg.lstsq(X_mat, Y_w, rcond=None)[0]
            hedge = beta[1]

            spread = y[i] - hedge * x[i]
            spreads[i] = spread
            sw = y[i - self.window:i] - hedge * x[i - self.window:i]
            mu, sigma = np.mean(sw), np.std(sw)
            z = (spread - mu) / sigma if sigma > 0 else 0
            zscores[i] = z

            # Detect regime
            vol = np.std(sw)
            vol_threshold = np.median(np.abs(np.diff(sw)))

            if vol > vol_threshold * 2:
                regime = HIGH_VOL
            else:
                # Check Hurst
                h = RegimeDetector.hurst_exponent(sw)
                regime = TRENDING if h > 0.55 else LOW_VOL

            regime_labels[i] = regime.label
            cfg = regime

            # Trading logic with regime-dependent thresholds
            prev = positions[i - 1]
            prev_hold = holding[i - 1]

            if prev == 0:
                if z > cfg.entry_z:
                    positions[i] = -1
                    holding[i] = 1
                elif z < -cfg.entry_z:
                    positions[i] = 1
                    holding[i] = 1
            else:
                if (abs(z) < cfg.exit_z or abs(z) > cfg.stop_loss_z or prev_hold >= cfg.max_holding):
                    positions[i] = 0
                    holding[i] = 0
                else:
                    positions[i] = prev
                    holding[i] = prev_hold + 1

            if i > self.window:
                pnl[i] = (spreads[i] - spreads[i - 1]) * positions[i - 1]
                if positions[i] != positions[i - 1]:
                    notional = abs(y[i]) + abs(hedge * x[i])
                    pnl[i] -= notional * self.tc_bps / 10000

        results = pd.DataFrame({
            'spread': spreads, 'zscore': zscores, 'position': positions,
            'pnl': pnl, 'regime': regime_labels, 'holding': holding
        })
        results['cum_pnl'] = results['pnl'].cumsum()
        return results


if __name__ == "__main__":
    from stat_arb_engine import MeanReversionBacktester, BacktestConfig

    gen = DataGenerator()
    df = gen.cointegrated_pair(1500, beta=0.8, noise_std=1.0)

    print("Regime-Aware Mean Reversion")
    print("=" * 50)

    # Static backtest
    static_bt = MeanReversionBacktester(BacktestConfig(window=100, entry_z=1.5, exit_z=0.5))
    static_res = static_bt.backtest(df['x'].values, df['y'].values)
    static_stats = PerformanceAnalytics.compute(static_res['pnl'].values, 'Static')

    # Regime-aware backtest
    regime_bt = RegimeAwareBacktester(window=100)
    regime_res = regime_bt.backtest(df['x'].values, df['y'].values)
    regime_stats = PerformanceAnalytics.compute(regime_res['pnl'].values, 'Regime-Aware')

    print(f"\n{'Metric':<20} {'Static':>10} {'Regime':>10}")
    print("-" * 42)
    for key in ['total_return', 'sharpe', 'max_drawdown', 'win_rate', 'profit_factor']:
        sv = static_stats[key]
        rv = regime_stats[key]
        fmt = '.2f' if key != 'win_rate' else '.1%'
        print(f"{key:<20} {sv:>10{fmt}} {rv:>10{fmt}}")

    # Regime distribution
    regimes = regime_res['regime'].values
    for r in ['LOW_VOL', 'HIGH_VOL', 'TRENDING']:
        pct = np.mean(regimes[100:] == r) * 100
        print(f"\n{r}: {pct:.0f}% of time")
        mask = regimes == r
        if np.sum(mask) > 10:
            r_pnl = regime_res.loc[mask, 'pnl'].values
            r_stats = PerformanceAnalytics.compute(r_pnl, r)
            print(f"  Sharpe: {r_stats['sharpe']:.2f}, WR: {r_stats['win_rate']:.1%}")

    # Hurst analysis
    hursts = RegimeDetector.rolling_hurst(static_res['spread'].values)
    valid_h = hursts[hursts != 0.5]
    if len(valid_h) > 0:
        print(f"\nHurst exponent: mean={np.mean(valid_h):.3f}, median={np.median(valid_h):.3f}")
        print(f"  < 0.5 (mean-reverting): {np.mean(valid_h < 0.5):.0%}")
        print(f"  > 0.5 (trending):       {np.mean(valid_h > 0.5):.0%}")
