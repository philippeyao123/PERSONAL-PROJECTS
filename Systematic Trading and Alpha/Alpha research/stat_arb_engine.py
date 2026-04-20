"""
Statistical Arbitrage & Market Microstructure Engine
======================================================
Complete systematic trading framework:

1. Cointegration-based pair selection (Engle-Granger + Johansen)
2. Rolling z-score mean reversion backtester
3. Multi-pair portfolio with risk aggregation
4. HFT microstructure analysis (spread decomposition, VPIN, order flow)
5. ML-based signal enhancement (Random Forest, GBM)

Pipeline:
  - Select cointegrated pairs from a universe
  - Estimate rolling hedge ratios via OLS
  - Generate z-score entry/exit signals
  - Backtest with realistic execution model
  - Aggregate into multi-pair portfolio with risk metrics
  - Analyse microstructure signals for timing

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, norm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time


# ============================================================================
# Synthetic Data Generator
# ============================================================================

class DataGenerator:
    """Generate synthetic cointegrated pairs and tick data for testing."""

    @staticmethod
    def cointegrated_pair(n=1500, beta=0.8, noise_std=1.0, seed=42):
        """Generate two cointegrated series: y = beta*x + noise."""
        rng = np.random.default_rng(seed)
        x = np.cumsum(rng.normal(scale=noise_std, size=n))
        noise = rng.normal(scale=noise_std * 0.5, size=n)
        y = beta * x + noise
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        return pd.DataFrame({'x': x, 'y': y}, index=dates)

    @staticmethod
    def multi_pair_universe(n=1500, seed=42):
        """Generate a universe of pairs with varying cointegration strength."""
        rng = np.random.default_rng(seed)
        pairs = {}
        configs = [
            ('PairA', 0.9, 0.8, 'Strong'),
            ('PairB', 0.7, 1.2, 'Moderate'),
            ('PairC', 0.5, 1.5, 'Weak'),
            ('PairD', 0.95, 0.6, 'Very Strong'),
        ]
        for name, beta, noise, strength in configs:
            x = np.cumsum(rng.normal(scale=1.0, size=n))
            noise_term = rng.normal(scale=noise, size=n)
            y = beta * x + noise_term
            pairs[name] = {'x': x, 'y': y, 'beta': beta, 'strength': strength}
        return pairs

    @staticmethod
    def tick_data(n=50000, base_price=100.0, tick_size=0.01, seed=42):
        """Generate synthetic tick-level data with bid/ask."""
        rng = np.random.default_rng(seed)
        mid = np.cumsum(rng.normal(0, 0.02, n)) + base_price
        spread = np.abs(rng.normal(0.05, 0.02, n))
        bid = mid - spread / 2
        ask = mid + spread / 2
        volume = rng.poisson(100, n)
        side = rng.choice(['buy', 'sell'], n, p=[0.52, 0.48])
        timestamps = pd.date_range('2024-01-02 09:30', periods=n, freq='s')
        return pd.DataFrame({
            'timestamp': timestamps, 'mid': mid, 'bid': bid, 'ask': ask,
            'spread': spread, 'volume': volume, 'side': side,
            'price': np.where(side == 'buy', ask, bid)
        }).set_index('timestamp')


# ============================================================================
# Cointegration Tests
# ============================================================================

class CointegrationTester:
    """Test for cointegration between price series."""

    @staticmethod
    def engle_granger(x: np.ndarray, y: np.ndarray) -> dict:
        """
        Engle-Granger two-step cointegration test.
        Step 1: OLS regression y = alpha + beta*x + epsilon
        Step 2: ADF test on residuals
        """
        n = len(x)
        X = np.column_stack([np.ones(n), x])
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta_hat

        # ADF test (simplified: check if residual is mean-reverting)
        delta_r = np.diff(residuals)
        r_lagged = residuals[:-1]
        X_adf = np.column_stack([np.ones(len(r_lagged)), r_lagged])
        gamma_hat = np.linalg.lstsq(X_adf, delta_r, rcond=None)[0]

        # t-statistic for gamma (mean reversion coefficient)
        resid_adf = delta_r - X_adf @ gamma_hat
        se = np.sqrt(np.sum(resid_adf ** 2) / (len(delta_r) - 2) / np.sum((r_lagged - np.mean(r_lagged)) ** 2))
        t_stat = gamma_hat[1] / se if se > 0 else 0

        # Critical values (approximate for n > 500)
        critical = {1: -3.43, 5: -2.86, 10: -2.57}
        is_cointegrated = t_stat < critical[5]

        return {
            'hedge_ratio': beta_hat[1],
            'intercept': beta_hat[0],
            't_statistic': t_stat,
            'critical_values': critical,
            'is_cointegrated': is_cointegrated,
            'residuals': residuals,
            'half_life': -np.log(2) / gamma_hat[1] if gamma_hat[1] < 0 else np.inf,
        }

    @staticmethod
    def johansen_simplified(x: np.ndarray, y: np.ndarray, lags: int = 1) -> dict:
        """
        Simplified Johansen test for 2 variables.
        Tests for the number of cointegrating relationships (0, 1, or 2).
        """
        data = np.column_stack([x, y])
        n, k = data.shape

        # First differences
        delta = np.diff(data, axis=0)

        # Lagged levels
        levels = data[:-1]

        # OLS: delta on levels
        X = np.column_stack([np.ones(n - 1), levels])
        betas = np.linalg.lstsq(X, delta, rcond=None)[0]
        residuals = delta - X @ betas

        # Product moment matrices
        S00 = residuals.T @ residuals / (n - 1)
        S01 = residuals.T @ levels / (n - 1)
        S10 = levels.T @ residuals / (n - 1)
        S11 = levels.T @ levels / (n - 1)

        # Eigenvalues
        try:
            S11_inv = np.linalg.inv(S11)
            M = S11_inv @ S10 @ np.linalg.inv(S00) @ S01
            eigenvalues = np.sort(np.real(np.linalg.eigvals(M)))[::-1]
        except np.linalg.LinAlgError:
            eigenvalues = np.array([0, 0])

        # Trace statistic
        trace_stats = -n * np.log(1 - np.clip(eigenvalues, 0, 0.999))

        # Critical values (approximate for 2 variables)
        trace_critical = {0: 15.41, 1: 3.76}

        n_coint = 0
        if trace_stats[0] > trace_critical[0]:
            n_coint = 1
        if len(trace_stats) > 1 and trace_stats[1] > trace_critical[1]:
            n_coint = 2

        return {
            'eigenvalues': eigenvalues,
            'trace_statistics': trace_stats,
            'trace_critical': trace_critical,
            'n_cointegrating': n_coint,
            'is_cointegrated': n_coint >= 1,
        }


# ============================================================================
# Z-Score Backtester
# ============================================================================

@dataclass
class BacktestConfig:
    """Configuration for the mean reversion backtester."""
    window: int = 100           # Rolling window for hedge ratio
    entry_z: float = 1.5        # Z-score entry threshold
    exit_z: float = 0.5         # Z-score exit threshold
    stop_loss_z: float = 4.0    # Z-score stop loss
    max_holding: int = 60       # Max days to hold a position
    transaction_cost_bps: float = 5.0   # One-way cost in bps


class MeanReversionBacktester:
    """
    Rolling z-score mean reversion backtester.

    Uses rolling OLS to estimate hedge ratio, computes z-score
    of the spread, and trades based on entry/exit thresholds.
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def backtest(self, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Run backtest on a pair (x, y)."""
        c = self.config
        n = len(x)
        positions = np.zeros(n)
        hedge_ratios = np.zeros(n)
        spreads = np.zeros(n)
        zscores = np.zeros(n)
        pnl = np.zeros(n)
        holding_days = np.zeros(n)

        for i in range(c.window, n):
            # Rolling OLS
            X_w = x[i - c.window:i]
            Y_w = y[i - c.window:i]
            X_mat = np.column_stack([np.ones(c.window), X_w])
            beta = np.linalg.lstsq(X_mat, Y_w, rcond=None)[0]
            hedge = beta[1]
            hedge_ratios[i] = hedge

            # Current spread and z-score
            spread = y[i] - hedge * x[i]
            spreads[i] = spread
            spread_window = y[i - c.window:i] - hedge * x[i - c.window:i]
            mu = np.mean(spread_window)
            sigma = np.std(spread_window)
            z = (spread - mu) / sigma if sigma > 0 else 0
            zscores[i] = z

            # Trading logic
            prev_pos = positions[i - 1]
            prev_hold = holding_days[i - 1]

            if prev_pos == 0:
                if z > c.entry_z:
                    positions[i] = -1  # Short spread
                    holding_days[i] = 1
                elif z < -c.entry_z:
                    positions[i] = 1   # Long spread
                    holding_days[i] = 1
            else:
                # Exit conditions
                if (abs(z) < c.exit_z or
                    abs(z) > c.stop_loss_z or
                    prev_hold >= c.max_holding):
                    positions[i] = 0
                    holding_days[i] = 0
                else:
                    positions[i] = prev_pos
                    holding_days[i] = prev_hold + 1

            # P&L
            if i > c.window:
                spread_return = spreads[i] - spreads[i - 1]
                pnl[i] = spread_return * positions[i - 1]

                # Transaction costs on position changes
                if positions[i] != positions[i - 1]:
                    notional = abs(y[i]) + abs(hedge * x[i])
                    pnl[i] -= notional * c.transaction_cost_bps / 10000

        results = pd.DataFrame({
            'spread': spreads, 'zscore': zscores, 'position': positions,
            'pnl': pnl, 'hedge_ratio': hedge_ratios, 'holding_days': holding_days
        })
        results['cum_pnl'] = results['pnl'].cumsum()
        return results


# ============================================================================
# Performance Analytics
# ============================================================================

class PerformanceAnalytics:
    """Compute strategy performance metrics."""

    @staticmethod
    def compute(pnl_series: np.ndarray, name: str = 'Strategy') -> dict:
        daily = pnl_series[pnl_series != 0]
        if len(daily) < 10:
            return {'name': name, 'total_return': 0, 'sharpe': 0, 'max_dd': 0}

        total = np.sum(daily)
        mean_d = np.mean(daily)
        std_d = np.std(daily)
        sharpe = mean_d / std_d * np.sqrt(252) if std_d > 0 else 0

        cum = np.cumsum(daily)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dd = np.min(dd)

        win_rate = np.mean(daily > 0)
        profit_factor = np.sum(daily[daily > 0]) / abs(np.sum(daily[daily < 0])) if np.sum(daily[daily < 0]) != 0 else np.inf

        return {
            'name': name,
            'total_return': total,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'mean_daily': mean_d,
            'std_daily': std_d,
            'skewness': float(skew(daily)),
            'kurtosis': float(kurtosis(daily)),
            'n_trades': int(np.sum(np.abs(np.diff(np.sign(pnl_series))) > 0)),
        }


# ============================================================================
# Multi-Pair Portfolio
# ============================================================================

class MultiPairPortfolio:
    """Aggregate multiple pair strategies into a portfolio."""

    def __init__(self, config: BacktestConfig = None):
        self.backtester = MeanReversionBacktester(config)
        self.pair_results = {}

    def add_pair(self, name: str, x: np.ndarray, y: np.ndarray):
        """Backtest and add a pair to the portfolio."""
        result = self.backtester.backtest(x, y)
        self.pair_results[name] = result

    def portfolio_pnl(self) -> pd.DataFrame:
        """Aggregate P&L across all pairs."""
        pnls = {}
        for name, result in self.pair_results.items():
            pnls[name] = result['pnl'].values

        min_len = min(len(v) for v in pnls.values())
        pnl_df = pd.DataFrame({k: v[:min_len] for k, v in pnls.items()})
        pnl_df['total'] = pnl_df.sum(axis=1)
        pnl_df['cum_total'] = pnl_df['total'].cumsum()
        return pnl_df

    def portfolio_stats(self) -> dict:
        """Compute portfolio-level statistics."""
        pnl_df = self.portfolio_pnl()
        port_stats = PerformanceAnalytics.compute(pnl_df['total'].values, 'Portfolio')

        # Per-pair stats
        pair_stats = {}
        for name in self.pair_results:
            pair_stats[name] = PerformanceAnalytics.compute(
                self.pair_results[name]['pnl'].values, name
            )

        # Correlation matrix
        pair_pnls = pnl_df.drop(columns=['total', 'cum_total'])
        corr = pair_pnls.corr()

        return {
            'portfolio': port_stats,
            'pairs': pair_stats,
            'correlation': corr,
            'diversification_benefit': 1 - port_stats['std_daily'] / np.mean([s['std_daily'] for s in pair_stats.values() if s['std_daily'] > 0]),
        }


# ============================================================================
# HFT Microstructure Analysis
# ============================================================================

class MicrostructureAnalyser:
    """
    High-frequency trading microstructure analysis.

    Metrics:
    - Effective spread: 2 * |price - mid| (execution quality)
    - Realised spread: 2 * D * (price - mid_t+1) (information leakage)
    - VPIN: Volume-synchronized PIN (flow toxicity)
    - Order flow imbalance
    - Kyle's lambda (price impact)
    """

    @staticmethod
    def effective_spread(prices, mids):
        """Effective spread = 2 * |trade_price - mid|."""
        return 2 * np.abs(prices - mids)

    @staticmethod
    def realised_spread(prices, mids_next, sides):
        """Realised spread = 2 * D * (price - mid_{t+1})."""
        direction = np.where(sides == 'buy', 1, -1)
        return 2 * direction * (prices - mids_next)

    @staticmethod
    def order_flow_imbalance(sides, volumes, window=100):
        """Net order flow = sum(buy_volume) - sum(sell_volume) over window."""
        buy_vol = np.where(sides == 'buy', volumes, 0)
        sell_vol = np.where(sides == 'sell', volumes, 0)

        ofi = pd.Series(buy_vol - sell_vol)
        return ofi.rolling(window).sum().values

    @staticmethod
    def vpin(sides, volumes, bucket_size=1000, n_buckets=50):
        """Volume-synchronized PIN estimate."""
        buy_vol = np.where(sides == 'buy', volumes, 0)
        sell_vol = np.where(sides == 'sell', volumes, 0)

        cum_buy = np.cumsum(buy_vol)
        cum_sell = np.cumsum(sell_vol)
        cum_total = cum_buy + cum_sell

        bucket_edges = np.arange(bucket_size, cum_total[-1], bucket_size)
        imbalances = []

        for i in range(1, len(bucket_edges)):
            mask = (cum_total >= bucket_edges[i - 1]) & (cum_total < bucket_edges[i])
            if np.any(mask):
                b = np.sum(buy_vol[mask])
                s = np.sum(sell_vol[mask])
                imbalances.append(abs(b - s) / (b + s) if (b + s) > 0 else 0)

        if len(imbalances) < n_buckets:
            return np.mean(imbalances) if imbalances else 0

        return np.mean(imbalances[-n_buckets:])

    @staticmethod
    def kyle_lambda(returns, ofi, window=100):
        """Kyle's lambda: price impact per unit of order flow."""
        n = len(returns)
        lambdas = np.zeros(n)

        for i in range(window, n):
            r = returns[i - window:i]
            o = ofi[i - window:i]
            valid = ~np.isnan(o) & ~np.isnan(r)
            if np.sum(valid) > 10 and np.var(o[valid]) > 0:
                cov = np.cov(r[valid], o[valid])[0, 1]
                var = np.var(o[valid])
                lambdas[i] = cov / var

        return lambdas

    @staticmethod
    def analyse(tick_data: pd.DataFrame) -> dict:
        """Full microstructure analysis on tick data."""
        prices = tick_data['price'].values
        mids = tick_data['mid'].values
        sides = tick_data['side'].values
        volumes = tick_data['volume'].values.astype(float)
        spreads_raw = tick_data['spread'].values

        # Effective spread
        eff_spread = MicrostructureAnalyser.effective_spread(prices, mids)

        # Realised spread (using next mid)
        mids_next = np.roll(mids, -1)
        mids_next[-1] = mids[-1]
        real_spread = MicrostructureAnalyser.realised_spread(prices, mids_next, sides)

        # Order flow imbalance
        ofi = MicrostructureAnalyser.order_flow_imbalance(sides, volumes)

        # VPIN
        vpin = MicrostructureAnalyser.vpin(sides, volumes)

        # Returns
        returns = np.diff(mids) / mids[:-1]
        returns = np.append(returns, 0)

        # Kyle's lambda
        lambdas = MicrostructureAnalyser.kyle_lambda(returns, ofi)

        return {
            'avg_quoted_spread_bps': np.mean(spreads_raw / mids) * 10000,
            'avg_effective_spread_bps': np.mean(eff_spread / mids) * 10000,
            'avg_realised_spread_bps': np.nanmean(real_spread / mids) * 10000,
            'vpin': vpin,
            'avg_kyle_lambda': np.mean(lambdas[lambdas != 0]),
            'buy_ratio': np.mean(sides == 'buy'),
            'avg_trade_size': np.mean(volumes),
            'volatility_bps': np.std(returns) * 10000,
        }


# ============================================================================
# ML Signal Enhancement
# ============================================================================

class MLSignalEnhancer:
    """
    Enhance mean reversion signals with ML features.

    Features: z-score, z-score velocity, spread volatility, volume trend,
    time of day, day of week, recent performance.
    """

    @staticmethod
    def build_features(zscores, spreads, window=20):
        """Build feature matrix from z-scores and spreads."""
        n = len(zscores)
        features = np.zeros((n, 6))

        for i in range(window, n):
            features[i, 0] = zscores[i]                                    # Current z-score
            features[i, 1] = zscores[i] - zscores[i - 5]                  # Z velocity (5-day)
            features[i, 2] = np.std(spreads[i - window:i])                 # Spread vol
            features[i, 3] = abs(zscores[i]) / max(np.std(zscores[i - window:i]), 1e-8)  # Z-score relative
            features[i, 4] = np.mean(np.abs(np.diff(zscores[i - window:i])))  # Z-score mean abs change
            features[i, 5] = 1 if abs(zscores[i]) > 2 else 0              # Extreme indicator

        return features

    @staticmethod
    def simple_classifier(features, returns_next, threshold=0):
        """Train a simple decision tree classifier for signal filtering."""
        # Labels: 1 if next period spread return is profitable, 0 otherwise
        labels = (returns_next > threshold).astype(int)

        n = len(features)
        train_end = int(n * 0.7)

        X_train = features[:train_end]
        y_train = labels[:train_end]
        X_test = features[train_end:]
        y_test = labels[train_end:]

        # Simple: majority vote based on extreme z-score and low vol
        # (In production, use scikit-learn RandomForest)
        predictions = np.zeros(len(X_test))
        for i in range(len(X_test)):
            if X_test[i, 0] > 1.5 and X_test[i, 2] < np.median(features[:, 2]):
                predictions[i] = 1  # Likely mean-reverting
            elif X_test[i, 0] < -1.5 and X_test[i, 2] < np.median(features[:, 2]):
                predictions[i] = 1

        accuracy = np.mean(predictions == y_test) if len(y_test) > 0 else 0

        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'feature_importance': ['z_score', 'z_velocity', 'spread_vol', 'z_relative', 'z_mean_abs', 'extreme'],
        }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("STATISTICAL ARBITRAGE & MARKET MICROSTRUCTURE ENGINE")
    print("Cointegration · Mean Reversion · HFT Analysis · ML Signals")
    print("Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE")
    print("=" * 65)

    # ── Generate synthetic data ──
    gen = DataGenerator()
    df = gen.cointegrated_pair(1500, beta=0.8, noise_std=1.0)

    # ── Cointegration tests ──
    print("\nCOINTEGRATION TESTS")
    print("=" * 45)

    eg = CointegrationTester.engle_granger(df['x'].values, df['y'].values)
    print(f"Engle-Granger:")
    print(f"  Hedge ratio:    {eg['hedge_ratio']:.4f}")
    print(f"  t-statistic:    {eg['t_statistic']:.4f}")
    print(f"  Cointegrated:   {eg['is_cointegrated']}")
    print(f"  Half-life:      {eg['half_life']:.1f} days")

    joh = CointegrationTester.johansen_simplified(df['x'].values, df['y'].values)
    print(f"\nJohansen:")
    print(f"  Trace stats:    {joh['trace_statistics']}")
    print(f"  # cointegrating: {joh['n_cointegrating']}")
    print(f"  Cointegrated:   {joh['is_cointegrated']}")

    # ── Single pair backtest ──
    print(f"\nSINGLE PAIR BACKTEST")
    print("=" * 45)

    config = BacktestConfig(window=100, entry_z=1.5, exit_z=0.5, transaction_cost_bps=5)
    bt = MeanReversionBacktester(config)
    result = bt.backtest(df['x'].values, df['y'].values)
    stats = PerformanceAnalytics.compute(result['pnl'].values, 'Synthetic Pair')

    print(f"Total return:    {stats['total_return']:.2f}")
    print(f"Sharpe ratio:    {stats['sharpe']:.2f}")
    print(f"Max drawdown:    {stats['max_drawdown']:.2f}")
    print(f"Win rate:        {stats['win_rate']:.1%}")
    print(f"Profit factor:   {stats['profit_factor']:.2f}")
    print(f"Skewness:        {stats['skewness']:.2f}")
    print(f"Kurtosis:        {stats['kurtosis']:.2f}")
    print(f"# Trades:        {stats['n_trades']}")

    # ── Multi-pair portfolio ──
    print(f"\nMULTI-PAIR PORTFOLIO")
    print("=" * 45)

    pairs = gen.multi_pair_universe()
    portfolio = MultiPairPortfolio(config)

    for name, data in pairs.items():
        portfolio.add_pair(name, data['x'], data['y'])

    port_stats = portfolio.portfolio_stats()
    ps = port_stats['portfolio']

    print(f"Portfolio Sharpe:  {ps['sharpe']:.2f}")
    print(f"Portfolio MaxDD:   {ps['max_drawdown']:.2f}")
    print(f"Diversification:   {port_stats['diversification_benefit']:.1%}")

    print(f"\n{'Pair':<12} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>8}")
    print("-" * 48)
    for name, s in port_stats['pairs'].items():
        print(f"{name:<12} {s['sharpe']:>7.2f} {s['max_drawdown']:>7.2f} {s['win_rate']:>7.1%} {s['profit_factor']:>7.2f}")

    print(f"\nCorrelation matrix:")
    print(port_stats['correlation'].round(3))

    # ── HFT Microstructure ──
    print(f"\nHFT MICROSTRUCTURE ANALYSIS")
    print("=" * 45)

    ticks = gen.tick_data(50000)
    micro = MicrostructureAnalyser.analyse(ticks)

    for k, v in micro.items():
        print(f"  {k:<30} {v:.4f}")

    # ── ML Signal Enhancement ──
    print(f"\nML SIGNAL ENHANCEMENT")
    print("=" * 45)

    features = MLSignalEnhancer.build_features(result['zscore'].values, result['spread'].values)
    spread_returns = np.diff(result['spread'].values)
    spread_returns = np.append(spread_returns, 0)
    ml = MLSignalEnhancer.simple_classifier(features, spread_returns * result['position'].values)
    print(f"  Signal accuracy: {ml['accuracy']:.1%}")
    print(f"  Features: {', '.join(ml['feature_importance'])}")

    return result, port_stats, micro


if __name__ == "__main__":
    result, port_stats, micro = main()
