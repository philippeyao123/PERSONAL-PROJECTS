"""
Extension 2: Transaction Costs & Execution Simulation
========================================================
Realistic execution modelling:
  - Spread crossing costs (bid-ask)
  - Market impact (Kyle's lambda × order size)
  - Slippage from delay
  - Borrow costs for shorts
  - Comparison: gross vs net P&L at different cost levels

Shows how alpha decays as execution costs increase — critical for
any systematic strategy that trades frequently.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from stat_arb_engine import (
    DataGenerator, MeanReversionBacktester, BacktestConfig, PerformanceAnalytics
)


@dataclass
class ExecutionCosts:
    """Execution cost model."""
    spread_bps: float = 3.0         # Half spread in bps
    impact_bps: float = 2.0         # Market impact per trade in bps
    slippage_bps: float = 1.0       # Execution delay slippage
    borrow_bps_annual: float = 50.0 # Short borrow cost (annualized)
    commission_bps: float = 1.0     # Broker commission per trade

    @property
    def one_way_cost_bps(self):
        return self.spread_bps + self.impact_bps + self.slippage_bps + self.commission_bps

    @property
    def daily_borrow_bps(self):
        return self.borrow_bps_annual / 252


def backtest_with_costs(x, y, config: BacktestConfig, costs: ExecutionCosts) -> pd.DataFrame:
    """Backtest with detailed execution cost model."""
    n = len(x)
    window = config.window
    positions = np.zeros(n)
    spreads = np.zeros(n)
    zscores = np.zeros(n)
    pnl_gross = np.zeros(n)
    pnl_net = np.zeros(n)
    cost_breakdown = np.zeros((n, 4))  # spread, impact, slippage, borrow
    holding = np.zeros(n)

    for i in range(window, n):
        X_w = x[i - window:i]
        Y_w = y[i - window:i]
        X_mat = np.column_stack([np.ones(window), X_w])
        beta = np.linalg.lstsq(X_mat, Y_w, rcond=None)[0]
        hedge = beta[1]

        spread = y[i] - hedge * x[i]
        spreads[i] = spread
        sw = y[i - window:i] - hedge * x[i - window:i]
        mu, sigma = np.mean(sw), np.std(sw)
        z = (spread - mu) / sigma if sigma > 0 else 0
        zscores[i] = z

        prev = positions[i - 1]
        prev_hold = holding[i - 1]

        if prev == 0:
            if z > config.entry_z:
                positions[i] = -1
                holding[i] = 1
            elif z < -config.entry_z:
                positions[i] = 1
                holding[i] = 1
        else:
            if abs(z) < config.exit_z or abs(z) > config.stop_loss_z or prev_hold >= config.max_holding:
                positions[i] = 0
                holding[i] = 0
            else:
                positions[i] = prev
                holding[i] = prev_hold + 1

        if i > window:
            spread_return = spreads[i] - spreads[i - 1]
            gross_pnl = spread_return * positions[i - 1]
            pnl_gross[i] = gross_pnl

            # Cost calculations
            notional = abs(y[i]) + abs(hedge * x[i])
            trade_cost = 0

            # Trading costs (on position changes)
            if positions[i] != positions[i - 1]:
                spread_cost = notional * costs.spread_bps / 10000
                impact_cost = notional * costs.impact_bps / 10000
                slip_cost = notional * costs.slippage_bps / 10000
                commission = notional * costs.commission_bps / 10000
                trade_cost = spread_cost + impact_cost + slip_cost + commission
                cost_breakdown[i] = [spread_cost, impact_cost, slip_cost, commission]

            # Borrow cost (daily, on short positions)
            borrow_cost = 0
            if positions[i - 1] != 0:
                borrow_cost = notional * costs.daily_borrow_bps / 10000

            pnl_net[i] = gross_pnl - trade_cost - borrow_cost

    results = pd.DataFrame({
        'spread': spreads, 'zscore': zscores, 'position': positions,
        'pnl_gross': pnl_gross, 'pnl_net': pnl_net,
        'spread_cost': cost_breakdown[:, 0], 'impact_cost': cost_breakdown[:, 1],
        'slippage_cost': cost_breakdown[:, 2], 'commission': cost_breakdown[:, 3],
    })
    results['cum_gross'] = results['pnl_gross'].cumsum()
    results['cum_net'] = results['pnl_net'].cumsum()
    results['total_costs'] = (results['cum_gross'] - results['cum_net'])
    return results


def cost_sensitivity_analysis(x, y, config: BacktestConfig) -> pd.DataFrame:
    """Analyse how Sharpe decays across cost levels."""
    cost_levels = [0, 1, 3, 5, 8, 10, 15, 20, 30]
    results = []

    for c in cost_levels:
        costs = ExecutionCosts(spread_bps=c / 3, impact_bps=c / 3, slippage_bps=c / 6, commission_bps=c / 6)
        res = backtest_with_costs(x, y, config, costs)
        stats_net = PerformanceAnalytics.compute(res['pnl_net'].values, f'{c}bps')
        stats_gross = PerformanceAnalytics.compute(res['pnl_gross'].values, f'{c}bps_gross')

        results.append({
            'total_cost_bps': c,
            'gross_sharpe': stats_gross['sharpe'],
            'net_sharpe': stats_net['sharpe'],
            'gross_return': stats_gross['total_return'],
            'net_return': stats_net['total_return'],
            'alpha_decay': 1 - stats_net['total_return'] / max(stats_gross['total_return'], 1e-8),
        })

    return pd.DataFrame(results)


def turnover_analysis(positions: np.ndarray) -> dict:
    """Analyse trading frequency and turnover."""
    changes = np.abs(np.diff(positions))
    trades = np.sum(changes > 0)
    n_days = len(positions)

    return {
        'total_trades': int(trades),
        'trades_per_year': trades / n_days * 252,
        'avg_holding_period': n_days / max(trades, 1),
        'daily_turnover': np.mean(changes),
    }


if __name__ == "__main__":
    gen = DataGenerator()
    df = gen.cointegrated_pair(1500, beta=0.8)

    print("Transaction Costs & Execution Simulation")
    print("=" * 55)

    config = BacktestConfig(window=100, entry_z=1.5, exit_z=0.5)
    x, y = df['x'].values, df['y'].values

    # Detailed cost run
    costs = ExecutionCosts()
    res = backtest_with_costs(x, y, config, costs)

    gross_stats = PerformanceAnalytics.compute(res['pnl_gross'].values, 'Gross')
    net_stats = PerformanceAnalytics.compute(res['pnl_net'].values, 'Net')

    print(f"\nCost model: spread={costs.spread_bps}bps, impact={costs.impact_bps}bps, "
          f"slippage={costs.slippage_bps}bps, commission={costs.commission_bps}bps")
    print(f"One-way cost: {costs.one_way_cost_bps}bps")

    print(f"\n{'':20} {'Gross':>10} {'Net':>10} {'Drag':>10}")
    print("-" * 52)
    print(f"{'Total Return':<20} {gross_stats['total_return']:>10.2f} {net_stats['total_return']:>10.2f} {gross_stats['total_return'] - net_stats['total_return']:>10.2f}")
    print(f"{'Sharpe':<20} {gross_stats['sharpe']:>10.2f} {net_stats['sharpe']:>10.2f} {gross_stats['sharpe'] - net_stats['sharpe']:>10.2f}")
    print(f"{'Max Drawdown':<20} {gross_stats['max_drawdown']:>10.2f} {net_stats['max_drawdown']:>10.2f}")
    print(f"{'Win Rate':<20} {gross_stats['win_rate']:>10.1%} {net_stats['win_rate']:>10.1%}")

    # Cost breakdown
    total_costs = res['total_costs'].iloc[-1]
    print(f"\nTotal cost drag: {total_costs:.2f}")
    print(f"  Spread crossing: {res['spread_cost'].sum():.2f}")
    print(f"  Market impact:   {res['impact_cost'].sum():.2f}")
    print(f"  Slippage:        {res['slippage_cost'].sum():.2f}")
    print(f"  Commission:      {res['commission'].sum():.2f}")

    # Turnover
    turn = turnover_analysis(res['position'].values)
    print(f"\nTurnover:")
    for k, v in turn.items():
        print(f"  {k}: {v:.1f}")

    # Sensitivity
    print(f"\nCOST SENSITIVITY")
    print("-" * 60)
    sens = cost_sensitivity_analysis(x, y, config)
    print(f"{'Cost (bps)':<12} {'Gross Sharpe':>13} {'Net Sharpe':>12} {'Alpha Decay':>12}")
    print("-" * 52)
    for _, row in sens.iterrows():
        print(f"{row['total_cost_bps']:<12.0f} {row['gross_sharpe']:>13.2f} {row['net_sharpe']:>12.2f} {row['alpha_decay']:>11.0%}")

    # Break-even cost
    be = sens[sens['net_sharpe'] > 0]
    if len(be) > 0:
        max_cost = be['total_cost_bps'].max()
        print(f"\nBreak-even cost: ~{max_cost:.0f} bps (strategy unprofitable above this)")
