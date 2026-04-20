"""
Systematic Trading Engine - Visualisations
"""

import numpy as np
import matplotlib.pyplot as plt
from stat_arb_engine import (
    DataGenerator, CointegrationTester, MeanReversionBacktester,
    BacktestConfig, MultiPairPortfolio, MicrostructureAnalyser, PerformanceAnalytics
)
from transaction_costs import backtest_with_costs, ExecutionCosts, cost_sensitivity_analysis

GOLD = '#C9A96E'; BLUE = '#7DAFCB'; RED = '#CB7D7D'
GREEN = '#7DCB8A'; PURPLE = '#B07DCB'; BG = '#0C0C0E'
CARD = '#141417'; TEXT = '#F0EDE6'; MUTED = '#9B978E'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': '#2A2A2F',
    'axes.labelcolor': MUTED, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': '#2A2A2F', 'grid.alpha': 0.5, 'font.size': 10,
})
OUT = '/home/claude/systematic-trading/'


def plot_cointegration_and_spread():
    gen = DataGenerator()
    df = gen.cointegrated_pair(1500, beta=0.8)
    eg = CointegrationTester.engle_granger(df['x'].values, df['y'].values)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    ax.plot(df['x'].values, color=BLUE, linewidth=1, label='Asset X')
    ax.plot(df['y'].values, color=GOLD, linewidth=1, label='Asset Y')
    ax.set_title('Cointegrated Pair', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(eg['residuals'], color=GREEN, linewidth=0.8)
    ax.axhline(0, color=MUTED, linestyle='--', linewidth=0.5)
    ax.set_title(f'Spread (hedge ratio={eg["hedge_ratio"]:.3f})', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Backtest
    config = BacktestConfig(window=100, entry_z=1.5, exit_z=0.5)
    bt = MeanReversionBacktester(config)
    res = bt.backtest(df['x'].values, df['y'].values)

    ax = axes[2]
    ax.plot(res['cum_pnl'].values, color=GOLD, linewidth=1.5)
    ax.set_title('Cumulative P&L', color=TEXT, fontweight='bold')
    ax.set_ylabel('P&L')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT + 'cointegration_spread.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cointegration_spread.png")


def plot_zscore_signals():
    gen = DataGenerator()
    df = gen.cointegrated_pair(1500, beta=0.8)
    config = BacktestConfig(window=100, entry_z=1.5, exit_z=0.5)
    bt = MeanReversionBacktester(config)
    res = bt.backtest(df['x'].values, df['y'].values)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [2, 1]})

    ax = axes[0]
    z = res['zscore'].values
    ax.plot(z, color=BLUE, linewidth=0.7, alpha=0.8)
    ax.axhline(1.5, color=RED, linestyle='--', linewidth=0.8, label='Entry ±1.5')
    ax.axhline(-1.5, color=RED, linestyle='--', linewidth=0.8)
    ax.axhline(0.5, color=GREEN, linestyle='--', linewidth=0.8, label='Exit ±0.5')
    ax.axhline(-0.5, color=GREEN, linestyle='--', linewidth=0.8)
    ax.axhline(0, color=MUTED, linewidth=0.3)
    ax.fill_between(range(len(z)), z, 0, where=z > 1.5, color=RED, alpha=0.15)
    ax.fill_between(range(len(z)), z, 0, where=z < -1.5, color=GREEN, alpha=0.15)
    ax.set_title('Z-Score & Trading Signals', color=TEXT, fontweight='bold', fontsize=13)
    ax.set_ylabel('Z-Score')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    pos = res['position'].values
    ax.fill_between(range(len(pos)), pos, 0, where=pos > 0, color=GREEN, alpha=0.6, label='Long spread')
    ax.fill_between(range(len(pos)), pos, 0, where=pos < 0, color=RED, alpha=0.6, label='Short spread')
    ax.set_title('Position', color=TEXT, fontweight='bold')
    ax.set_ylabel('Position')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT + 'zscore_signals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: zscore_signals.png")


def plot_multi_pair():
    gen = DataGenerator()
    pairs = gen.multi_pair_universe()
    config = BacktestConfig(window=100, entry_z=1.5, exit_z=0.5)
    portfolio = MultiPairPortfolio(config)
    for name, data in pairs.items():
        portfolio.add_pair(name, data['x'], data['y'])
    pnl_df = portfolio.portfolio_pnl()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = [GOLD, BLUE, GREEN, RED]
    for i, col_name in enumerate([c for c in pnl_df.columns if c not in ['total', 'cum_total']]):
        cum = pnl_df[col_name].cumsum()
        ax.plot(cum.values, color=colors[i % len(colors)], linewidth=1, alpha=0.7, label=col_name)
    ax.plot(pnl_df['cum_total'].values, color=TEXT, linewidth=2, label='Portfolio')
    ax.set_title('Multi-Pair Portfolio P&L', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Correlation heatmap
    ax = axes[1]
    pair_cols = [c for c in pnl_df.columns if c not in ['total', 'cum_total']]
    corr = pnl_df[pair_cols].corr()
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(pair_cols)))
    ax.set_yticks(range(len(pair_cols)))
    ax.set_xticklabels(pair_cols, fontsize=9)
    ax.set_yticklabels(pair_cols, fontsize=9)
    for i in range(len(pair_cols)):
        for j in range(len(pair_cols)):
            ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=9, color='black')
    ax.set_title('Pair Correlation Matrix', color=TEXT, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT + 'multi_pair_portfolio.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: multi_pair_portfolio.png")


def plot_cost_sensitivity():
    gen = DataGenerator()
    df = gen.cointegrated_pair(1500, beta=0.8)
    config = BacktestConfig(window=100, entry_z=1.5, exit_z=0.5)
    sens = cost_sensitivity_analysis(df['x'].values, df['y'].values, config)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(sens['total_cost_bps'], sens['gross_sharpe'], 'o-', color=GREEN, linewidth=2, markersize=5, label='Gross')
    ax.plot(sens['total_cost_bps'], sens['net_sharpe'], 's-', color=RED, linewidth=2, markersize=5, label='Net')
    ax.axhline(0, color=MUTED, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Total Cost (bps)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe vs Transaction Costs', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(sens['total_cost_bps'], sens['alpha_decay'] * 100, color=GOLD, alpha=0.8, width=2)
    ax.set_xlabel('Total Cost (bps)')
    ax.set_ylabel('Alpha Decay (%)')
    ax.set_title('Alpha Decay vs Cost', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT + 'cost_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cost_sensitivity.png")


def plot_microstructure():
    gen = DataGenerator()
    ticks = gen.tick_data(50000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Spread over time
    ax = axes[0, 0]
    spreads = ticks['spread'].values[:5000]
    ax.plot(spreads * 10000, color=GOLD, linewidth=0.5, alpha=0.7)
    ax.set_title('Quoted Spread (bps)', color=TEXT, fontweight='bold')
    ax.set_ylabel('bps')
    ax.grid(True, alpha=0.3)

    # Order flow imbalance
    ax = axes[0, 1]
    ofi = MicrostructureAnalyser.order_flow_imbalance(ticks['side'].values, ticks['volume'].values.astype(float), window=200)
    ax.plot(ofi[:5000], color=BLUE, linewidth=0.5, alpha=0.7)
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_title('Order Flow Imbalance', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Mid price
    ax = axes[1, 0]
    ax.plot(ticks['mid'].values[:5000], color=GREEN, linewidth=0.5)
    ax.set_title('Mid Price', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Return distribution
    ax = axes[1, 1]
    rets = np.diff(ticks['mid'].values) / ticks['mid'].values[:-1]
    ax.hist(rets * 10000, bins=100, color=PURPLE, alpha=0.7, edgecolor='none')
    ax.set_title('Return Distribution (bps)', color=TEXT, fontweight='bold')
    ax.set_xlabel('bps')
    ax.grid(True, alpha=0.3)

    fig.suptitle('HFT Microstructure Analysis', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'microstructure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: microstructure.png")


if __name__ == "__main__":
    print("Generating visualisations...\n")
    plot_cointegration_and_spread()
    plot_zscore_signals()
    plot_multi_pair()
    plot_cost_sensitivity()
    plot_microstructure()
    print("\nAll charts generated.")
