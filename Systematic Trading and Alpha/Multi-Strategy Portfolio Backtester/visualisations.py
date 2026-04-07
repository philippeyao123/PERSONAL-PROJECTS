"""
Multi-Strategy Portfolio Backtester - Visualisations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from portfolio_backtester import (
    create_pod_shop_portfolio, Backtester, DrawdownAnalyser, RiskBudgeter
)

GOLD = '#C9A96E'; GOLD_LIGHT = '#E8D5A8'; BLUE = '#7DAFCB'
RED = '#CB7D7D'; GREEN = '#7DCB8A'; PURPLE = '#B07DCB'; BG = '#0C0C0E'
CARD = '#141417'; TEXT = '#F0EDE6'; MUTED = '#9B978E'
COLORS = [GOLD, BLUE, GREEN, RED, PURPLE, '#CB9E7D', '#7DCBCB']

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': '#2A2A2F',
    'axes.labelcolor': MUTED, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': '#2A2A2F', 'grid.alpha': 0.5, 'font.size': 10,
})

OUT = '/home/claude/portfolio-backtester/'


def run():
    strategies, config = create_pod_shop_portfolio()
    bt = Backtester(strategies, config, seed=42)
    return bt.run(n_years=3.0), strategies


def plot_cumulative_returns(results, strategies):
    """Cumulative returns: portfolio + individual strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    days = np.arange(results['n_days'])
    
    # Panel 1: Cumulative returns
    ax = axes[0]
    ax.plot(days, results['cum_port'] * 100, color=TEXT, linewidth=2.5, label='Portfolio', zorder=10)
    for i, strat in enumerate(strategies):
        ax.plot(days, results['cum_strat'][:, i] * 100, color=COLORS[i], linewidth=1, alpha=0.7, label=strat.name)
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Multi-Strategy Portfolio Performance', fontsize=14, color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Underwater chart
    ax = axes[1]
    dd = DrawdownAnalyser.underwater_chart_data(results['cum_port'])
    ax.fill_between(days, dd * 100, alpha=0.5, color=RED)
    ax.plot(days, dd * 100, color=RED, linewidth=1)
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Trading Days')
    ax.set_title('Portfolio Drawdown', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT + 'cumulative_returns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cumulative_returns.png")


def plot_risk_attribution(results, strategies):
    """Risk contribution and weight comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    names = [s.name.replace(' ', '\n') for s in strategies]
    
    # Weights comparison
    ax = axes[0]
    x = np.arange(len(strategies))
    w = 0.35
    ax.bar(x - w/2, results['erc_weights'], w, color=GOLD, alpha=0.8, label='ERC')
    ax.bar(x + w/2, results['inv_vol_weights'], w, color=BLUE, alpha=0.8, label='Inv Vol')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title('Weight Allocation', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Risk contributions (pie)
    ax = axes[1]
    rc = results['risk_contributions']
    wedges, texts, autotexts = ax.pie(
        rc, labels=[s.name for s in strategies],
        colors=COLORS[:len(strategies)], autopct='%.0f%%',
        textprops={'color': TEXT, 'fontsize': 8},
        pctdistance=0.75, startangle=90
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color(BG)
    ax.set_title('Risk Contribution', color=TEXT, fontweight='bold')
    
    # Factor attribution
    ax = axes[2]
    fa = results['factor_attribution']
    factor_names = list(fa.keys())
    factor_vals = [fa[f]['annualised'] * 100 for f in factor_names]
    colors_bar = [BLUE, GREEN, PURPLE, GOLD]
    bars = ax.bar(factor_names, factor_vals, color=colors_bar[:len(factor_names)], alpha=0.8)
    for bar, val in zip(bars, factor_vals):
        ypos = bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.3
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:+.1f}%',
               ha='center', color=TEXT, fontsize=10, fontweight='bold')
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_ylabel('Annual Return (%)')
    ax.set_title('Factor Attribution', color=TEXT, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.suptitle('Risk & Attribution Analysis', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'risk_attribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: risk_attribution.png")


def plot_strategy_metrics(results, strategies):
    """Strategy-level metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    names = [s.name for s in strategies]
    n = len(names)
    x = np.arange(n)
    
    metrics = results['strat_metrics']
    
    # Sharpe ratios
    ax = axes[0, 0]
    sharpes = [metrics[s.name]['sharpe'] for s in strategies]
    colors_bar = [GREEN if s > 0 else RED for s in sharpes]
    ax.bar(x, sharpes, color=colors_bar, alpha=0.8)
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=15)
    ax.set_title('Sharpe Ratio', color=TEXT, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Max Drawdown
    ax = axes[0, 1]
    dds = [metrics[s.name]['max_drawdown'] * 100 for s in strategies]
    ax.bar(x, dds, color=RED, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=15)
    ax.set_title('Max Drawdown (%)', color=TEXT, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Return vs Vol scatter
    ax = axes[1, 0]
    for i, s in enumerate(strategies):
        m = metrics[s.name]
        ax.scatter(m['ann_vol'] * 100, m['ann_return'] * 100, color=COLORS[i], s=100, zorder=5, label=s.name)
    # Portfolio
    pm = results['port_metrics']
    ax.scatter(pm['ann_vol'] * 100, pm['ann_return'] * 100, color=TEXT, s=150, marker='*', zorder=10, label='Portfolio')
    ax.set_xlabel('Annual Vol (%)')
    ax.set_ylabel('Annual Return (%)')
    ax.set_title('Risk-Return', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Tail risk
    ax = axes[1, 1]
    cvars = [metrics[s.name]['cvar_95'] * 100 for s in strategies]
    skews = [metrics[s.name]['skewness'] for s in strategies]
    ax.bar(x - 0.2, cvars, 0.4, color=RED, alpha=0.8, label='CVaR 95 (%)')
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, skews, 0.4, color=BLUE, alpha=0.8, label='Skewness')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=15)
    ax.set_title('Tail Risk', color=TEXT, fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.suptitle('Strategy Performance Comparison', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'strategy_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: strategy_metrics.png")


def plot_correlation_regime(results, strategies):
    """Rolling correlation and diversification."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    
    # Rolling avg correlation
    ax = axes[0]
    corr_series = results['correlation_series']
    days = np.arange(len(corr_series))
    ax.plot(days, corr_series, color=GOLD, linewidth=1.5)
    ax.fill_between(days, corr_series, alpha=0.2, color=GOLD)
    ax.axhline(0, color=MUTED, linewidth=0.5, linestyle='--')
    ax.set_ylabel('Avg Pairwise Correlation')
    ax.set_title('Rolling Strategy Correlation (63-day window)', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Correlation heatmap
    ax = axes[1]
    returns = results['returns']
    corr_matrix = np.corrcoef(returns.T)
    names = [s.name for s in strategies]
    
    im = ax.imshow(corr_matrix, cmap='RdYlGn_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=20)
    ax.set_yticklabels(names, fontsize=8)
    
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                   color='white' if abs(corr_matrix[i,j]) > 0.3 else TEXT, fontsize=9)
    
    ax.set_title('Strategy Correlation Matrix', color=TEXT, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig.tight_layout()
    fig.savefig(OUT + 'correlation_regime.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_regime.png")


def plot_risk_dashboard(results, strategies):
    """Pod risk dashboard summary."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    dash = results['risk_dashboard']
    pm = results['port_metrics']
    
    # Header
    header = (
        f"DAILY RISK DASHBOARD\n"
        f"{'='*60}\n"
        f"Portfolio: Return {pm['ann_return']:+.1%} | Vol {pm['ann_vol']:.1%} | "
        f"Sharpe {pm['sharpe']:.2f} | MaxDD {pm['max_drawdown']:.1%}\n"
        f"Diversification Ratio: {results['diversification_ratio']:.2f} | "
        f"Breaches: {len(results['breach_log'])}\n\n"
    )
    
    rows = f"{'Strategy':<25} {'YTD':>8} {'DD':>8} {'Vol':>8} {'Sharpe':>8} {'Status':>10}\n"
    rows += "-" * 70 + "\n"
    for name, d in dash.items():
        status_color = 'OK' if d['status'] == 'OK' else d['status']
        rows += (f"{name:<25} {d['ytd_return']:>+7.1%} {d['current_dd']:>7.1%} "
                f"{d['rolling_vol']:>7.1%} {d['rolling_sharpe']:>7.2f} {d['status']:>10}\n")
    
    ax.text(0.05, 0.95, header + rows, transform=ax.transAxes,
           fontfamily='monospace', fontsize=11, color=GOLD, verticalalignment='top')
    
    fig.tight_layout()
    fig.savefig(OUT + 'risk_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: risk_dashboard.png")


if __name__ == "__main__":
    print("Generating visualisations...\n")
    results, strategies = run()
    
    plot_cumulative_returns(results, strategies)
    plot_risk_attribution(results, strategies)
    plot_strategy_metrics(results, strategies)
    plot_correlation_regime(results, strategies)
    plot_risk_dashboard(results, strategies)
    
    print("\nAll charts generated.")
