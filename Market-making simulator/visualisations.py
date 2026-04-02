"""
Market-Making Simulator - Visualisations
==========================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from market_maker import (
    MarketParams, MMStrategy, MarketMakingSimulator,
    PerformanceAnalytics, run_multi_day, gamma_sensitivity
)

GOLD = '#C9A96E'
GOLD_LIGHT = '#E8D5A8'
BLUE = '#7DAFCB'
RED = '#CB7D7D'
GREEN = '#7DCB8A'
BG = '#0C0C0E'
CARD = '#141417'
TEXT = '#F0EDE6'
MUTED = '#9B978E'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': '#2A2A2F',
    'axes.labelcolor': MUTED, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': '#2A2A2F', 'grid.alpha': 0.5, 'font.size': 10,
})

OUT = '/home/claude/market-maker/'


def plot_single_day(state):
    """Full single-day trading dashboard."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1.2, 1, 1]})
    
    mid = np.array(state.mid_history)
    bid = np.array(state.bid_history)
    ask = np.array(state.ask_history)
    pnl = np.array(state.pnl_history)
    inv = np.array(state.inventory_history)
    regimes = np.array(state.regime_history)
    t = np.arange(len(mid))
    
    # Panel 1: Mid-price with quotes and regime shading
    ax = axes[0]
    # Shade high-vol regime
    high_vol = regimes == 'high'
    for i in range(len(high_vol) - 1):
        if high_vol[i]:
            ax.axvspan(i, i+1, alpha=0.15, color=RED, linewidth=0)
    
    ax.plot(t, mid, color=TEXT, linewidth=0.8, alpha=0.9, label='Mid')
    ax.plot(t, bid, color=GREEN, linewidth=0.4, alpha=0.5, label='Bid')
    ax.plot(t, ask, color=RED, linewidth=0.4, alpha=0.5, label='Ask')
    ax.fill_between(t, bid, ask, alpha=0.1, color=GOLD)
    ax.set_ylabel('Price')
    ax.set_title('Mid-Price, Quotes & Regime', color=TEXT, fontweight='bold', fontsize=13)
    ax.legend(loc='upper left', fontsize=8, facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.2)
    
    # Panel 2: P&L
    ax = axes[1]
    ax.fill_between(t, pnl, alpha=0.3, color=GOLD)
    ax.plot(t, pnl, color=GOLD, linewidth=1.5)
    ax.set_ylabel('Cumulative P&L ($)')
    ax.set_title('P&L', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # Panel 3: Inventory
    ax = axes[2]
    colors = np.where(np.array(inv) > 0, GREEN, RED)
    ax.bar(t, inv, color=colors, alpha=0.6, width=1.0)
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_ylabel('Inventory')
    ax.set_title('Inventory Position', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # Panel 4: Spread
    ax = axes[3]
    spreads_bps = np.array(state.spread_history) / mid * 10000
    ax.plot(t, spreads_bps, color=BLUE, linewidth=0.8, alpha=0.7)
    ax.set_ylabel('Spread (bps)')
    ax.set_xlabel('Time Step')
    ax.set_title('Quoted Spread', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    fig.tight_layout()
    fig.savefig(OUT + 'single_day.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: single_day.png")


def plot_multi_day(multi):
    """Multi-day aggregate results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Cumulative P&L
    ax = axes[0, 0]
    days = np.arange(1, multi['n_days'] + 1)
    ax.fill_between(days, multi['cumulative_pnl'], alpha=0.3, color=GOLD)
    ax.plot(days, multi['cumulative_pnl'], color=GOLD, linewidth=2)
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Cumulative P&L ($)')
    ax.set_title('Cumulative P&L', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Daily P&L histogram
    ax = axes[0, 1]
    ax.hist(multi['daily_pnl'], bins=15, color=GOLD, alpha=0.8, edgecolor='none')
    ax.axvline(multi['mean_daily_pnl'], color=GOLD_LIGHT, linestyle='--', linewidth=2, 
               label=f'Mean: ${multi["mean_daily_pnl"]:,.0f}')
    ax.set_xlabel('Daily P&L ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Daily P&L Distribution', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    # Daily P&L bar chart
    ax = axes[1, 0]
    colors = [GREEN if p > 0 else RED for p in multi['daily_pnl']]
    ax.bar(days, multi['daily_pnl'], color=colors, alpha=0.8)
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('P&L ($)')
    ax.set_title('Daily P&L', color=TEXT, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Summary stats text
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = (
        f"AGGREGATE STATISTICS\n"
        f"{'='*30}\n\n"
        f"Mean daily P&L:  ${multi['mean_daily_pnl']:>12,.0f}\n"
        f"Std daily P&L:   ${multi['std_daily_pnl']:>12,.0f}\n"
        f"Sharpe ratio:    {multi['sharpe_ratio']:>12.1f}\n"
        f"Win rate:        {multi['win_rate']:>11.0%}\n"
        f"Max daily loss:  ${multi['max_daily_loss']:>12,.0f}\n"
        f"Max daily gain:  ${multi['max_daily_gain']:>12,.0f}\n"
        f"Avg volume/day:  {multi['avg_volume']:>12,.0f}\n"
        f"Total P&L:       ${multi['cumulative_pnl'][-1]:>12,.0f}"
    )
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
            fontfamily='monospace', color=GOLD, verticalalignment='center')
    
    fig.suptitle('Multi-Day Performance (20 Trading Days)', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'multi_day.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: multi_day.png")


def plot_gamma_sensitivity(gamma_results):
    """Risk aversion parameter sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    gammas = list(gamma_results.keys())
    mean_pnl = [gamma_results[g]['mean_pnl'] for g in gammas]
    sharpe = [gamma_results[g]['sharpe'] for g in gammas]
    max_inv = [gamma_results[g]['avg_max_inv'] for g in gammas]
    
    # Mean P&L vs Gamma
    ax = axes[0]
    ax.plot(gammas, mean_pnl, color=GOLD, linewidth=2, marker='o', markersize=6)
    ax.set_xlabel('Gamma (risk aversion)')
    ax.set_ylabel('Mean Daily P&L ($)')
    ax.set_title('P&L vs Risk Aversion', color=TEXT, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Sharpe vs Gamma
    ax = axes[1]
    ax.plot(gammas, sharpe, color=GREEN, linewidth=2, marker='o', markersize=6)
    ax.set_xlabel('Gamma (risk aversion)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe vs Risk Aversion', color=TEXT, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Max Inventory vs Gamma
    ax = axes[2]
    ax.plot(gammas, max_inv, color=RED, linewidth=2, marker='o', markersize=6)
    ax.set_xlabel('Gamma (risk aversion)')
    ax.set_ylabel('Avg Max Inventory')
    ax.set_title('Inventory vs Risk Aversion', color=TEXT, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Gamma Sensitivity Analysis', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'gamma_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: gamma_sensitivity.png")


def plot_pnl_decomposition(state):
    """P&L decomposition: spread capture vs inventory carry."""
    analytics = PerformanceAnalytics(state)
    decomp = analytics.decompose_pnl()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    labels = ['Spread\nCapture', 'Inventory\nCarry', 'Total\nP&L']
    values = [decomp['spread_pnl'], decomp['inventory_pnl'], decomp['total_pnl']]
    colors = [GOLD, BLUE, GREEN]
    
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='none')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
               f'${val:,.0f}', ha='center', va='bottom', color=TEXT, fontsize=10, fontweight='bold')
    
    ax.set_ylabel('P&L ($)')
    ax.set_title('P&L Decomposition', color=TEXT, fontweight='bold', fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT + 'pnl_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pnl_decomposition.png")


if __name__ == "__main__":
    print("Generating visualisations...\n")
    
    # Single day
    sim = MarketMakingSimulator(seed=42)
    state = sim.run(n_steps=2340)
    state.book = sim.book
    plot_single_day(state)
    plot_pnl_decomposition(state)
    
    # Multi day
    multi = run_multi_day(n_days=20, steps_per_day=2340, seed=42)
    plot_multi_day(multi)
    
    # Gamma sensitivity
    gamma_results = gamma_sensitivity(n_days=10, steps_per_day=2340)
    plot_gamma_sensitivity(gamma_results)
    
    print("\nAll charts generated.")
