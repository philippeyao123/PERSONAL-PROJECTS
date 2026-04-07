"""
Autocallable Worst-of Pricing - Visualisations
================================================
Generates publication-quality charts for the project.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from autocallable_pricer import (
    AutocallableNote, MarketData, MonteCarloEngine,
    AutocallablePayoff, run_scenario_analysis
)

plt.rcParams.update({
    'figure.facecolor': '#0C0C0E',
    'axes.facecolor': '#141417',
    'axes.edgecolor': '#2A2A2F',
    'axes.labelcolor': '#9B978E',
    'text.color': '#F0EDE6',
    'xtick.color': '#9B978E',
    'ytick.color': '#9B978E',
    'grid.color': '#2A2A2F',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'font.size': 10,
})

GOLD = '#C9A96E'
GOLD_LIGHT = '#E8D5A8'
BLUE = '#7DAFCB'
RED = '#CB7D7D'
GREEN = '#7DCB8A'
TEXT = '#F0EDE6'


def setup():
    note = AutocallableNote()
    market = MarketData(
        spots=np.array([4500.0, 5200.0, 38000.0]),
        vols=np.array([0.20, 0.18, 0.22]),
        correlation_matrix=np.array([
            [1.0, 0.75, 0.55],
            [0.75, 1.0, 0.50],
            [0.55, 0.50, 1.0]
        ]),
        risk_free_rate=0.04,
        dividend_yields=np.array([0.025, 0.015, 0.02])
    )
    engine = MonteCarloEngine(n_paths=100_000, seed=42)
    return note, market, engine


def plot_payoff_distribution(results, note):
    """Payoff distribution with knock-in and autocall regions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    payoffs = results['payoff_distribution'] / note.notional
    
    bins = np.linspace(payoffs.min(), payoffs.max(), 80)
    n, bins_out, patches = ax.hist(payoffs, bins=bins, density=True, alpha=0.8, color=GOLD, edgecolor='none')
    
    # Color knock-in region
    for i, (b, patch) in enumerate(zip(bins_out, patches)):
        if b < 0.95:
            patch.set_facecolor(RED)
            patch.set_alpha(0.7)
    
    ax.axvline(1.0, color=TEXT, linestyle='--', alpha=0.5, label='Par (100%)')
    ax.axvline(np.mean(payoffs), color=GOLD_LIGHT, linestyle='-', linewidth=2, label=f'Mean: {np.mean(payoffs):.2%}')
    
    ax.set_xlabel('Payoff (% of Notional)')
    ax.set_ylabel('Density')
    ax.set_title('Autocallable Payoff Distribution', fontsize=14, color=TEXT, fontweight='bold')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(facecolor='#1A1A1F', edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig('/home/claude/autocallable/payoff_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: payoff_distribution.png")


def plot_sample_paths(market, note, engine):
    """Sample correlated paths with barrier levels."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    
    # Simulate continuous paths (fewer for viz)
    small_engine = MonteCarloEngine(n_paths=500, seed=42)
    times, paths = small_engine.simulate_paths_continuous(market, note.maturity_years, n_steps=300)
    
    asset_names = ['SX5E', 'SPX', 'NKY']
    
    for i, (ax, name) in enumerate(zip(axes, asset_names)):
        for p in range(min(100, paths.shape[0])):
            ax.plot(times, paths[p, :, i], alpha=0.08, color=GOLD, linewidth=0.5)
        
        # Plot barriers
        ax.axhline(note.autocall_barrier, color=GREEN, linestyle='--', alpha=0.8, label=f'Autocall ({note.autocall_barrier:.0%})')
        ax.axhline(note.coupon_barrier, color=BLUE, linestyle='--', alpha=0.8, label=f'Coupon ({note.coupon_barrier:.0%})')
        ax.axhline(note.knock_in_barrier, color=RED, linestyle='--', alpha=0.8, label=f'Knock-in ({note.knock_in_barrier:.0%})')
        
        # Observation dates
        for t in note.observation_times:
            ax.axvline(t, color='#2A2A2F', linestyle=':', alpha=0.5)
        
        ax.set_title(name, fontsize=12, color=GOLD)
        ax.set_xlabel('Time (years)')
        if i == 0:
            ax.set_ylabel('Performance (S/S0)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0.3, 1.8)
        ax.grid(True, alpha=0.2)
    
    axes[0].legend(loc='upper left', fontsize=8, facecolor='#1A1A1F', edgecolor='#2A2A2F', labelcolor=TEXT)
    
    fig.suptitle('Correlated Asset Paths with Barrier Levels', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('/home/claude/autocallable/sample_paths.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sample_paths.png")


def plot_scenario_analysis(scenarios):
    """Bar chart of scenario prices."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    names = {
        'base': 'Base Case',
        'high_vol': 'High Vol\n(+10%)',
        'low_vol': 'Low Vol\n(-5%)',
        'high_corr': 'High Corr\n(0.90)',
        'low_corr': 'Low Corr\n(0.30)'
    }
    
    labels = [names[k] for k in scenarios.keys()]
    prices = [s['price_pct'] for s in scenarios.values()]
    autocall = [s['autocall_prob'] for s in scenarios.values()]
    ki = [s['knock_in_prob'] for s in scenarios.values()]
    
    x = np.arange(len(labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, prices, width, label='Price (% notional)', color=GOLD, alpha=0.9)
    bars2 = ax.bar(x, autocall, width, label='Autocall probability', color=GREEN, alpha=0.7)
    bars3 = ax.bar(x + width, ki, width, label='Knock-in probability', color=RED, alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel('Value')
    ax.set_title('Scenario Analysis', fontsize=14, color=TEXT, fontweight='bold')
    ax.legend(facecolor='#1A1A1F', edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig('/home/claude/autocallable/scenario_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: scenario_analysis.png")


def plot_correlation_surface(note, market, engine):
    """Price sensitivity to pairwise correlation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    corr_range = np.arange(0.1, 0.96, 0.05)
    prices = []
    ki_probs = []
    
    for rho in corr_range:
        n = market.n_assets
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        
        mkt = MarketData(
            market.spots, market.vols, corr,
            market.risk_free_rate, market.dividend_yields
        )
        paths = engine.simulate_paths(mkt, note.observation_times)
        res = AutocallablePayoff(note, mkt).evaluate(paths)
        prices.append(res['price_pct'])
        ki_probs.append(res['knock_in_prob'])
    
    ax.plot(corr_range, prices, color=GOLD, linewidth=2, label='Price (% notional)')
    ax.set_xlabel('Pairwise Correlation')
    ax.set_ylabel('Price (% notional)', color=GOLD)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis='y', labelcolor=GOLD)
    
    ax2 = ax.twinx()
    ax2.plot(corr_range, ki_probs, color=RED, linewidth=2, linestyle='--', label='Knock-in prob')
    ax2.set_ylabel('Knock-in Probability', color=RED)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.tick_params(axis='y', labelcolor=RED)
    
    ax.set_title('Price & Knock-in Sensitivity to Correlation', fontsize=14, color=TEXT, fontweight='bold')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, facecolor='#1A1A1F', edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig('/home/claude/autocallable/correlation_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_sensitivity.png")


def plot_redemption_profile(results, note):
    """Distribution of early redemption times."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    times = results['redemption_times']
    obs_times = list(note.observation_times)
    
    counts = []
    labels = []
    for t in obs_times:
        count = np.sum(np.abs(times - t) < 0.01)
        counts.append(count / len(times))
        if t < note.maturity_years:
            labels.append(f'{t:.1f}Y')
        else:
            labels.append(f'{t:.1f}Y\n(maturity)')
    
    colors = [GREEN if t < note.maturity_years else BLUE for t in obs_times]
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='none')
    
    ax.set_ylabel('Probability')
    ax.set_xlabel('Redemption Time')
    ax.set_title('Redemption Time Distribution', fontsize=14, color=TEXT, fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar, count in zip(bars, counts):
        if count > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{count:.1%}', ha='center', va='bottom', color=TEXT, fontsize=9)
    
    fig.tight_layout()
    fig.savefig('/home/claude/autocallable/redemption_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: redemption_profile.png")


if __name__ == "__main__":
    note, market, engine = setup()
    
    paths = engine.simulate_paths(market, note.observation_times)
    results = AutocallablePayoff(note, market).evaluate(paths)
    scenarios = run_scenario_analysis(note, market, engine)
    
    print("Generating visualisations...\n")
    plot_payoff_distribution(results, note)
    plot_sample_paths(market, note, engine)
    plot_scenario_analysis(scenarios)
    plot_correlation_surface(note, market, engine)
    plot_redemption_profile(results, note)
    print("\nAll charts generated.")
