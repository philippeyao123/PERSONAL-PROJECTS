"""
Merton Credit Model - Visualisations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import norm
from merton_credit import (
    MertonModel, KMVModel, BlackCoxModel, CreditGradesModel,
    PortfolioCreditModel, CDSCalibrator, create_sample_firms
)

GOLD = '#C9A96E'; GOLD_LIGHT = '#E8D5A8'; BLUE = '#7DAFCB'
RED = '#CB7D7D'; GREEN = '#7DCB8A'; PURPLE = '#B07DCB'
BG = '#0C0C0E'; CARD = '#141417'; TEXT = '#F0EDE6'; MUTED = '#9B978E'
COLORS = [GOLD, BLUE, GREEN, RED, PURPLE, '#CB9E7D']

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': '#2A2A2F',
    'axes.labelcolor': MUTED, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': '#2A2A2F', 'grid.alpha': 0.5, 'font.size': 10,
})

OUT = '/home/claude/merton-credit/'


def run_base():
    merton = MertonModel()
    firms = create_sample_firms()
    results = {}
    for firm in firms:
        results[firm.name] = merton.calibrate(firm)
    return merton, firms, results


def plot_dd_vs_spread(merton, firms, results):
    """Distance to Default vs Credit Spread."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, firm in enumerate(firms):
        r = results[firm.name]
        dd = r['distance_to_default']
        cds = r['cds_implied_bps']
        ax.scatter(dd, cds, color=COLORS[i], s=120, zorder=5)
        ax.annotate(firm.name.split(' ')[0], (dd, cds),
                   textcoords="offset points", xytext=(8, 8),
                   color=COLORS[i], fontsize=9)
    
    # Theoretical curve
    dd_range = np.linspace(0.5, 8, 100)
    pd_range = norm.cdf(-dd_range)
    cds_range = pd_range * 0.60 * 10000  # LGD = 60%
    ax.plot(dd_range, cds_range, color=MUTED, linewidth=1.5, linestyle='--', alpha=0.5, label='Theoretical (LGD=60%)')
    
    ax.set_xlabel('Distance to Default')
    ax.set_ylabel('Implied CDS Spread (bps)')
    ax.set_title('Distance to Default vs Credit Spread', fontsize=14, color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT + 'dd_vs_spread.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: dd_vs_spread.png")


def plot_term_structure(merton, firms, results):
    """CDS term structure for selected firms."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    maturities = np.linspace(0.5, 10, 40)
    selected = [firms[2], firms[5]]  # Ford, EM Telecom
    
    # PD term structure
    ax = axes[0]
    for i, firm in enumerate(selected):
        r = results[firm.name]
        pds = merton.term_structure_pd(r['firm_value'], firm.total_debt,
                                        firm.risk_free_rate, r['firm_vol'], maturities)
        ax.plot(maturities, pds * 100, color=COLORS[i+2], linewidth=2, label=firm.name)
        
        # Black-Cox comparison
        bc = BlackCoxModel()
        pds_bc = bc.term_structure(r['firm_value'], firm.total_debt,
                                    firm.risk_free_rate, r['firm_vol'], maturities)
        ax.plot(maturities, pds_bc * 100, color=COLORS[i+2], linewidth=1.5, linestyle='--', alpha=0.6)
    
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('Default Probability (%)')
    ax.set_title('PD Term Structure\n(solid=Merton, dashed=Black-Cox)', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    # CDS term structure
    ax = axes[1]
    for i, firm in enumerate(selected):
        r = results[firm.name]
        spreads = merton.term_structure_cds(r['firm_value'], firm.total_debt,
                                             firm.risk_free_rate, r['firm_vol'], firm.lgd, maturities)
        ax.plot(maturities, spreads, color=COLORS[i+2], linewidth=2, label=firm.name)
    
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('CDS Spread (bps)')
    ax.set_title('CDS Spread Term Structure', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Credit Curve Analysis', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'term_structure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: term_structure.png")


def plot_model_comparison(firms, results):
    """Compare Merton, Black-Cox, CreditGrades CDS spreads vs market."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    names = [f.name.split(' ')[0] for f in firms]
    x = np.arange(len(firms))
    w = 0.2
    
    merton_cds = [results[f.name]['cds_implied_bps'] for f in firms]
    
    bc = BlackCoxModel()
    bc_cds = []
    for f in firms:
        r = results[f.name]
        pd_bc = bc.default_probability(r['firm_value'], f.total_debt, f.risk_free_rate, r['firm_vol'], 5.0)
        h = -np.log(1-pd_bc)/5 if pd_bc < 1 else 1
        bc_cds.append(h * f.lgd * 10000)
    
    cg = CreditGradesModel(0.3)
    cg_cds = []
    for f in firms:
        r = results[f.name]
        cg_cds.append(cg.cds_spread(r['firm_value'], f.total_debt, f.lgd, r['firm_vol'], f.risk_free_rate, 5.0))
    
    market_cds = [f.cds_spread_5y * 10000 if f.cds_spread_5y else 0 for f in firms]
    
    ax.bar(x - 1.5*w, merton_cds, w, color=GOLD, alpha=0.8, label='Merton')
    ax.bar(x - 0.5*w, bc_cds, w, color=BLUE, alpha=0.8, label='Black-Cox')
    ax.bar(x + 0.5*w, cg_cds, w, color=GREEN, alpha=0.8, label='CreditGrades')
    ax.bar(x + 1.5*w, market_cds, w, color=RED, alpha=0.8, label='Market CDS')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('5Y CDS Spread (bps)')
    ax.set_title('Model vs Market CDS Spreads', fontsize=14, color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT + 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison.png")


def plot_portfolio_loss(firms, results):
    """Portfolio loss distribution under different correlations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    pds = np.array([results[f.name]['pd_risk_neutral'] for f in firms])
    lgds = np.array([f.lgd for f in firms])
    exposures = np.array([10, 50, 20, 15, 30, 10], dtype=float)
    
    portfolio = PortfolioCreditModel(seed=42)
    
    # Loss distributions
    ax = axes[0]
    for corr, color, label in [(0.10, GREEN, 'ρ=10%'), (0.20, GOLD, 'ρ=20%'), (0.40, RED, 'ρ=40%')]:
        result = portfolio.simulate_defaults(pds, lgds, exposures, corr, 100_000)
        losses = result['loss_distribution']
        losses_pos = losses[losses > 0]
        if len(losses_pos) > 10:
            ax.hist(losses_pos, bins=50, density=True, alpha=0.4, color=color, label=label, edgecolor='none')
    
    ax.set_xlabel('Portfolio Loss ($M)')
    ax.set_ylabel('Density')
    ax.set_title('Loss Distribution by Correlation', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    # VaR/CVaR comparison
    ax = axes[1]
    corrs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    vars_99 = []
    cvars_99 = []
    
    for c in corrs:
        result = portfolio.simulate_defaults(pds, lgds, exposures, c, 50_000)
        vars_99.append(result['var_99'])
        cvars_99.append(result['cvar_99'])
    
    ax.plot(corrs, vars_99, color=GOLD, linewidth=2, marker='o', label='VaR 99%')
    ax.plot(corrs, cvars_99, color=RED, linewidth=2, marker='s', label='CVaR 99%')
    ax.set_xlabel('Asset Correlation')
    ax.set_ylabel('Loss ($M)')
    ax.set_title('Tail Risk vs Correlation', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Portfolio Credit Risk', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'portfolio_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: portfolio_loss.png")


def plot_merton_payoff():
    """Merton model intuition: equity as call option on firm value."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    D = 100  # Debt
    V = np.linspace(0, 250, 500)
    
    equity = np.maximum(V - D, 0)
    debt = np.minimum(V, D)
    
    ax.plot(V, equity, color=GOLD, linewidth=2.5, label='Equity = max(V-D, 0)')
    ax.plot(V, debt, color=BLUE, linewidth=2.5, label='Debt = min(V, D)')
    ax.axvline(D, color=RED, linestyle='--', alpha=0.7, label=f'Default barrier D={D}')
    
    # Fill default region
    ax.fill_between(V, 0, equity, where=V < D, alpha=0.1, color=RED)
    ax.annotate('DEFAULT\nREGION', xy=(50, 20), fontsize=11, color=RED, ha='center', fontweight='bold')
    ax.annotate('EQUITY\nVALUE', xy=(180, 50), fontsize=11, color=GOLD, ha='center', fontweight='bold')
    
    ax.set_xlabel('Firm Value (V)')
    ax.set_ylabel('Payoff')
    ax.set_title('Merton Model: Equity = Call Option on Firm Value', fontsize=14, color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT + 'merton_payoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: merton_payoff.png")


if __name__ == "__main__":
    print("Generating visualisations...\n")
    
    merton, firms, results = run_base()
    
    plot_merton_payoff()
    plot_dd_vs_spread(merton, firms, results)
    plot_term_structure(merton, firms, results)
    plot_model_comparison(firms, results)
    plot_portfolio_loss(firms, results)
    
    print("\nAll charts generated.")
