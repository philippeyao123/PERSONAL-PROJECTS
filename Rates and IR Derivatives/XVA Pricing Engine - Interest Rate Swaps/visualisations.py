"""
XVA Pricing Engine - Visualisations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from xva_pricer import (
    IRSwap, HullWhiteParams, HullWhiteModel, ExposureEngine,
    CreditCurve, XVACalculator, SIMMCalculator, CapitalCalculator,
    xva_sensitivity_analysis
)

GOLD = '#C9A96E'; GOLD_LIGHT = '#E8D5A8'; BLUE = '#7DAFCB'
RED = '#CB7D7D'; GREEN = '#7DCB8A'; BG = '#0C0C0E'; CARD = '#141417'
TEXT = '#F0EDE6'; MUTED = '#9B978E'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': '#2A2A2F',
    'axes.labelcolor': MUTED, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': '#2A2A2F', 'grid.alpha': 0.5, 'font.size': 10,
})

OUT = '/home/claude/xva-pricer/'


def run_base_case():
    swap = IRSwap()
    hw = HullWhiteModel(HullWhiteParams())
    time_grid = np.linspace(0, swap.maturity_years, 120)
    rates = hw.simulate_rates(50_000, time_grid, seed=42)
    
    exp_engine = ExposureEngine(hw, swap)
    exposures = exp_engine.compute_profiles(rates, time_grid)
    
    simm_calc = SIMMCalculator()
    dv01 = simm_calc.compute_dv01_profile(hw, swap, rates, time_grid)
    simm = simm_calc.compute_simm_profile(dv01, time_grid, swap.maturity_years)
    
    cap_calc = CapitalCalculator()
    capital = cap_calc.compute_capital_profile(exposures['ee'], time_grid, swap)
    
    cpty = CreditCurve("Cpty", 0.005, 0.40)
    own = CreditCurve("Own", 0.003, 0.40)
    xva_calc = XVACalculator(cpty, own)
    xva = xva_calc.compute_all(exposures, time_grid, simm, capital)
    
    sens = xva_sensitivity_analysis(swap, HullWhiteParams(), n_paths=30_000)
    
    return swap, time_grid, rates, exposures, xva, simm, capital, sens


def plot_exposure_profiles(time_grid, exposures):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.fill_between(time_grid, exposures['ee'], alpha=0.2, color=GOLD)
    ax.plot(time_grid, exposures['ee'], color=GOLD, linewidth=2, label='EE (Expected Exposure)')
    ax.fill_between(time_grid, exposures['ene'], alpha=0.2, color=RED)
    ax.plot(time_grid, exposures['ene'], color=RED, linewidth=2, label='ENE (Expected Neg Exposure)')
    ax.plot(time_grid, exposures['pfe_97_5'], color=BLUE, linewidth=1.5, linestyle='--', label='PFE 97.5%')
    ax.plot(time_grid, exposures['pfe_99'], color=BLUE, linewidth=1, linestyle=':', label='PFE 99%', alpha=0.7)
    ax.plot(time_grid, exposures['eepe'], color=GREEN, linewidth=1.5, linestyle='-.', label='Effective EE')
    
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Exposure ($)')
    ax.set_title('Counterparty Exposure Profiles — 10Y Payer IRS', fontsize=14, color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT + 'exposure_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: exposure_profiles.png")


def plot_xva_waterfall(xva, notional):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    components = ['CVA', 'DVA', 'FVA', 'KVA', 'MVA', 'Total XVA']
    values_bps = [
        xva['cva'] / notional * 10000,
        -xva['dva'] / notional * 10000,  # DVA is a benefit
        xva['fva'] / notional * 10000,
        xva['kva'] / notional * 10000,
        xva['mva'] / notional * 10000,
        xva['total_xva'] / notional * 10000
    ]
    
    colors = [RED, GREEN, BLUE, GOLD, '#B07DCB', TEXT]
    
    bars = ax.bar(components, values_bps, color=colors, alpha=0.8, edgecolor='none')
    
    for bar, val in zip(bars, values_bps):
        ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 0.8
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
               f'{val:+.1f}', ha='center', va='bottom' if val >= 0 else 'top',
               color=TEXT, fontsize=11, fontweight='bold')
    
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_ylabel('bps of Notional')
    ax.set_title('XVA Waterfall — 10Y Payer IRS (10M Notional)', fontsize=14, color=TEXT, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT + 'xva_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: xva_waterfall.png")


def plot_simm_capital(time_grid, simm, capital):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.fill_between(time_grid, simm, alpha=0.3, color=GOLD)
    ax.plot(time_grid, simm, color=GOLD, linewidth=2)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('SIMM IM ($)')
    ax.set_title('ISDA SIMM Initial Margin', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.fill_between(time_grid, capital, alpha=0.3, color=RED)
    ax.plot(time_grid, capital, color=RED, linewidth=2)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Capital ($)')
    ax.set_title('SA-CCR Regulatory Capital', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Margin & Capital Profiles', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'simm_capital.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: simm_capital.png")


def plot_cva_sensitivity(sens, notional):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # CVA vs CDS spread
    ax = axes[0]
    spreads = list(sens['cva_by_spread'].keys())
    cvas = [sens['cva_by_spread'][s] / notional * 10000 for s in spreads]
    spread_bps = [s * 10000 for s in spreads]
    ax.plot(spread_bps, cvas, color=GOLD, linewidth=2, marker='o', markersize=6)
    ax.set_xlabel('CDS Spread (bps)')
    ax.set_ylabel('CVA (bps of notional)')
    ax.set_title('CVA vs Counterparty Credit', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # FVA vs funding spread
    ax = axes[1]
    fspreads = list(sens['fva_by_funding'].keys())
    fvas = [sens['fva_by_funding'][s] / notional * 10000 for s in fspreads]
    fs_bps = [s * 10000 for s in fspreads]
    ax.plot(fs_bps, fvas, color=BLUE, linewidth=2, marker='o', markersize=6)
    ax.set_xlabel('Funding Spread (bps)')
    ax.set_ylabel('FVA (bps of notional)')
    ax.set_title('FVA vs Funding Cost', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('XVA Sensitivity Analysis', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'xva_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: xva_sensitivity.png")


def plot_rate_paths_and_mtm(time_grid, rates, exposures):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 1.2]})
    
    # Rate paths
    ax = axes[0]
    for i in range(min(200, rates.shape[0])):
        ax.plot(time_grid, rates[i, :] * 100, alpha=0.03, color=GOLD, linewidth=0.5)
    ax.plot(time_grid, np.mean(rates, axis=0) * 100, color=GOLD_LIGHT, linewidth=2, label='Mean rate')
    ax.plot(time_grid, np.percentile(rates, 5, axis=0) * 100, color=RED, linewidth=1, linestyle='--', label='5th/95th pctl')
    ax.plot(time_grid, np.percentile(rates, 95, axis=0) * 100, color=RED, linewidth=1, linestyle='--')
    ax.set_ylabel('Short Rate (%)')
    ax.set_title('Hull-White Short Rate Paths', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # MtM distribution over time
    ax = axes[1]
    mtm = exposures['mtm']
    ax.fill_between(time_grid, np.percentile(mtm, 5, axis=0), np.percentile(mtm, 95, axis=0),
                    alpha=0.2, color=GOLD, label='5-95th percentile')
    ax.fill_between(time_grid, np.percentile(mtm, 25, axis=0), np.percentile(mtm, 75, axis=0),
                    alpha=0.3, color=GOLD, label='25-75th percentile')
    ax.plot(time_grid, np.mean(mtm, axis=0), color=GOLD_LIGHT, linewidth=2, label='Mean MtM')
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('MtM ($)')
    ax.set_title('IRS Mark-to-Market Distribution', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT + 'rates_and_mtm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: rates_and_mtm.png")


if __name__ == "__main__":
    print("Generating XVA visualisations...\n")
    
    swap, time_grid, rates, exposures, xva, simm, capital, sens = run_base_case()
    
    plot_exposure_profiles(time_grid, exposures)
    plot_xva_waterfall(xva, swap.notional)
    plot_simm_capital(time_grid, simm, capital)
    plot_cva_sensitivity(sens, swap.notional)
    plot_rate_paths_and_mtm(time_grid, rates, exposures)
    
    print("\nAll charts generated.")
