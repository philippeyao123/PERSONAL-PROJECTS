"""
FX Options Pricing Engine - Visualisations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from fx_options_pricer import (
    build_smile_market, DupireLocalVol, gk_call, GKGreeks, FXMarket,
    BergomiLSVEngine, BergomiParams, mc_local_vol, barrier_mc
)
from sabr_calibration import SABRCalibrator, sabr_smile, SABRParams
from rr_butterfly import FXSmileConventions, SmileDynamicsAnalyser

GOLD = '#C9A96E'; BLUE = '#7DAFCB'; RED = '#CB7D7D'
GREEN = '#7DCB8A'; PURPLE = '#B07DCB'; BG = '#0C0C0E'
CARD = '#141417'; TEXT = '#F0EDE6'; MUTED = '#9B978E'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': '#2A2A2F',
    'axes.labelcolor': MUTED, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': '#2A2A2F', 'grid.alpha': 0.5, 'font.size': 10,
})

OUT = '/home/claude/fx-options/'


def run():
    S0, rd, rf = 1.25, 0.02, 0.01
    maturities, strikes, call_prices, iv_surface = build_smile_market(S0, rd, rf, skew=-0.03, convexity=0.02)
    dupire = DupireLocalVol(maturities, strikes, call_prices, rd, rf)
    return S0, rd, rf, maturities, strikes, iv_surface, dupire


def plot_iv_surface(S0, maturities, strikes, iv_surface):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(CARD)
    
    T_grid, K_grid = np.meshgrid(maturities, strikes, indexing='ij')
    surf = ax.plot_surface(T_grid, K_grid, iv_surface * 100, cmap='YlOrRd', alpha=0.8, edgecolor='none')
    
    ax.set_xlabel('Maturity (Y)')
    ax.set_ylabel('Strike')
    ax.set_zlabel('IV (%)')
    ax.set_title('FX Implied Volatility Surface (EUR/USD)', color=TEXT, fontweight='bold', fontsize=13)
    
    fig.tight_layout()
    fig.savefig(OUT + 'iv_surface.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: iv_surface.png")


def plot_smile_comparison(S0, rd, rf, maturities, strikes, iv_surface):
    """Compare SABR calibrated smile with market."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Smile at different maturities
    ax = axes[0]
    for i, T in enumerate(maturities):
        color = [GOLD, BLUE, GREEN, RED][i]
        ax.plot(strikes, iv_surface[i] * 100, 'o-', color=color, label=f'T={T}Y', markersize=4)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied Vol (%)')
    ax.set_title('FX Smile by Maturity', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    # SABR fit for 1Y
    ax = axes[1]
    T = 1.0
    F = S0 * np.exp((rd - rf) * T)
    market_ivs = iv_surface[2]
    
    cal = SABRCalibrator(beta=0.50)
    params = cal.calibrate(F, strikes, T, market_ivs)
    
    dense_K = np.linspace(strikes[0], strikes[-1], 80)
    sabr_ivs = sabr_smile(F, dense_K, T, params)
    
    ax.plot(strikes, market_ivs * 100, 'o', color=GOLD, markersize=8, label='Market')
    ax.plot(dense_K, sabr_ivs * 100, '-', color=BLUE, linewidth=2, label=f'SABR (ρ={params.rho:+.2f}, ν={params.nu:.2f})')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied Vol (%)')
    ax.set_title('SABR Calibration (1Y)', color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT + 'smile_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: smile_comparison.png")


def plot_greeks_surface(S0, rd, rf):
    """Greeks as a function of spot and strike."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    spots = np.linspace(S0 * 0.8, S0 * 1.2, 50)
    T = 1.0
    sigma = 0.20
    K = S0
    greeks = GKGreeks()
    
    deltas = [greeks.delta(S, K, T, rd, rf, sigma) for S in spots]
    gammas = [greeks.gamma(S, K, T, rd, rf, sigma) for S in spots]
    vegas = [greeks.vega(S, K, T, rd, rf, sigma) for S in spots]
    thetas = [greeks.theta(S, K, T, rd, rf, sigma) for S in spots]
    
    for ax, values, name, color in [
        (axes[0,0], deltas, 'Delta', GOLD),
        (axes[0,1], gammas, 'Gamma', BLUE),
        (axes[1,0], vegas, 'Vega', GREEN),
        (axes[1,1], thetas, 'Theta', RED),
    ]:
        ax.plot(spots, values, color=color, linewidth=2)
        ax.axvline(K, color=MUTED, linestyle='--', alpha=0.5, label=f'K={K}')
        ax.set_xlabel('Spot')
        ax.set_ylabel(name)
        ax.set_title(name, color=TEXT, fontweight='bold')
        ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT, fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('GK Greeks vs Spot (1Y ATM Call)', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'greeks_surface.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: greeks_surface.png")


def plot_rr_bf_term_structure(S0, rd, rf, maturities, strikes, iv_surface):
    """RR and BF term structure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    ts = SmileDynamicsAnalyser.term_structure_rr_bf(S0, rd, rf, maturities, strikes, iv_surface)
    
    ax = axes[0]
    ax.plot(maturities, ts['atm'] * 100, color=GOLD, linewidth=2, marker='o')
    ax.set_xlabel('Maturity')
    ax.set_ylabel('ATM Vol (%)')
    ax.set_title('ATM Vol Term Structure', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(maturities, ts['rr_25d'] * 100, color=BLUE, linewidth=2, marker='o')
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_xlabel('Maturity')
    ax.set_ylabel('25Δ RR (%)')
    ax.set_title('Risk Reversal Term Structure', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(maturities, ts['bf_25d'] * 100, color=GREEN, linewidth=2, marker='o')
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_xlabel('Maturity')
    ax.set_ylabel('25Δ BF (%)')
    ax.set_title('Butterfly Term Structure', color=TEXT, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('FX Smile Conventions', fontsize=14, color=TEXT, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'rr_bf_term.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: rr_bf_term.png")


def plot_local_vol_surface(S0, maturities, strikes, dupire):
    """Local vol surface plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, T in enumerate(maturities):
        color = [GOLD, BLUE, GREEN, RED][i]
        lv = [dupire.interp(np.array([[T, K]])).item() for K in strikes]
        ax.plot(strikes, np.array(lv) * 100, 'o-', color=color, label=f'T={T}Y', linewidth=2, markersize=5)
    
    ax.set_xlabel('Strike')
    ax.set_ylabel('Local Vol (%)')
    ax.set_title('Dupire Local Volatility Surface', fontsize=14, color=TEXT, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT + 'local_vol_surface.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: local_vol_surface.png")


if __name__ == "__main__":
    print("Generating FX visualisations...\n")
    
    S0, rd, rf, maturities, strikes, iv_surface, dupire = run()
    
    plot_iv_surface(S0, maturities, strikes, iv_surface)
    plot_smile_comparison(S0, rd, rf, maturities, strikes, iv_surface)
    plot_greeks_surface(S0, rd, rf)
    plot_rr_bf_term_structure(S0, rd, rf, maturities, strikes, iv_surface)
    plot_local_vol_surface(S0, maturities, strikes, dupire)
    
    print("\nAll charts generated.")
