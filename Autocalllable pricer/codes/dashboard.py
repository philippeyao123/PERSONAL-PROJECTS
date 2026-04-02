"""
Autocallable Worst-of Pricing Dashboard
=========================================
Interactive Streamlit dashboard integrating all pricing extensions.

Run with: streamlit run dashboard.py

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time

from autocallable_pricer import (
    AutocallableNote, MarketData, MonteCarloEngine,
    AutocallablePayoff, GreeksCalculator, run_scenario_analysis
)
from heston_model import (
    HestonParams, HestonMarketData, HestonMonteCarloEngine
)
from brownian_bridge import compare_discrete_vs_continuous
from term_structure import (
    Curve, TermStructureMarket, TermStructureMCEngine
)
from cva_adjustment import CreditParams, CVACalculator

# ── Page config ──
st.set_page_config(
    page_title="Autocallable Pricer",
    page_icon="📊",
    layout="wide"
)

# ── Styling ──
GOLD = '#C9A96E'
BLUE = '#7DAFCB'
RED = '#CB7D7D'
GREEN = '#7DCB8A'
BG = '#0C0C0E'
CARD = '#141417'
TEXT = '#F0EDE6'
MUTED = '#9B978E'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': CARD,
    'axes.edgecolor': '#2A2A2F',
    'axes.labelcolor': MUTED,
    'text.color': TEXT,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'grid.color': '#2A2A2F',
    'grid.alpha': 0.5,
    'font.size': 10,
})

st.markdown("""
<style>
    .stApp { background-color: #0C0C0E; }
    .stMetric label { color: #9B978E !important; }
    .stMetric [data-testid="stMetricValue"] { color: #C9A96E !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ──
st.title("📊 Autocallable Worst-of Pricing Engine")
st.markdown("*Monte Carlo pricer with Heston SV, Brownian bridge, term structures & CVA*")
st.markdown("---")

# ══════════════════════════════════════════════════
# SIDEBAR: Parameters
# ══════════════════════════════════════════════════

with st.sidebar:
    st.header("Product Parameters")
    
    notional = st.number_input("Notional", value=1_000_000, step=100_000)
    maturity = st.slider("Maturity (years)", 1.0, 5.0, 3.0, 0.5)
    n_assets = st.selectbox("Number of assets", [2, 3, 4], index=1)
    
    st.subheader("Barriers")
    autocall_barrier = st.slider("Autocall barrier", 0.80, 1.20, 1.00, 0.05)
    coupon_barrier = st.slider("Coupon barrier", 0.50, 0.90, 0.70, 0.05)
    knock_in_barrier = st.slider("Knock-in barrier", 0.40, 0.80, 0.60, 0.05)
    
    st.subheader("Coupon")
    coupon_rate = st.slider("Annual coupon rate", 0.04, 0.15, 0.08, 0.01)
    coupon_freq = st.selectbox("Frequency", [2, 4], index=0, format_func=lambda x: "Semi-annual" if x == 2 else "Quarterly")
    memory = st.checkbox("Memory coupon", value=True)
    
    st.subheader("Market Data")
    vols = []
    for i in range(n_assets):
        v = st.slider(f"Vol asset {i+1}", 0.10, 0.50, [0.20, 0.18, 0.22, 0.25][i], 0.01)
        vols.append(v)
    
    r_flat = st.slider("Risk-free rate", 0.00, 0.08, 0.04, 0.005)
    corr_level = st.slider("Avg correlation", 0.10, 0.95, 0.60, 0.05)
    
    st.subheader("Simulation")
    n_paths = st.select_slider("Paths", [10_000, 50_000, 100_000, 200_000], value=100_000)
    model_choice = st.selectbox("Model", ["GBM (Black-Scholes)", "Heston Stochastic Vol"])
    
    st.subheader("CVA Parameters")
    cds_spread = st.slider("CDS spread (bps)", 10, 500, 50, 10) / 10000
    recovery = st.slider("Recovery rate", 0.20, 0.60, 0.40, 0.05)
    
    run_button = st.button("🚀 Run Pricing", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════
# BUILD OBJECTS
# ══════════════════════════════════════════════════

note = AutocallableNote(
    notional=notional,
    maturity_years=maturity,
    n_assets=n_assets,
    autocall_barrier=autocall_barrier,
    coupon_barrier=coupon_barrier,
    knock_in_barrier=knock_in_barrier,
    coupon_rate=coupon_rate,
    coupon_frequency=coupon_freq,
    memory_coupon=memory
)

# Build correlation matrix
corr = np.full((n_assets, n_assets), corr_level)
np.fill_diagonal(corr, 1.0)

spots = np.array([4500, 5200, 38000, 3200][:n_assets], dtype=float)
vols_arr = np.array(vols[:n_assets])
divs = np.array([0.025, 0.015, 0.02, 0.018][:n_assets])

market = MarketData(
    spots=spots, vols=vols_arr, correlation_matrix=corr,
    risk_free_rate=r_flat, dividend_yields=divs
)


# ══════════════════════════════════════════════════
# PRICING
# ══════════════════════════════════════════════════

if run_button:
    
    # ── Monte Carlo ──
    with st.spinner("Running Monte Carlo simulation..."):
        t0 = time.time()
        
        if model_choice == "GBM (Black-Scholes)":
            engine = MonteCarloEngine(n_paths=n_paths, seed=42)
            paths = engine.simulate_paths(market, note.observation_times)
            model_label = "GBM"
        else:
            heston_params = [
                HestonParams(v0=v**2, kappa=2.0, theta=v**2, xi=0.3, rho_sv=-0.7)
                for v in vols_arr
            ]
            heston_market = HestonMarketData(
                spots=spots, heston_params=heston_params,
                spot_correlation=corr, risk_free_rate=r_flat,
                dividend_yields=divs
            )
            heston_engine = HestonMonteCarloEngine(n_paths=n_paths, seed=42)
            paths, var_paths = heston_engine.simulate_paths(heston_market, note.observation_times)
            engine = MonteCarloEngine(n_paths=n_paths, seed=42)  # For Greeks
            model_label = "Heston"
        
        results = AutocallablePayoff(note, market).evaluate(paths)
        t_total = time.time() - t0
    
    # ── RESULTS DISPLAY ──
    st.header("Pricing Results")
    st.caption(f"Model: {model_label} | {n_paths:,} paths | {t_total:.1f}s")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Price", f"{results['price_pct']:.2%}")
    col2.metric("Autocall Prob", f"{results['autocall_prob']:.1%}")
    col3.metric("Knock-in Prob", f"{results['knock_in_prob']:.1%}")
    col4.metric("Avg Redemption", f"{results['avg_redemption_time']:.1f}Y")
    col5.metric("Avg Coupon", f"{results['avg_coupon']:.1%}")
    
    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Payoff Analysis",
        "📈 Greeks",
        "🌉 Barrier Correction",
        "💳 CVA",
        "🔬 Scenarios"
    ])
    
    # ── TAB 1: Payoff ──
    with tab1:
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig, ax = plt.subplots(figsize=(8, 4))
            payoffs = results['payoff_distribution'] / note.notional
            bins = np.linspace(payoffs.min(), payoffs.max(), 60)
            n_hist, bins_out, patches = ax.hist(payoffs, bins=bins, density=True, alpha=0.8, color=GOLD, edgecolor='none')
            for b, patch in zip(bins_out, patches):
                if b < 0.95:
                    patch.set_facecolor(RED)
                    patch.set_alpha(0.7)
            ax.axvline(np.mean(payoffs), color=GOLD, linestyle='-', linewidth=2, label=f'Mean: {np.mean(payoffs):.2%}')
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_title('Payoff Distribution', color=TEXT, fontweight='bold')
            ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col_b:
            fig, ax = plt.subplots(figsize=(8, 4))
            times = results['redemption_times']
            obs_times = list(note.observation_times)
            counts = []
            labels = []
            for t in obs_times:
                counts.append(np.sum(np.abs(times - t) < 0.01) / len(times))
                labels.append(f'{t:.1f}Y' if t < note.maturity_years else f'{t:.1f}Y\n(mat)')
            colors = [GREEN if t < note.maturity_years else BLUE for t in obs_times]
            ax.bar(labels, counts, color=colors, alpha=0.8)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_title('Redemption Profile', color=TEXT, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Stats table
        st.subheader("Distribution Statistics")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        stats_col1.metric("5th Percentile", f"{np.percentile(payoffs, 5):.1%}")
        stats_col2.metric("Median", f"{np.median(payoffs):.1%}")
        stats_col3.metric("Std Dev", f"{np.std(payoffs):.1%}")
        stats_col4.metric("Max Loss", f"{(np.min(payoffs) - 1):.1%}")
    
    # ── TAB 2: Greeks ──
    with tab2:
        with st.spinner("Computing Greeks (bump-and-revalue)..."):
            if model_choice == "GBM (Black-Scholes)":
                greeks_calc = GreeksCalculator(engine)
                greeks = greeks_calc.compute_greeks(note, market)
            else:
                greeks = {'base_price': results['price']}
                st.info("Greeks computed via GBM bump-and-revalue (Heston Greeks require adjoint methods)")
                greeks_calc = GreeksCalculator(MonteCarloEngine(n_paths=n_paths, seed=42))
                greeks = greeks_calc.compute_greeks(note, market)
        
        gcols = st.columns(n_assets + 3)
        for i in range(n_assets):
            gcols[i].metric(f"Delta (Asset {i+1})", f"{greeks[f'delta_asset_{i+1}']:,.0f}")
        gcols[n_assets].metric("Vega (1%)", f"{greeks['vega']:,.0f}")
        gcols[n_assets + 1].metric("Rho (10bp)", f"{greeks['rho']:,.0f}")
        if 'corr_sensitivity' in greeks:
            gcols[n_assets + 2].metric("Corr Sens (5%)", f"{greeks.get('corr_sensitivity', 0):,.0f}")
        
        st.markdown("""
        **Interpretation:**
        - **Negative Vega**: The investor is short volatility. Higher vol increases knock-in probability.
        - **Positive Correlation Sensitivity**: Higher correlation makes the worst-of less dispersed, reducing knock-in risk.
        """)
    
    # ── TAB 3: Brownian Bridge ──
    with tab3:
        st.subheader("Continuous vs Discrete Barrier Monitoring")
        
        comparison = compare_discrete_vs_continuous(
            paths, note.knock_in_barrier, vols_arr, note.observation_times
        )
        
        bcol1, bcol2, bcol3 = st.columns(3)
        bcol1.metric("Discrete KI", f"{comparison['ki_prob_discrete']:.2%}")
        bcol2.metric("Continuous KI", f"{comparison['ki_prob_continuous']:.2%}")
        bcol3.metric("Correction", f"{comparison['correction_factor']:.2f}x")
        
        st.markdown(f"""
        The Brownian bridge correction reveals that discrete monitoring **underestimates** 
        the knock-in probability by **{(comparison['correction_factor']-1)*100:.0f}%**.
        
        With semi-annual observations, there are long periods between monitoring dates where 
        the worst-of can breach the {note.knock_in_barrier:.0%} barrier without being detected.
        The correction adds **{comparison['additional_ki_paths']:,}** additional knock-in paths 
        ({comparison['additional_ki_pct']:.2%} of total).
        
        **Impact on pricing**: The corrected price would be lower, as more paths experience 
        capital loss at maturity.
        """)
    
    # ── TAB 4: CVA ──
    with tab4:
        st.subheader("Credit Valuation Adjustment")
        
        credit = CreditParams(cds_spread=cds_spread, recovery_rate=recovery)
        cva_calc = CVACalculator(credit, market.risk_free_rate)
        
        cva_results = cva_calc.compute_cva(
            results['payoff_distribution'],
            results['redemption_times'],
            note.notional
        )
        
        ccol1, ccol2, ccol3, ccol4 = st.columns(4)
        ccol1.metric("Risk-free Price", f"{cva_results['risk_free_price']/note.notional:.2%}")
        ccol2.metric("CVA", f"{cva_results['cva_bps']:.1f} bps")
        ccol3.metric("CVA-adjusted Price", f"{cva_results['cva_adjusted_price']/note.notional:.2%}")
        ccol4.metric("Default Prob", f"{cva_results['default_prob']:.2%}")
        
        # CVA by credit quality
        st.subheader("CVA by Credit Quality")
        quality = cva_calc.cva_by_credit_quality(
            results['payoff_distribution'],
            results['redemption_times'],
            note.notional
        )
        
        fig, ax = plt.subplots(figsize=(10, 4))
        names = list(quality.keys())
        cva_vals = [quality[n]['cva_bps'] for n in names]
        colors_bar = [GREEN, GREEN, GOLD, GOLD, RED, RED]
        ax.bar(names, cva_vals, color=colors_bar[:len(names)], alpha=0.8)
        ax.set_ylabel('CVA (bps)')
        ax.set_title('CVA by Issuer Credit Quality', color=TEXT, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=15)
        st.pyplot(fig)
        plt.close()
        
        # EPE profile
        st.subheader("Expected Positive Exposure Profile")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(cva_results['epe_times'], cva_results['epe_profile'], alpha=0.3, color=GOLD)
        ax.plot(cva_results['epe_times'], cva_results['epe_profile'], color=GOLD, linewidth=2)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('EPE')
        ax.set_title('Expected Positive Exposure', color=TEXT, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # ── TAB 5: Scenarios ──
    with tab5:
        with st.spinner("Running scenario analysis..."):
            scenarios = run_scenario_analysis(note, market, engine)
        
        names_map = {
            'base': 'Base Case',
            'high_vol': 'High Vol (+10%)',
            'low_vol': 'Low Vol (-5%)',
            'high_corr': 'High Corr (0.90)',
            'low_corr': 'Low Corr (0.30)'
        }
        
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [names_map[k] for k in scenarios.keys()]
        prices = [s['price_pct'] for s in scenarios.values()]
        autocall = [s['autocall_prob'] for s in scenarios.values()]
        ki = [s['knock_in_prob'] for s in scenarios.values()]
        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, prices, w, label='Price', color=GOLD, alpha=0.9)
        ax.bar(x, autocall, w, label='Autocall %', color=GREEN, alpha=0.7)
        ax.bar(x + w, ki, w, label='Knock-in %', color=RED, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title('Scenario Analysis', color=TEXT, fontweight='bold')
        ax.legend(facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
        ax.grid(True, axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Correlation scan
        st.subheader("Correlation Sensitivity Scan")
        with st.spinner("Scanning correlation..."):
            corr_range = np.arange(0.15, 0.96, 0.1)
            corr_prices = []
            corr_ki = []
            small_engine = MonteCarloEngine(n_paths=min(n_paths, 50_000), seed=42)
            for rho in corr_range:
                c = np.full((n_assets, n_assets), rho)
                np.fill_diagonal(c, 1.0)
                mkt = MarketData(spots, vols_arr, c, r_flat, divs)
                p = small_engine.simulate_paths(mkt, note.observation_times)
                r = AutocallablePayoff(note, mkt).evaluate(p)
                corr_prices.append(r['price_pct'])
                corr_ki.append(r['knock_in_prob'])
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(corr_range, corr_prices, color=GOLD, linewidth=2, label='Price')
        ax.set_ylabel('Price (% notional)', color=GOLD)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2 = ax.twinx()
        ax2.plot(corr_range, corr_ki, color=RED, linewidth=2, linestyle='--', label='KI Prob')
        ax2.set_ylabel('Knock-in Prob', color=RED)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xlabel('Pairwise Correlation')
        ax.set_title('Price & KI vs Correlation', color=TEXT, fontweight='bold')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, facecolor=CARD, edgecolor='#2A2A2F', labelcolor=TEXT)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

else:
    st.info("👈 Configure parameters in the sidebar and click **Run Pricing** to start.")
    
    st.markdown("""
    ### Features
    - **Multi-asset Monte Carlo** with correlated GBM or Heston stochastic volatility
    - **Path-dependent payoff** with memory coupons, autocall, and knock-in barriers
    - **Greeks** via bump-and-revalue (Delta, Vega, Rho, Correlation sensitivity)
    - **Brownian bridge** correction for continuous barrier monitoring
    - **CVA adjustment** with CDS-implied hazard rates and EPE profiles
    - **Scenario analysis** across vol, correlation, and credit quality regimes
    """)
