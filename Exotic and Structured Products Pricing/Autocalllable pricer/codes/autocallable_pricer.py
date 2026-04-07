"""
Autocallable Worst-of Pricing Engine
=====================================
Monte Carlo pricer for worst-of autocallable structured notes.

Product structure:
- N underlying assets (e.g. 3 equity indices)
- Observation dates (typically quarterly or semi-annual)
- At each observation: if ALL underlyings >= autocall barrier, note redeems early with coupon
- At maturity: if worst-of underlying < knock-in barrier, investor bears the loss
- Otherwise: principal returned + final coupon

This is the most traded structured product in Europe. Understanding its pricing
requires multi-asset simulation with correlation, path-dependency, and barrier monitoring.

Author: Philippe-Emmanuel Yao
MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time


# ============================================================================
# Product Definition
# ============================================================================

@dataclass
class AutocallableNote:
    """Worst-of Autocallable Structured Note specification."""
    
    notional: float = 1_000_000          # Notional in currency units
    maturity_years: float = 3.0          # Total maturity
    n_assets: int = 3                    # Number of underlyings
    
    # Barrier levels (as fraction of initial spot)
    autocall_barrier: float = 1.0        # Autocall trigger (100% of initial)
    coupon_barrier: float = 0.70         # Coupon payment barrier (70%)
    knock_in_barrier: float = 0.60       # Capital protection barrier (60%)
    
    # Coupon structure
    coupon_rate: float = 0.08            # Annual coupon rate (8% p.a.)
    coupon_frequency: int = 2            # Semi-annual observations
    memory_coupon: bool = True           # Missed coupons paid later if barrier hit
    
    @property
    def n_observations(self) -> int:
        return int(self.maturity_years * self.coupon_frequency)
    
    @property
    def observation_times(self) -> np.ndarray:
        """Observation dates as year fractions."""
        return np.linspace(
            1 / self.coupon_frequency,
            self.maturity_years,
            self.n_observations
        )
    
    @property
    def coupon_per_period(self) -> float:
        return self.coupon_rate / self.coupon_frequency
    
    def describe(self) -> str:
        return (
            f"Worst-of Autocallable Note\n"
            f"{'=' * 40}\n"
            f"Notional:          {self.notional:,.0f}\n"
            f"Maturity:          {self.maturity_years}Y\n"
            f"Underlyings:       {self.n_assets} assets\n"
            f"Autocall barrier:  {self.autocall_barrier:.0%}\n"
            f"Coupon barrier:    {self.coupon_barrier:.0%}\n"
            f"Knock-in barrier:  {self.knock_in_barrier:.0%}\n"
            f"Coupon:            {self.coupon_rate:.1%} p.a. "
            f"({'memory' if self.memory_coupon else 'no memory'})\n"
            f"Observations:      {self.n_observations} "
            f"({'semi-annual' if self.coupon_frequency == 2 else 'quarterly'})\n"
        )


# ============================================================================
# Market Data
# ============================================================================

@dataclass
class MarketData:
    """Market parameters for multi-asset simulation."""
    
    spots: np.ndarray                    # Initial spot prices
    vols: np.ndarray                     # Annualised volatilities
    correlation_matrix: np.ndarray       # Asset correlation matrix
    risk_free_rate: float = 0.04         # Risk-free rate
    dividend_yields: Optional[np.ndarray] = None  # Continuous dividend yields
    
    def __post_init__(self):
        n = len(self.spots)
        if self.dividend_yields is None:
            self.dividend_yields = np.zeros(n)
        
        # Validate correlation matrix
        assert self.correlation_matrix.shape == (n, n), "Correlation matrix dimension mismatch"
        assert np.allclose(self.correlation_matrix, self.correlation_matrix.T), "Correlation must be symmetric"
        eigenvalues = np.linalg.eigvalsh(self.correlation_matrix)
        assert np.all(eigenvalues >= -1e-10), "Correlation matrix not positive semi-definite"
    
    @property
    def n_assets(self) -> int:
        return len(self.spots)
    
    @property
    def cholesky(self) -> np.ndarray:
        """Cholesky decomposition for correlated Brownian motion."""
        return np.linalg.cholesky(self.correlation_matrix)


# ============================================================================
# Monte Carlo Engine
# ============================================================================

class MonteCarloEngine:
    """
    Multi-asset Monte Carlo simulation engine.
    
    Uses correlated geometric Brownian motion:
        dS_i / S_i = (r - q_i) dt + sigma_i dW_i
    
    where dW_i are correlated via the Cholesky decomposition of the
    correlation matrix.
    """
    
    def __init__(self, n_paths: int = 100_000, seed: int = 42):
        self.n_paths = n_paths
        self.seed = seed
    
    def simulate_paths(
        self,
        market: MarketData,
        observation_times: np.ndarray,
        n_steps_per_period: int = 20
    ) -> np.ndarray:
        """
        Simulate correlated asset paths.
        
        Returns:
            paths: array of shape (n_paths, n_observations, n_assets)
                   containing S_i(t_j) / S_i(0) for each path, observation, and asset
        """
        rng = np.random.default_rng(self.seed)
        
        n_obs = len(observation_times)
        n_assets = market.n_assets
        L = market.cholesky
        
        # Build fine time grid
        all_times = [0.0]
        for i, t in enumerate(observation_times):
            t_prev = 0.0 if i == 0 else observation_times[i - 1]
            dt_period = t - t_prev
            n_sub = max(1, n_steps_per_period)
            sub_times = np.linspace(t_prev, t, n_sub + 1)[1:]
            all_times.extend(sub_times.tolist())
        
        all_times = np.array(all_times)
        dt_array = np.diff(all_times)
        n_total_steps = len(dt_array)
        
        # Map observation indices in the fine grid
        obs_indices = []
        for t in observation_times:
            idx = np.argmin(np.abs(all_times - t))
            obs_indices.append(idx)
        
        # Simulate log-returns
        # S(t) / S(0) = exp((r - q - 0.5*sigma^2)*t + sigma*W(t))
        log_S = np.zeros((self.n_paths, n_assets))
        paths_at_obs = np.zeros((self.n_paths, n_obs, n_assets))
        
        drift = market.risk_free_rate - market.dividend_yields - 0.5 * market.vols**2
        
        for step in range(n_total_steps):
            dt = dt_array[step]
            sqrt_dt = np.sqrt(dt)
            
            # Correlated normals: Z = L @ eps, where eps ~ N(0,1)
            eps = rng.standard_normal((self.n_paths, n_assets))
            Z = eps @ L.T  # (n_paths, n_assets)
            
            log_S += drift * dt + market.vols * sqrt_dt * Z
            
            # Record at observation dates
            if (step + 1) in obs_indices:
                obs_idx = obs_indices.index(step + 1)
                paths_at_obs[:, obs_idx, :] = np.exp(log_S)
        
        return paths_at_obs  # S(t) / S(0) for each observation
    
    def simulate_paths_continuous(
        self,
        market: MarketData,
        maturity: float,
        n_steps: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate full continuous paths for visualisation.
        
        Returns:
            times: (n_steps+1,)
            paths: (n_paths, n_steps+1, n_assets) as S(t)/S(0)
        """
        rng = np.random.default_rng(self.seed)
        
        n_assets = market.n_assets
        L = market.cholesky
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)
        
        times = np.linspace(0, maturity, n_steps + 1)
        paths = np.ones((self.n_paths, n_steps + 1, n_assets))
        
        drift = market.risk_free_rate - market.dividend_yields - 0.5 * market.vols**2
        log_S = np.zeros((self.n_paths, n_assets))
        
        for step in range(n_steps):
            eps = rng.standard_normal((self.n_paths, n_assets))
            Z = eps @ L.T
            log_S += drift * dt + market.vols * sqrt_dt * Z
            paths[:, step + 1, :] = np.exp(log_S)
        
        return times, paths


# ============================================================================
# Payoff Engine
# ============================================================================

class AutocallablePayoff:
    """
    Evaluates autocallable worst-of payoffs from simulated paths.
    
    Path-dependent logic:
    1. At each observation date, check if worst-of performance >= autocall barrier
       -> If yes: early redemption with accumulated coupons
    2. Track knock-in: has worst-of ever breached knock-in barrier?
    3. At maturity (if not autocalled):
       - If knock-in occurred: investor receives worst-of final performance
       - If no knock-in: principal returned + final coupon
    """
    
    def __init__(self, note: AutocallableNote, market: MarketData):
        self.note = note
        self.market = market
    
    def evaluate(self, paths: np.ndarray) -> dict:
        """
        Evaluate payoffs for all simulated paths.
        
        Args:
            paths: (n_paths, n_observations, n_assets) as S(t)/S(0)
        
        Returns:
            Dictionary with pricing results and diagnostics
        """
        n_paths = paths.shape[0]
        n_obs = paths.shape[1]
        note = self.note
        r = self.market.risk_free_rate
        obs_times = note.observation_times
        
        # Worst-of performance at each observation
        worst_of = np.min(paths, axis=2)  # (n_paths, n_obs)
        
        # Track results per path
        payoffs = np.zeros(n_paths)           # Undiscounted payoff
        disc_payoffs = np.zeros(n_paths)      # Discounted payoff
        redemption_times = np.full(n_paths, note.maturity_years)
        autocalled = np.zeros(n_paths, dtype=bool)
        knock_in_hit = np.zeros(n_paths, dtype=bool)
        coupons_paid = np.zeros(n_paths)
        
        # Track missed coupons for memory feature
        missed_coupons = np.zeros(n_paths)
        
        for path_idx in range(n_paths):
            path_done = False
            path_knock_in = False
            path_missed = 0.0
            path_coupons = 0.0
            
            for obs_idx in range(n_obs):
                t = obs_times[obs_idx]
                wo = worst_of[path_idx, obs_idx]
                df = np.exp(-r * t)
                
                # Check knock-in (continuous approximation via observation dates)
                if wo < note.knock_in_barrier:
                    path_knock_in = True
                
                # Check coupon barrier
                if wo >= note.coupon_barrier:
                    coupon = note.coupon_per_period
                    if note.memory_coupon:
                        coupon += path_missed
                        path_missed = 0.0
                    path_coupons += coupon
                else:
                    if note.memory_coupon:
                        path_missed += note.coupon_per_period
                
                # Check autocall (typically not on first observation)
                if obs_idx > 0 and wo >= note.autocall_barrier:
                    payoff = note.notional * (1.0 + path_coupons)
                    payoffs[path_idx] = payoff
                    disc_payoffs[path_idx] = payoff * df
                    redemption_times[path_idx] = t
                    autocalled[path_idx] = True
                    coupons_paid[path_idx] = path_coupons
                    path_done = True
                    break
            
            if not path_done:
                # Maturity payoff
                t = note.maturity_years
                df = np.exp(-r * t)
                wo_final = worst_of[path_idx, -1]
                
                if path_knock_in and wo_final < note.knock_in_barrier:
                    # Capital loss: investor receives worst-of performance
                    payoff = note.notional * wo_final
                else:
                    # Principal protected + final coupon
                    if wo_final >= note.coupon_barrier:
                        coupon = note.coupon_per_period
                        if note.memory_coupon:
                            coupon += path_missed
                        path_coupons += coupon
                    payoff = note.notional * (1.0 + path_coupons)
                
                payoffs[path_idx] = payoff
                disc_payoffs[path_idx] = payoff * df
                knock_in_hit[path_idx] = path_knock_in and wo_final < note.knock_in_barrier
                coupons_paid[path_idx] = path_coupons
        
        # Compute statistics
        price = np.mean(disc_payoffs)
        price_pct = price / note.notional
        std_err = np.std(disc_payoffs) / np.sqrt(n_paths)
        
        return {
            'price': price,
            'price_pct': price_pct,
            'std_error': std_err,
            'confidence_95': (price - 1.96 * std_err, price + 1.96 * std_err),
            'autocall_prob': np.mean(autocalled),
            'knock_in_prob': np.mean(knock_in_hit),
            'avg_redemption_time': np.mean(redemption_times),
            'avg_coupon': np.mean(coupons_paid),
            'payoff_distribution': disc_payoffs,
            'autocalled': autocalled,
            'redemption_times': redemption_times,
            'worst_of_final': worst_of[:, -1],
        }


# ============================================================================
# Greeks via Bump-and-Revalue
# ============================================================================

class GreeksCalculator:
    """
    Finite-difference Greeks for the autocallable.
    
    Uses bump-and-revalue with the same random seed for variance reduction.
    """
    
    def __init__(self, engine: MonteCarloEngine):
        self.engine = engine
    
    def compute_greeks(
        self,
        note: AutocallableNote,
        market: MarketData,
        spot_bump: float = 0.01,
        vol_bump: float = 0.01,
        rate_bump: float = 0.001
    ) -> dict:
        """Compute Delta, Vega, Rho for each underlying."""
        
        base_paths = self.engine.simulate_paths(market, note.observation_times)
        base_price = AutocallablePayoff(note, market).evaluate(base_paths)['price']
        
        greeks = {'base_price': base_price}
        
        # Delta per asset (dV/dS * S/V %)
        for i in range(market.n_assets):
            # Bump spot up
            spots_up = market.spots.copy()
            spots_up[i] *= (1 + spot_bump)
            market_up = MarketData(
                spots_up, market.vols, market.correlation_matrix,
                market.risk_free_rate, market.dividend_yields
            )
            paths_up = self.engine.simulate_paths(market_up, note.observation_times)
            price_up = AutocallablePayoff(note, market_up).evaluate(paths_up)['price']
            
            # Bump spot down
            spots_dn = market.spots.copy()
            spots_dn[i] *= (1 - spot_bump)
            market_dn = MarketData(
                spots_dn, market.vols, market.correlation_matrix,
                market.risk_free_rate, market.dividend_yields
            )
            paths_dn = self.engine.simulate_paths(market_dn, note.observation_times)
            price_dn = AutocallablePayoff(note, market_dn).evaluate(paths_dn)['price']
            
            delta = (price_up - price_dn) / (2 * spot_bump * market.spots[i])
            greeks[f'delta_asset_{i+1}'] = delta
        
        # Vega (parallel vol bump)
        vols_up = market.vols + vol_bump
        market_vup = MarketData(
            market.spots, vols_up, market.correlation_matrix,
            market.risk_free_rate, market.dividend_yields
        )
        paths_vup = self.engine.simulate_paths(market_vup, note.observation_times)
        price_vup = AutocallablePayoff(note, market_vup).evaluate(paths_vup)['price']
        greeks['vega'] = (price_vup - base_price) / vol_bump
        
        # Rho
        market_rup = MarketData(
            market.spots, market.vols, market.correlation_matrix,
            market.risk_free_rate + rate_bump, market.dividend_yields
        )
        paths_rup = self.engine.simulate_paths(market_rup, note.observation_times)
        price_rup = AutocallablePayoff(note, market_rup).evaluate(paths_rup)['price']
        greeks['rho'] = (price_rup - base_price) / rate_bump
        
        # Correlation sensitivity (bump rho_12 by 5%)
        corr_bump = 0.05
        corr_up = market.correlation_matrix.copy()
        if market.n_assets >= 2:
            corr_up[0, 1] = min(corr_up[0, 1] + corr_bump, 0.99)
            corr_up[1, 0] = corr_up[0, 1]
            try:
                market_cup = MarketData(
                    market.spots, market.vols, corr_up,
                    market.risk_free_rate, market.dividend_yields
                )
                paths_cup = self.engine.simulate_paths(market_cup, note.observation_times)
                price_cup = AutocallablePayoff(note, market_cup).evaluate(paths_cup)['price']
                greeks['corr_sensitivity'] = (price_cup - base_price) / corr_bump
            except AssertionError:
                greeks['corr_sensitivity'] = np.nan
        
        return greeks


# ============================================================================
# Scenario Analysis
# ============================================================================

def run_scenario_analysis(
    note: AutocallableNote,
    market: MarketData,
    engine: MonteCarloEngine
) -> dict:
    """Run pricing across different market scenarios."""
    
    scenarios = {}
    
    # Base case
    paths = engine.simulate_paths(market, note.observation_times)
    scenarios['base'] = AutocallablePayoff(note, market).evaluate(paths)
    
    # High vol scenario (+10% absolute)
    market_hv = MarketData(
        market.spots, market.vols + 0.10, market.correlation_matrix,
        market.risk_free_rate, market.dividend_yields
    )
    paths_hv = engine.simulate_paths(market_hv, note.observation_times)
    scenarios['high_vol'] = AutocallablePayoff(note, market_hv).evaluate(paths_hv)
    
    # Low vol scenario (-5% absolute)
    market_lv = MarketData(
        market.spots, np.maximum(market.vols - 0.05, 0.05), market.correlation_matrix,
        market.risk_free_rate, market.dividend_yields
    )
    paths_lv = engine.simulate_paths(market_lv, note.observation_times)
    scenarios['low_vol'] = AutocallablePayoff(note, market_lv).evaluate(paths_lv)
    
    # High correlation (0.9)
    n = market.n_assets
    corr_high = np.full((n, n), 0.9)
    np.fill_diagonal(corr_high, 1.0)
    market_hc = MarketData(
        market.spots, market.vols, corr_high,
        market.risk_free_rate, market.dividend_yields
    )
    paths_hc = engine.simulate_paths(market_hc, note.observation_times)
    scenarios['high_corr'] = AutocallablePayoff(note, market_hc).evaluate(paths_hc)
    
    # Low correlation (0.3)
    corr_low = np.full((n, n), 0.3)
    np.fill_diagonal(corr_low, 1.0)
    market_lc = MarketData(
        market.spots, market.vols, corr_low,
        market.risk_free_rate, market.dividend_yields
    )
    paths_lc = engine.simulate_paths(market_lc, note.observation_times)
    scenarios['low_corr'] = AutocallablePayoff(note, market_lc).evaluate(paths_lc)
    
    return scenarios


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 60)
    print("AUTOCALLABLE WORST-OF PRICING ENGINE")
    print("Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE")
    print("=" * 60)
    
    # ── Product specification ──
    note = AutocallableNote(
        notional=1_000_000,
        maturity_years=3.0,
        n_assets=3,
        autocall_barrier=1.0,
        coupon_barrier=0.70,
        knock_in_barrier=0.60,
        coupon_rate=0.08,
        coupon_frequency=2,
        memory_coupon=True
    )
    print("\n" + note.describe())
    
    # ── Market data (3 equity indices) ──
    # Example: SX5E, SPX, NKY
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
    
    print("Market Data")
    print("-" * 40)
    print(f"Assets:      SX5E={market.spots[0]:.0f}, SPX={market.spots[1]:.0f}, NKY={market.spots[2]:.0f}")
    print(f"Vols:        {market.vols[0]:.0%}, {market.vols[1]:.0%}, {market.vols[2]:.0%}")
    print(f"Risk-free:   {market.risk_free_rate:.1%}")
    print(f"Div yields:  {market.dividend_yields[0]:.1%}, {market.dividend_yields[1]:.1%}, {market.dividend_yields[2]:.1%}")
    print(f"Correlation:\n{market.correlation_matrix}\n")
    
    # ── Monte Carlo pricing ──
    print("Running Monte Carlo simulation...")
    engine = MonteCarloEngine(n_paths=200_000, seed=42)
    
    t0 = time.time()
    paths = engine.simulate_paths(market, note.observation_times)
    t_sim = time.time() - t0
    
    t0 = time.time()
    results = AutocallablePayoff(note, market).evaluate(paths)
    t_payoff = time.time() - t0
    
    print(f"Simulation:  {t_sim:.2f}s")
    print(f"Payoff eval: {t_payoff:.2f}s")
    print(f"Paths:       {engine.n_paths:,}\n")
    
    print("PRICING RESULTS")
    print("=" * 40)
    print(f"Price:              {results['price']:,.0f} ({results['price_pct']:.2%} of notional)")
    print(f"95% CI:             [{results['confidence_95'][0]:,.0f}, {results['confidence_95'][1]:,.0f}]")
    print(f"Std Error:          {results['std_error']:,.0f}")
    print(f"")
    print(f"Autocall prob:      {results['autocall_prob']:.1%}")
    print(f"Knock-in prob:      {results['knock_in_prob']:.1%}")
    print(f"Avg redemption:     {results['avg_redemption_time']:.2f}Y")
    print(f"Avg coupon earned:  {results['avg_coupon']:.2%}")
    
    # ── Greeks ──
    print("\n\nGREEKS (bump-and-revalue)")
    print("=" * 40)
    greeks_calc = GreeksCalculator(engine)
    greeks = greeks_calc.compute_greeks(note, market)
    
    for i in range(market.n_assets):
        print(f"Delta (asset {i+1}):    {greeks[f'delta_asset_{i+1}']:,.0f}")
    print(f"Vega (1% vol):      {greeks['vega']:,.0f}")
    print(f"Rho (10bp):         {greeks['rho']:,.0f}")
    if 'corr_sensitivity' in greeks:
        print(f"Corr sens (5%):     {greeks['corr_sensitivity']:,.0f}")
    
    # ── Scenario analysis ──
    print("\n\nSCENARIO ANALYSIS")
    print("=" * 40)
    scenarios = run_scenario_analysis(note, market, engine)
    
    print(f"{'Scenario':<16} {'Price':>12} {'% Notional':>12} {'Autocall%':>10} {'KI%':>8}")
    print("-" * 60)
    for name, res in scenarios.items():
        print(f"{name:<16} {res['price']:>12,.0f} {res['price_pct']:>11.2%} {res['autocall_prob']:>9.1%} {res['knock_in_prob']:>7.1%}")
    
    # ── Payoff distribution statistics ──
    payoffs = results['payoff_distribution']
    print("\n\nPAYOFF DISTRIBUTION")
    print("=" * 40)
    print(f"Mean:       {np.mean(payoffs):,.0f}")
    print(f"Std Dev:    {np.std(payoffs):,.0f}")
    print(f"Skewness:   {float(np.mean(((payoffs - np.mean(payoffs)) / np.std(payoffs))**3)):.2f}")
    print(f"5th pct:    {np.percentile(payoffs, 5):,.0f}")
    print(f"25th pct:   {np.percentile(payoffs, 25):,.0f}")
    print(f"Median:     {np.median(payoffs):,.0f}")
    print(f"75th pct:   {np.percentile(payoffs, 75):,.0f}")
    print(f"95th pct:   {np.percentile(payoffs, 95):,.0f}")
    print(f"Max loss:   {np.min(payoffs):,.0f} ({np.min(payoffs)/note.notional - 1:.1%})")
    
    return results, greeks, scenarios


if __name__ == "__main__":
    results, greeks, scenarios = main()
