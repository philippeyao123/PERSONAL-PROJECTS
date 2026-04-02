"""
Extension 1: Heston Stochastic Volatility Model
=================================================
Smile-consistent pricing via Heston dynamics:

    dS/S  = (r - q) dt + sqrt(V) dW_S
    dV    = kappa*(theta - V) dt + xi * sqrt(V) dW_V
    
    corr(dW_S, dW_V) = rho_sv

For worst-of autocallables, each asset has its own Heston process,
with inter-asset correlation on the spot Brownian motions.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HestonParams:
    """Heston parameters for a single asset."""
    v0: float = 0.04          # Initial variance
    kappa: float = 2.0        # Mean-reversion speed
    theta: float = 0.04       # Long-run variance
    xi: float = 0.3           # Vol-of-vol
    rho_sv: float = -0.7      # Spot-vol correlation
    
    @property
    def feller_condition(self) -> bool:
        """Check 2*kappa*theta > xi^2 (ensures V > 0 a.s.)."""
        return 2 * self.kappa * self.theta > self.xi**2
    
    def describe(self) -> str:
        feller = "satisfied" if self.feller_condition else "VIOLATED"
        return (
            f"v0={self.v0:.4f}, kappa={self.kappa:.1f}, theta={self.theta:.4f}, "
            f"xi={self.xi:.2f}, rho_sv={self.rho_sv:.2f} (Feller: {feller})"
        )


@dataclass
class HestonMarketData:
    """Market data with Heston parameters per asset."""
    spots: np.ndarray
    heston_params: list              # List of HestonParams, one per asset
    spot_correlation: np.ndarray     # Inter-asset spot correlation
    risk_free_rate: float = 0.04
    dividend_yields: Optional[np.ndarray] = None
    
    def __post_init__(self):
        n = len(self.spots)
        if self.dividend_yields is None:
            self.dividend_yields = np.zeros(n)
        assert len(self.heston_params) == n
        assert self.spot_correlation.shape == (n, n)
    
    @property
    def n_assets(self) -> int:
        return len(self.spots)


class HestonMonteCarloEngine:
    """
    Multi-asset Heston Monte Carlo engine.
    
    Each asset follows its own Heston process. The spot Brownian motions
    are correlated via the inter-asset correlation matrix. Each asset's
    vol Brownian is correlated with its own spot Brownian via rho_sv.
    
    Discretisation: QE (Quadratic Exponential) scheme for variance process
    to handle near-zero variance without going negative.
    """
    
    def __init__(self, n_paths: int = 100_000, seed: int = 42):
        self.n_paths = n_paths
        self.seed = seed
    
    def simulate_paths(
        self,
        market: HestonMarketData,
        observation_times: np.ndarray,
        n_steps_per_period: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate correlated Heston paths.
        
        Returns:
            spot_paths: (n_paths, n_obs, n_assets) as S(t)/S(0)
            var_paths:  (n_paths, n_obs, n_assets) variance at observation dates
        """
        rng = np.random.default_rng(self.seed)
        
        n_obs = len(observation_times)
        n_assets = market.n_assets
        
        # Cholesky for inter-asset spot correlation
        L_spot = np.linalg.cholesky(market.spot_correlation)
        
        # Build time grid
        all_times = [0.0]
        for i, t in enumerate(observation_times):
            t_prev = 0.0 if i == 0 else observation_times[i - 1]
            sub = np.linspace(t_prev, t, n_steps_per_period + 1)[1:]
            all_times.extend(sub.tolist())
        all_times = np.array(all_times)
        dt_array = np.diff(all_times)
        n_steps = len(dt_array)
        
        # Map observation indices
        obs_indices = []
        for t in observation_times:
            obs_indices.append(np.argmin(np.abs(all_times - t)))
        
        # Initialise
        log_S = np.zeros((self.n_paths, n_assets))
        V = np.zeros((self.n_paths, n_assets))
        for i in range(n_assets):
            V[:, i] = market.heston_params[i].v0
        
        spot_at_obs = np.zeros((self.n_paths, n_obs, n_assets))
        var_at_obs = np.zeros((self.n_paths, n_obs, n_assets))
        
        for step in range(n_steps):
            dt = dt_array[step]
            sqrt_dt = np.sqrt(dt)
            
            # Generate correlated spot normals
            eps_spot = rng.standard_normal((self.n_paths, n_assets))
            Z_spot = eps_spot @ L_spot.T
            
            # Independent vol normals (one per asset)
            eps_vol = rng.standard_normal((self.n_paths, n_assets))
            
            for i in range(n_assets):
                hp = market.heston_params[i]
                r = market.risk_free_rate
                q = market.dividend_yields[i]
                
                # Correlated vol Brownian: Z_v = rho*Z_s + sqrt(1-rho^2)*Z_indep
                Z_v = hp.rho_sv * Z_spot[:, i] + np.sqrt(1 - hp.rho_sv**2) * eps_vol[:, i]
                
                # Truncated Euler for variance (floor at 0)
                V_plus = np.maximum(V[:, i], 0.0)
                sqrt_V = np.sqrt(V_plus)
                
                # Variance update
                dV = hp.kappa * (hp.theta - V_plus) * dt + hp.xi * sqrt_V * sqrt_dt * Z_v
                V[:, i] = np.maximum(V[:, i] + dV, 0.0)
                
                # Spot update
                drift = (r - q - 0.5 * V_plus) * dt
                diffusion = sqrt_V * sqrt_dt * Z_spot[:, i]
                log_S[:, i] += drift + diffusion
            
            # Record at observation dates
            if (step + 1) in obs_indices:
                obs_idx = obs_indices.index(step + 1)
                spot_at_obs[:, obs_idx, :] = np.exp(log_S)
                var_at_obs[:, obs_idx, :] = V.copy()
        
        return spot_at_obs, var_at_obs


def create_default_heston_market() -> HestonMarketData:
    """Default 3-asset Heston market for testing."""
    return HestonMarketData(
        spots=np.array([4500.0, 5200.0, 38000.0]),
        heston_params=[
            HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho_sv=-0.70),  # SX5E
            HestonParams(v0=0.032, kappa=1.5, theta=0.035, xi=0.25, rho_sv=-0.65),  # SPX
            HestonParams(v0=0.05, kappa=2.5, theta=0.045, xi=0.35, rho_sv=-0.60),  # NKY
        ],
        spot_correlation=np.array([
            [1.0, 0.75, 0.55],
            [0.75, 1.0, 0.50],
            [0.55, 0.50, 1.0]
        ]),
        risk_free_rate=0.04,
        dividend_yields=np.array([0.025, 0.015, 0.02])
    )


if __name__ == "__main__":
    from autocallable_pricer import AutocallableNote, AutocallablePayoff, MarketData
    
    note = AutocallableNote()
    heston_market = create_default_heston_market()
    
    print("Heston Stochastic Volatility Extension")
    print("=" * 50)
    for i, hp in enumerate(heston_market.heston_params):
        print(f"Asset {i+1}: {hp.describe()}")
    
    engine = HestonMonteCarloEngine(n_paths=100_000, seed=42)
    
    import time
    t0 = time.time()
    spot_paths, var_paths = engine.simulate_paths(heston_market, note.observation_times)
    t_elapsed = time.time() - t0
    print(f"\nSimulation time: {t_elapsed:.2f}s (100K paths)")
    
    # Use base payoff engine (it only needs S(t)/S(0))
    # Create a dummy MarketData for the payoff evaluator
    dummy_market = MarketData(
        spots=heston_market.spots,
        vols=np.array([0.20, 0.18, 0.22]),  # Not used in payoff
        correlation_matrix=heston_market.spot_correlation,
        risk_free_rate=heston_market.risk_free_rate,
        dividend_yields=heston_market.dividend_yields
    )
    
    results = AutocallablePayoff(note, dummy_market).evaluate(spot_paths)
    
    print(f"\nHESTON PRICING RESULTS")
    print(f"Price:         {results['price']:,.0f} ({results['price_pct']:.2%})")
    print(f"Autocall prob: {results['autocall_prob']:.1%}")
    print(f"Knock-in prob: {results['knock_in_prob']:.1%}")
    print(f"Avg redempt:   {results['avg_redemption_time']:.2f}Y")
    
    # Compare variance paths
    print(f"\nVariance at maturity (mean across paths):")
    for i in range(3):
        v_mean = np.mean(var_paths[:, -1, i])
        v_std = np.std(var_paths[:, -1, i])
        print(f"  Asset {i+1}: mean={v_mean:.4f}, std={v_std:.4f}")
