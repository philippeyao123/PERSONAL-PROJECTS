"""
Extension 3: Term Structure of Rates and Dividends
====================================================
Replaces flat rates/dividends with piecewise-linear term structures
for more realistic pricing.

In practice, autocallable pricing is highly sensitive to:
- Forward rates (affects discounting and drift)
- Dividend curves (affects forward levels, especially for equity indices)

This module implements interpolated curves and a modified MC engine
that uses time-dependent drift.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Curve:
    """Piecewise-linear interpolated curve."""
    tenors: np.ndarray       # Year fractions
    values: np.ndarray       # Rate/yield values (continuously compounded)
    
    def __post_init__(self):
        assert len(self.tenors) == len(self.values)
        # Ensure sorted
        idx = np.argsort(self.tenors)
        self.tenors = self.tenors[idx]
        self.values = self.values[idx]
    
    def __call__(self, t: float) -> float:
        """Interpolate rate at time t."""
        return float(np.interp(t, self.tenors, self.values))
    
    def forward_rate(self, t1: float, t2: float) -> float:
        """Forward rate between t1 and t2."""
        if abs(t2 - t1) < 1e-10:
            return self(t1)
        r1 = self(t1)
        r2 = self(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)
    
    def discount_factor(self, t: float) -> float:
        """Discount factor to time t."""
        return np.exp(-self(t) * t)
    
    def describe(self) -> str:
        parts = [f"{t:.1f}Y: {v:.2%}" for t, v in zip(self.tenors, self.values)]
        return " | ".join(parts)


@dataclass
class TermStructureMarket:
    """Market data with term structures."""
    spots: np.ndarray
    vols: np.ndarray
    correlation_matrix: np.ndarray
    rate_curve: Curve                           # Risk-free rate curve
    dividend_curves: List[Curve]                # One per asset
    
    @property
    def n_assets(self) -> int:
        return len(self.spots)
    
    @property
    def cholesky(self) -> np.ndarray:
        return np.linalg.cholesky(self.correlation_matrix)
    
    def flat_rate_at(self, t: float) -> float:
        return self.rate_curve(t)
    
    def flat_div_at(self, asset: int, t: float) -> float:
        return self.dividend_curves[asset](t)


class TermStructureMCEngine:
    """Monte Carlo engine with time-dependent rates and dividends."""
    
    def __init__(self, n_paths: int = 100_000, seed: int = 42):
        self.n_paths = n_paths
        self.seed = seed
    
    def simulate_paths(
        self,
        market: TermStructureMarket,
        observation_times: np.ndarray,
        n_steps_per_period: int = 20
    ) -> np.ndarray:
        """
        Simulate paths with time-dependent drift from term structures.
        
        Returns:
            paths: (n_paths, n_obs, n_assets) as S(t)/S(0)
        """
        rng = np.random.default_rng(self.seed)
        
        n_obs = len(observation_times)
        n_assets = market.n_assets
        L = market.cholesky
        
        # Build fine time grid
        all_times = [0.0]
        for i, t in enumerate(observation_times):
            t_prev = 0.0 if i == 0 else observation_times[i - 1]
            sub = np.linspace(t_prev, t, n_steps_per_period + 1)[1:]
            all_times.extend(sub.tolist())
        all_times = np.array(all_times)
        dt_array = np.diff(all_times)
        n_steps = len(dt_array)
        
        obs_indices = []
        for t in observation_times:
            obs_indices.append(np.argmin(np.abs(all_times - t)))
        
        log_S = np.zeros((self.n_paths, n_assets))
        paths_at_obs = np.zeros((self.n_paths, n_obs, n_assets))
        
        for step in range(n_steps):
            dt = dt_array[step]
            sqrt_dt = np.sqrt(dt)
            t_mid = all_times[step] + dt / 2  # Midpoint for curve evaluation
            
            # Time-dependent drift per asset
            r_t = market.rate_curve(t_mid)
            drift = np.zeros(n_assets)
            for i in range(n_assets):
                q_t = market.dividend_curves[i](t_mid)
                drift[i] = (r_t - q_t - 0.5 * market.vols[i]**2) * dt
            
            eps = rng.standard_normal((self.n_paths, n_assets))
            Z = eps @ L.T
            
            log_S += drift + market.vols * sqrt_dt * Z
            
            if (step + 1) in obs_indices:
                obs_idx = obs_indices.index(step + 1)
                paths_at_obs[:, obs_idx, :] = np.exp(log_S)
        
        return paths_at_obs


def create_default_term_structure_market() -> TermStructureMarket:
    """Default market with realistic EUR/USD/JPY curves."""
    
    # EUR risk-free (ECB-style, inverted for short end)
    rate_curve = Curve(
        tenors=np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]),
        values=np.array([0.038, 0.037, 0.035, 0.033, 0.032, 0.031, 0.030])
    )
    
    # Dividend curves per asset
    div_sx5e = Curve(
        tenors=np.array([0.5, 1.0, 2.0, 3.0]),
        values=np.array([0.030, 0.028, 0.025, 0.023])
    )
    div_spx = Curve(
        tenors=np.array([0.5, 1.0, 2.0, 3.0]),
        values=np.array([0.015, 0.015, 0.014, 0.013])
    )
    div_nky = Curve(
        tenors=np.array([0.5, 1.0, 2.0, 3.0]),
        values=np.array([0.022, 0.020, 0.019, 0.018])
    )
    
    return TermStructureMarket(
        spots=np.array([4500.0, 5200.0, 38000.0]),
        vols=np.array([0.20, 0.18, 0.22]),
        correlation_matrix=np.array([
            [1.0, 0.75, 0.55],
            [0.75, 1.0, 0.50],
            [0.55, 0.50, 1.0]
        ]),
        rate_curve=rate_curve,
        dividend_curves=[div_sx5e, div_spx, div_nky]
    )


if __name__ == "__main__":
    from autocallable_pricer import AutocallableNote, AutocallablePayoff, MarketData
    import time
    
    note = AutocallableNote()
    ts_market = create_default_term_structure_market()
    
    print("Term Structure Extension")
    print("=" * 50)
    print(f"Rate curve: {ts_market.rate_curve.describe()}")
    for i, dc in enumerate(ts_market.dividend_curves):
        print(f"Div curve {i+1}: {dc.describe()}")
    
    # Price with term structure
    engine = TermStructureMCEngine(n_paths=100_000, seed=42)
    
    t0 = time.time()
    paths = engine.simulate_paths(ts_market, note.observation_times)
    print(f"\nSimulation: {time.time() - t0:.2f}s")
    
    # Need a MarketData wrapper for payoff (uses r for discounting)
    # Use the 3Y rate for flat discounting approximation
    r_3y = ts_market.rate_curve(3.0)
    dummy_market = MarketData(
        spots=ts_market.spots,
        vols=ts_market.vols,
        correlation_matrix=ts_market.correlation_matrix,
        risk_free_rate=r_3y,
        dividend_yields=np.array([
            ts_market.dividend_curves[i](1.5) for i in range(3)
        ])
    )
    
    results = AutocallablePayoff(note, dummy_market).evaluate(paths)
    
    print(f"\nTERM STRUCTURE PRICING RESULTS")
    print(f"Price:         {results['price']:,.0f} ({results['price_pct']:.2%})")
    print(f"Autocall prob: {results['autocall_prob']:.1%}")
    print(f"Knock-in prob: {results['knock_in_prob']:.1%}")
    
    # Compare forward levels
    print(f"\nForward rates:")
    for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        fwd = ts_market.rate_curve(t)
        print(f"  r({t:.1f}Y) = {fwd:.2%}")
