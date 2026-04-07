"""
Extension 2: Brownian Bridge Barrier Correction
=================================================
Discrete barrier monitoring underestimates the true knock-in probability
because the process can breach the barrier between observation dates.

The Brownian bridge gives the exact probability that a path crosses a
barrier between two discrete points:

    P(min S(t) < B | S(t1), S(t2)) = exp(-2 * ln(S(t1)/B) * ln(S(t2)/B) / (sigma^2 * dt))

This correction is critical for autocallables where knock-in barriers
are typically monitored continuously but simulated discretely.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from typing import Tuple


def brownian_bridge_min_probability(
    s1: np.ndarray,
    s2: np.ndarray,
    barrier: float,
    sigma: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Probability that the minimum of a GBM path between two points
    falls below a barrier level, conditional on endpoint values.
    
    Uses the Brownian bridge result for geometric Brownian motion:
        P(min_{t1<t<t2} S(t) < B | S(t1), S(t2)) 
        = exp(-2 * log(S(t1)/B) * log(S(t2)/B) / (sigma^2 * dt))
    
    Valid when both S(t1) > B and S(t2) > B.
    
    Args:
        s1: Asset values at t1 (as S/S0 performance)
        s2: Asset values at t2
        barrier: Barrier level (as fraction of initial)
        sigma: Volatility of each path
        dt: Time between observations
    
    Returns:
        Probability of barrier breach for each path
    """
    # If either endpoint is below barrier, breach is certain
    breach_certain = (s1 <= barrier) | (s2 <= barrier)
    
    # For paths where both endpoints are above barrier
    safe_mask = ~breach_certain
    prob = np.zeros_like(s1)
    prob[breach_certain] = 1.0
    
    if np.any(safe_mask):
        log_ratio_1 = np.log(s1[safe_mask] / barrier)
        log_ratio_2 = np.log(s2[safe_mask] / barrier)
        var_dt = sigma[safe_mask]**2 * dt
        
        # Avoid division by zero
        var_dt = np.maximum(var_dt, 1e-12)
        
        exponent = -2.0 * log_ratio_1 * log_ratio_2 / var_dt
        prob[safe_mask] = np.exp(np.minimum(exponent, 0.0))
    
    return prob


def apply_brownian_bridge_correction(
    paths: np.ndarray,
    knock_in_barrier: float,
    vols: np.ndarray,
    observation_times: np.ndarray,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Apply Brownian bridge correction to determine continuous knock-in events.
    
    For each path and each pair of consecutive observations, compute the
    probability that the worst-of breached the knock-in barrier between
    observations, and sample accordingly.
    
    Args:
        paths: (n_paths, n_obs, n_assets) as S(t)/S(0)
        knock_in_barrier: Barrier level
        vols: (n_assets,) volatilities
        observation_times: (n_obs,) observation dates
        rng: Random number generator
    
    Returns:
        knock_in_continuous: (n_paths,) boolean array indicating continuous knock-in
    """
    if rng is None:
        rng = np.random.default_rng(123)
    
    n_paths, n_obs, n_assets = paths.shape
    knock_in = np.zeros(n_paths, dtype=bool)
    
    # Check at observation dates first (discrete)
    worst_of = np.min(paths, axis=2)
    knock_in |= np.any(worst_of < knock_in_barrier, axis=1)
    
    # Between-observation correction
    for obs in range(n_obs):
        if obs == 0:
            # Between t=0 and first observation
            s1 = np.ones((n_paths, n_assets))  # S(0)/S(0) = 1
            s2 = paths[:, 0, :]
            dt = observation_times[0]
        else:
            s1 = paths[:, obs - 1, :]
            s2 = paths[:, obs, :]
            dt = observation_times[obs] - observation_times[obs - 1]
        
        # For each asset, compute bridge probability
        for asset in range(n_assets):
            vol_asset = np.full(n_paths, vols[asset])
            prob = brownian_bridge_min_probability(
                s1[:, asset], s2[:, asset],
                knock_in_barrier, vol_asset, dt
            )
            
            # Sample: does the path breach between observations?
            u = rng.uniform(size=n_paths)
            knock_in |= (u < prob)
    
    return knock_in


def compare_discrete_vs_continuous(
    paths: np.ndarray,
    knock_in_barrier: float,
    vols: np.ndarray,
    observation_times: np.ndarray
) -> dict:
    """
    Compare discrete vs continuous barrier monitoring.
    
    Returns statistics showing how much the discrete approximation
    underestimates knock-in probability.
    """
    n_paths = paths.shape[0]
    
    # Discrete monitoring
    worst_of = np.min(paths, axis=2)
    ki_discrete = np.any(worst_of < knock_in_barrier, axis=1)
    
    # Continuous monitoring (Brownian bridge)
    ki_continuous = apply_brownian_bridge_correction(
        paths, knock_in_barrier, vols, observation_times
    )
    
    return {
        'ki_prob_discrete': np.mean(ki_discrete),
        'ki_prob_continuous': np.mean(ki_continuous),
        'correction_factor': np.mean(ki_continuous) / max(np.mean(ki_discrete), 1e-10),
        'additional_ki_paths': np.sum(ki_continuous & ~ki_discrete),
        'additional_ki_pct': np.mean(ki_continuous & ~ki_discrete),
    }


if __name__ == "__main__":
    from autocallable_pricer import (
        AutocallableNote, MarketData, MonteCarloEngine, AutocallablePayoff
    )
    
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
    paths = engine.simulate_paths(market, note.observation_times)
    
    print("Brownian Bridge Barrier Correction")
    print("=" * 50)
    
    comparison = compare_discrete_vs_continuous(
        paths, note.knock_in_barrier, market.vols, note.observation_times
    )
    
    print(f"Discrete KI probability:   {comparison['ki_prob_discrete']:.2%}")
    print(f"Continuous KI probability:  {comparison['ki_prob_continuous']:.2%}")
    print(f"Correction factor:          {comparison['correction_factor']:.2f}x")
    print(f"Additional KI paths:        {comparison['additional_ki_paths']:,} ({comparison['additional_ki_pct']:.2%})")
    print(f"\nThe discrete approximation underestimates knock-in by ~{(comparison['correction_factor']-1)*100:.0f}%")
