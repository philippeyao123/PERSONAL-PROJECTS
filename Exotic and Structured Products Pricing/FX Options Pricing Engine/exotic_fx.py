"""
Extension 2: Exotic FX Options
================================
Beyond single barriers: structured FX products traded by exotic desks.

Products:
  1. Double No-Touch (DNT): pays if spot stays within corridor
  2. Window Barrier: barrier active only during a time window
  3. Digital (Binary): pays fixed amount if ITM at expiry
  4. Best-of Options: option on max/min of two FX pairs

These are the bread and butter of FX exotic desks at major banks.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import time


class ExoticFXPricer:
    """MC pricer for exotic FX options under local vol."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def _simulate_paths(
        self,
        S0: float,
        rd: float,
        rf: float,
        local_vol_func,
        T: float,
        n_paths: int = 100000,
        n_steps: int = 500
    ) -> np.ndarray:
        """Simulate full paths (n_paths, n_steps+1)."""
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for step in range(n_steps):
            t = step * dt
            S = paths[:, step]
            Z = self.rng.standard_normal(n_paths)
            pts = np.column_stack((np.full(n_paths, t), np.clip(S, 0.1, 5*S0)))
            try:
                sigma = local_vol_func(pts).flatten()
            except Exception:
                sigma = np.full(n_paths, 0.20)
            paths[:, step+1] = S * np.exp((rd - rf - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
            paths[:, step+1] = np.maximum(paths[:, step+1], 1e-8)
        
        return paths
    
    def double_no_touch(
        self,
        S0, rd, rf, local_vol_func, T,
        lower_barrier, upper_barrier,
        notional=1.0,
        n_paths=100000, n_steps=500
    ) -> dict:
        """
        Double No-Touch: pays notional if spot stays within [L, U] for entire life.
        
        Popular in FX for range-bound markets.
        """
        paths = self._simulate_paths(S0, rd, rf, local_vol_func, T, n_paths, n_steps)
        
        # Check if any path point breaches either barrier
        min_vals = np.min(paths, axis=1)
        max_vals = np.max(paths, axis=1)
        
        survived = (min_vals > lower_barrier) & (max_vals < upper_barrier)
        
        price = np.exp(-rd * T) * notional * np.mean(survived)
        prob = np.mean(survived)
        
        return {
            'price': price,
            'survival_probability': prob,
            'lower_barrier': lower_barrier,
            'upper_barrier': upper_barrier,
        }
    
    def window_barrier(
        self,
        S0, K, rd, rf, local_vol_func, T,
        barrier, barrier_type='down-and-out',
        window_start=0.0, window_end=None,
        option_type='call',
        n_paths=100000, n_steps=500
    ) -> dict:
        """
        Window Barrier: barrier is only active during [window_start, window_end].
        
        Cheaper than full-life barriers. Common in structured products.
        """
        if window_end is None:
            window_end = T
        
        paths = self._simulate_paths(S0, rd, rf, local_vol_func, T, n_paths, n_steps)
        dt = T / n_steps
        
        # Determine which time steps are in the window
        times = np.linspace(0, T, n_steps + 1)
        in_window = (times >= window_start) & (times <= window_end)
        
        # Check barrier only during window
        window_paths = paths[:, in_window]
        
        if barrier_type == 'down-and-out':
            knocked = np.min(window_paths, axis=1) <= barrier
        elif barrier_type == 'up-and-out':
            knocked = np.max(window_paths, axis=1) >= barrier
        else:
            knocked = np.zeros(n_paths, dtype=bool)
        
        alive = ~knocked
        S_final = paths[:, -1]
        
        if option_type == 'call':
            payoff = np.where(alive, np.maximum(S_final - K, 0), 0)
        else:
            payoff = np.where(alive, np.maximum(K - S_final, 0), 0)
        
        price = np.exp(-rd * T) * np.mean(payoff)
        
        # Compare with full-life barrier
        if barrier_type == 'down-and-out':
            full_knocked = np.min(paths, axis=1) <= barrier
        else:
            full_knocked = np.max(paths, axis=1) >= barrier
        full_alive = ~full_knocked
        if option_type == 'call':
            full_payoff = np.where(full_alive, np.maximum(S_final - K, 0), 0)
        else:
            full_payoff = np.where(full_alive, np.maximum(K - S_final, 0), 0)
        full_price = np.exp(-rd * T) * np.mean(full_payoff)
        
        return {
            'window_barrier_price': price,
            'full_barrier_price': full_price,
            'premium_over_full': price - full_price,
            'window': (window_start, window_end),
            'ko_probability_window': np.mean(knocked),
            'ko_probability_full': np.mean(full_knocked),
        }
    
    def digital_option(
        self,
        S0, K, rd, rf, local_vol_func, T,
        payout=1.0,
        option_type='call',
        n_paths=100000, n_steps=200
    ) -> dict:
        """
        Digital (Binary): pays fixed amount if ITM at expiry.
        
        Common in FX structured deposits.
        """
        paths = self._simulate_paths(S0, rd, rf, local_vol_func, T, n_paths, n_steps)
        S_final = paths[:, -1]
        
        if option_type == 'call':
            itm = S_final > K
        else:
            itm = S_final < K
        
        price = np.exp(-rd * T) * payout * np.mean(itm)
        prob_itm = np.mean(itm)
        
        return {
            'price': price,
            'itm_probability': prob_itm,
        }
    
    def best_of_call(
        self,
        S0_1, S0_2, K,
        rd, rf1, rf2,
        vol1, vol2, corr,
        T,
        n_paths=100000, n_steps=200
    ) -> dict:
        """
        Best-of Call: payoff = max(max(S1/S1_0, S2/S2_0) - K, 0).
        
        Multi-asset FX option, traded in exotic desks.
        """
        dt = T / n_steps
        S1 = np.full(n_paths, S0_1)
        S2 = np.full(n_paths, S0_2)
        
        # Cholesky for correlation
        L = np.array([[1.0, 0.0], [corr, np.sqrt(1 - corr**2)]])
        
        for step in range(n_steps):
            eps = self.rng.standard_normal((n_paths, 2))
            Z = eps @ L.T
            
            S1 = S1 * np.exp((rd - rf1 - 0.5*vol1**2)*dt + vol1*np.sqrt(dt)*Z[:, 0])
            S2 = S2 * np.exp((rd - rf2 - 0.5*vol2**2)*dt + vol2*np.sqrt(dt)*Z[:, 1])
        
        perf1 = S1 / S0_1
        perf2 = S2 / S0_2
        best = np.maximum(perf1, perf2)
        payoff = np.maximum(best - K, 0)
        
        price = np.exp(-rd * T) * np.mean(payoff)
        
        return {
            'price': price,
            'avg_best_perf': np.mean(best),
            'prob_asset1_best': np.mean(perf1 > perf2),
        }


if __name__ == "__main__":
    from fx_options_pricer import build_smile_market, DupireLocalVol, gk_call
    
    S0, rd, rf = 1.25, 0.02, 0.01
    maturities, strikes, call_prices, iv_surface = build_smile_market(S0, rd, rf)
    dupire = DupireLocalVol(maturities, strikes, call_prices, rd, rf)
    
    pricer = ExoticFXPricer(seed=42)
    T = 1.0
    
    print("Exotic FX Options Extension")
    print("=" * 55)
    
    # Double No-Touch
    dnt = pricer.double_no_touch(S0, rd, rf, dupire.interp, T,
                                  lower_barrier=1.10, upper_barrier=1.40,
                                  n_paths=100000)
    print(f"\nDOUBLE NO-TOUCH (corridor: {dnt['lower_barrier']:.2f} - {dnt['upper_barrier']:.2f})")
    print(f"Price:     {dnt['price']:.6f}")
    print(f"Survival:  {dnt['survival_probability']:.1%}")
    
    # Window Barrier
    wb = pricer.window_barrier(S0, S0, rd, rf, dupire.interp, T,
                                barrier=0.90*S0, barrier_type='down-and-out',
                                window_start=0.25, window_end=0.75,
                                n_paths=100000)
    print(f"\nWINDOW BARRIER (DO, B={0.90*S0:.3f}, window: 3M-9M)")
    print(f"Window barrier:  {wb['window_barrier_price']:.6f}")
    print(f"Full barrier:    {wb['full_barrier_price']:.6f}")
    print(f"Premium:         {wb['premium_over_full']:.6f}")
    print(f"KO prob (window): {wb['ko_probability_window']:.1%}")
    print(f"KO prob (full):   {wb['ko_probability_full']:.1%}")
    
    # Digital
    dig = pricer.digital_option(S0, S0, rd, rf, dupire.interp, T, payout=1.0)
    print(f"\nDIGITAL CALL (K={S0})")
    print(f"Price:  {dig['price']:.6f}")
    print(f"ITM %:  {dig['itm_probability']:.1%}")
    
    # Best-of
    bo = pricer.best_of_call(1.25, 1.10, 1.0, rd, 0.01, 0.005, 0.20, 0.22, 0.60, T)
    print(f"\nBEST-OF CALL (EUR/USD × GBP/USD)")
    print(f"Price:        {bo['price']:.6f}")
    print(f"Avg best:     {bo['avg_best_perf']:.4f}")
    print(f"P(EUR best):  {bo['prob_asset1_best']:.1%}")
