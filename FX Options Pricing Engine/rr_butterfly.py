"""
Extension 4: Risk Reversal & Butterfly Analysis
==================================================
FX-specific smile conventions used by every FX options desk:

  Risk Reversal (RR): IV(25d call) - IV(25d put) = measures skew
  Butterfly (BF):     0.5*(IV(25d call) + IV(25d put)) - IV(ATM) = measures curvature
  
  ATM convention: delta-neutral straddle (dns) or forward (atm_fwd)

The FX market quotes smiles in (ATM, RR, BF) rather than individual strikes.
Converting between the two is essential for any FX quant.

This module:
1. Converts between strike-space and delta-space quotes
2. Analyses smile dynamics: how RR/BF move with spot and time
3. Builds term structures of RR and BF

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict, Tuple
import time


def gk_delta(S0, K, T, rd, rf, sigma, option_type='call'):
    """GK delta (spot delta, premium-excluded)."""
    d1 = (np.log(S0 / K) + (rd - rf + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return np.exp(-rf * T) * norm.cdf(d1)
    return np.exp(-rf * T) * (norm.cdf(d1) - 1)


def strike_from_delta(S0, T, rd, rf, sigma, target_delta, option_type='call'):
    """Find strike K that gives the target delta."""
    def obj(K):
        return gk_delta(S0, K, T, rd, rf, sigma, option_type) - target_delta
    
    try:
        return brentq(obj, S0 * 0.5, S0 * 2.0)
    except ValueError:
        return S0


class FXSmileConventions:
    """
    Convert between strike-space and FX market conventions.
    
    FX desks quote:
    - ATM vol (delta-neutral straddle)
    - 25-delta Risk Reversal (25d call IV - 25d put IV)
    - 25-delta Butterfly (avg of 25d wings - ATM)
    - Optionally: 10-delta RR and BF
    """
    
    def __init__(self, S0, rd, rf):
        self.S0 = S0
        self.rd = rd
        self.rf = rf
    
    def quotes_to_strikes(
        self,
        T: float,
        atm_vol: float,
        rr_25d: float,
        bf_25d: float,
        rr_10d: float = None,
        bf_10d: float = None
    ) -> dict:
        """
        Convert (ATM, RR, BF) quotes to (strike, IV) pairs.
        
        ATM: delta-neutral straddle
        25d call IV = ATM + BF + 0.5*RR
        25d put IV  = ATM + BF - 0.5*RR
        """
        # 25-delta wing vols
        iv_25d_call = atm_vol + bf_25d + 0.5 * rr_25d
        iv_25d_put = atm_vol + bf_25d - 0.5 * rr_25d
        
        # ATM strike (delta-neutral straddle: call_delta + put_delta = 0)
        K_atm = self.S0 * np.exp((self.rd - self.rf + 0.5 * atm_vol**2) * T)
        
        # 25-delta strikes
        K_25d_call = strike_from_delta(self.S0, T, self.rd, self.rf, iv_25d_call, 0.25, 'call')
        K_25d_put = strike_from_delta(self.S0, T, self.rd, self.rf, iv_25d_put, -0.25, 'put')
        
        result = {
            'K_atm': K_atm,
            'iv_atm': atm_vol,
            'K_25d_call': K_25d_call,
            'iv_25d_call': iv_25d_call,
            'K_25d_put': K_25d_put,
            'iv_25d_put': iv_25d_put,
        }
        
        # 10-delta if provided
        if rr_10d is not None and bf_10d is not None:
            iv_10d_call = atm_vol + bf_10d + 0.5 * rr_10d
            iv_10d_put = atm_vol + bf_10d - 0.5 * rr_10d
            K_10d_call = strike_from_delta(self.S0, T, self.rd, self.rf, iv_10d_call, 0.10, 'call')
            K_10d_put = strike_from_delta(self.S0, T, self.rd, self.rf, iv_10d_put, -0.10, 'put')
            result.update({
                'K_10d_call': K_10d_call,
                'iv_10d_call': iv_10d_call,
                'K_10d_put': K_10d_put,
                'iv_10d_put': iv_10d_put,
            })
        
        return result
    
    def strikes_to_quotes(
        self,
        T: float,
        strikes: np.ndarray,
        ivs: np.ndarray
    ) -> dict:
        """Extract (ATM, RR, BF) from strike-space IV curve."""
        F = self.S0 * np.exp((self.rd - self.rf) * T)
        
        # ATM: closest to forward
        atm_idx = np.argmin(np.abs(strikes - F))
        atm_vol = ivs[atm_idx]
        
        # Find 25-delta strikes by interpolation
        deltas = np.array([
            gk_delta(self.S0, K, T, self.rd, self.rf, iv, 'call')
            for K, iv in zip(strikes, ivs)
        ])
        
        # 25-delta call: find IV where call delta = 0.25
        if np.any(deltas < 0.25) and np.any(deltas > 0.25):
            iv_25c = np.interp(0.25, deltas[::-1], ivs[::-1])  # Delta decreasing with K
        else:
            iv_25c = atm_vol + 0.01
        
        # 25-delta put
        put_deltas = deltas - np.exp(-self.rf * T)  # put delta = call delta - exp(-rf*T)
        if np.any(put_deltas < -0.25) and np.any(put_deltas > -0.25):
            iv_25p = np.interp(-0.25, put_deltas, ivs)
        else:
            iv_25p = atm_vol + 0.01
        
        rr_25d = iv_25c - iv_25p
        bf_25d = 0.5 * (iv_25c + iv_25p) - atm_vol
        
        return {
            'atm_vol': atm_vol,
            'rr_25d': rr_25d,
            'bf_25d': bf_25d,
            'iv_25d_call': iv_25c,
            'iv_25d_put': iv_25p,
        }


class SmileDynamicsAnalyser:
    """
    Analyse how the FX smile behaves:
    - Sticky strike: IV at fixed K doesn't change when spot moves
    - Sticky delta: IV at fixed delta doesn't change
    - Sticky moneyness: IV at fixed K/S doesn't change
    
    Also: term structure of RR and BF.
    """
    
    @staticmethod
    def term_structure_rr_bf(
        S0, rd, rf,
        maturities, strikes, iv_surface
    ) -> dict:
        """Compute RR and BF term structure."""
        conv = FXSmileConventions(S0, rd, rf)
        
        rr_ts = []
        bf_ts = []
        atm_ts = []
        
        for i, T in enumerate(maturities):
            quotes = conv.strikes_to_quotes(T, strikes, iv_surface[i])
            rr_ts.append(quotes['rr_25d'])
            bf_ts.append(quotes['bf_25d'])
            atm_ts.append(quotes['atm_vol'])
        
        return {
            'maturities': maturities,
            'atm': np.array(atm_ts),
            'rr_25d': np.array(rr_ts),
            'bf_25d': np.array(bf_ts),
        }
    
    @staticmethod
    def smile_sensitivity(S0, K, T, rd, rf, base_vol, skew, convexity):
        """How smile changes with spot (sticky strike vs sticky delta)."""
        spots = np.linspace(S0 * 0.90, S0 * 1.10, 11)
        
        results = {'spots': spots, 'iv_sticky_strike': [], 'iv_sticky_delta': []}
        
        F_base = S0 * np.exp((rd - rf) * T)
        log_m_base = np.log(K / F_base)
        iv_base = base_vol + skew * log_m_base + convexity * log_m_base**2
        
        for S in spots:
            # Sticky strike: IV at same K
            F = S * np.exp((rd - rf) * T)
            log_m = np.log(K / F)
            iv_ss = base_vol + skew * log_m + convexity * log_m**2
            results['iv_sticky_strike'].append(iv_ss)
            
            # Sticky delta: IV at same moneyness (K/S = const)
            K_adj = K * S / S0
            log_m_adj = np.log(K_adj / F)
            iv_sd = base_vol + skew * log_m_adj + convexity * log_m_adj**2
            results['iv_sticky_delta'].append(iv_sd)
        
        results['iv_sticky_strike'] = np.array(results['iv_sticky_strike'])
        results['iv_sticky_delta'] = np.array(results['iv_sticky_delta'])
        
        return results


if __name__ == "__main__":
    from fx_options_pricer import build_smile_market
    
    S0, rd, rf = 1.25, 0.02, 0.01
    maturities, strikes, call_prices, iv_surface = build_smile_market(
        S0, rd, rf, base_vol=0.20, skew=-0.03, convexity=0.02
    )
    
    print("Risk Reversal & Butterfly Analysis")
    print("=" * 55)
    
    conv = FXSmileConventions(S0, rd, rf)
    
    # Convert market quotes
    print(f"\nSMILE CONVENTIONS (1Y)")
    T = 1.0
    quotes = conv.strikes_to_quotes(T, strikes, iv_surface[2])
    print(f"ATM vol:      {quotes['atm_vol']:.2%}")
    print(f"25d RR:       {quotes['rr_25d']:+.2%}")
    print(f"25d BF:       {quotes['bf_25d']:+.2%}")
    print(f"25d call IV:  {quotes['iv_25d_call']:.2%}")
    print(f"25d put IV:   {quotes['iv_25d_put']:.2%}")
    
    # Term structure
    print(f"\nTERM STRUCTURE")
    ts = SmileDynamicsAnalyser.term_structure_rr_bf(S0, rd, rf, maturities, strikes, iv_surface)
    print(f"{'Tenor':<8} {'ATM':>8} {'RR 25d':>8} {'BF 25d':>8}")
    print("-" * 35)
    for i, T in enumerate(maturities):
        print(f"{T:<8.2f} {ts['atm'][i]:>7.2%} {ts['rr_25d'][i]:>+7.2%} {ts['bf_25d'][i]:>+7.2%}")
    
    # Quote to strike conversion
    print(f"\nQUOTE → STRIKE CONVERSION (1Y)")
    result = conv.quotes_to_strikes(
        T=1.0, atm_vol=0.20, rr_25d=-0.015, bf_25d=0.008,
        rr_10d=-0.025, bf_10d=0.020
    )
    for k, v in result.items():
        if 'K_' in k:
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v:.2%}")
    
    # Smile dynamics
    print(f"\nSMILE DYNAMICS (ATM strike, 1Y)")
    dynamics = SmileDynamicsAnalyser.smile_sensitivity(S0, S0, 1.0, rd, rf, 0.20, -0.03, 0.02)
    print(f"{'Spot':<8} {'Sticky K':>10} {'Sticky Δ':>10}")
    print("-" * 30)
    for i in range(0, len(dynamics['spots']), 2):
        s = dynamics['spots'][i]
        print(f"{s:<8.3f} {dynamics['iv_sticky_strike'][i]:>9.2%} {dynamics['iv_sticky_delta'][i]:>9.2%}")
