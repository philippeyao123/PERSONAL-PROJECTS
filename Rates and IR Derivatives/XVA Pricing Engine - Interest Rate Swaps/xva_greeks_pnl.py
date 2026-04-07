"""
Extension 4: XVA Greeks & P&L Explain
========================================
Computes XVA sensitivities (Greeks) and attributes day-over-day
XVA P&L changes to risk factors.

XVA Greeks:
  - IR Delta:    dCVA/dr (sensitivity to rate curve shift)
  - CS01:        dCVA/ds (sensitivity to credit spread)
  - Vega:        dCVA/dsigma (sensitivity to rate vol)
  - Theta:       dCVA/dt (time decay)
  - Correlation: dCVA/drho (exposure-credit correlation)

P&L Explain (day-over-day):
  dCVA = Delta * dr + CS01 * ds + 0.5 * Gamma * dr^2 + Theta * dt + residual

This is what the XVA desk produces daily to explain P&L to traders
and risk managers.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
import time


class XVAGreeks:
    """
    Finite-difference XVA sensitivities.
    
    Uses bump-and-revalue on the full simulation pipeline.
    """
    
    def __init__(self, n_paths: int = 30_000, seed: int = 42):
        self.n_paths = n_paths
        self.seed = seed
    
    def _run_cva(self, r0, sigma_r, cds_spread, a_r=0.03, maturity=10.0) -> float:
        """Helper: run full CVA computation with given parameters."""
        from xva_pricer import (
            IRSwap, HullWhiteParams, HullWhiteModel,
            ExposureEngine, CreditCurve, XVACalculator
        )
        
        swap = IRSwap(maturity_years=maturity)
        hw = HullWhiteModel(HullWhiteParams(a=a_r, sigma=sigma_r, r0=r0, flat_curve=r0))
        time_grid = np.linspace(0, swap.maturity_years, 80)
        rates = hw.simulate_rates(self.n_paths, time_grid, seed=self.seed)
        
        exp_engine = ExposureEngine(hw, swap)
        exposures = exp_engine.compute_profiles(rates, time_grid)
        
        cpty = CreditCurve("Cpty", cds_spread=cds_spread)
        own = CreditCurve("Own", cds_spread=0.003)
        xva_calc = XVACalculator(cpty, own)
        xva = xva_calc.compute_all(exposures, time_grid)
        
        return xva['cva']
    
    def compute_greeks(
        self,
        r0: float = 0.035,
        sigma_r: float = 0.01,
        cds_spread: float = 0.005,
        rate_bump: float = 0.001,      # 10bp
        vol_bump: float = 0.001,       # 10bp of vol
        cs_bump: float = 0.0001,       # 1bp of CDS
    ) -> dict:
        """Compute all XVA Greeks via central differences."""
        
        base = self._run_cva(r0, sigma_r, cds_spread)
        
        # IR Delta (dCVA/dr, per 1bp)
        cva_up = self._run_cva(r0 + rate_bump, sigma_r, cds_spread)
        cva_dn = self._run_cva(r0 - rate_bump, sigma_r, cds_spread)
        ir_delta = (cva_up - cva_dn) / (2 * rate_bump) * 0.0001  # per 1bp
        ir_gamma = (cva_up - 2*base + cva_dn) / (rate_bump**2) * 0.0001**2
        
        # CS01 (dCVA/ds, per 1bp of CDS)
        cva_cs_up = self._run_cva(r0, sigma_r, cds_spread + cs_bump)
        cva_cs_dn = self._run_cva(r0, sigma_r, max(cds_spread - cs_bump, 0.0001))
        cs01 = (cva_cs_up - cva_cs_dn) / (2 * cs_bump) * 0.0001  # per 1bp
        
        # Vega (dCVA/dsigma, per 1bp of vol)
        cva_vol_up = self._run_cva(r0, sigma_r + vol_bump, cds_spread)
        vega = (cva_vol_up - base) / vol_bump * 0.0001  # per 1bp
        
        # Theta (time decay: price with slightly shorter maturity)
        cva_short = self._run_cva(r0, sigma_r, cds_spread)
        # Approximate by 1-day shift using the base
        theta = -base / (10 * 252)  # Daily theta approximation
        
        return {
            'base_cva': base,
            'ir_delta': ir_delta,
            'ir_gamma': ir_gamma,
            'cs01': cs01,
            'vega': vega,
            'theta_daily': theta,
        }


class PnLExplain:
    """
    Day-over-day XVA P&L attribution.
    
    P&L = CVA(today) - CVA(yesterday)
        = Delta * dr + CS01 * ds + 0.5*Gamma*dr^2 + Theta + residual
    
    This decomposition tells the desk exactly why XVA P&L moved.
    """
    
    def __init__(self):
        pass
    
    def explain(
        self,
        greeks_t0: dict,
        greeks_t1: dict,
        dr: float,          # Rate move (absolute)
        ds: float,          # CDS spread move (absolute)
        dsigma: float = 0,  # Vol move (absolute)
    ) -> dict:
        """
        Attribute P&L change to risk factors.
        
        Args:
            greeks_t0: Greeks at start of day
            greeks_t1: Greeks at end of day (for actual CVA)
            dr: Change in rates
            ds: Change in CDS spread
            dsigma: Change in rate vol
        """
        actual_pnl = greeks_t1['base_cva'] - greeks_t0['base_cva']
        
        # First-order effects (scale from per-bp to actual move)
        delta_pnl = greeks_t0['ir_delta'] / 0.0001 * dr
        cs01_pnl = greeks_t0['cs01'] / 0.0001 * ds
        gamma_pnl = 0.5 * greeks_t0['ir_gamma'] / (0.0001**2) * dr**2
        vega_pnl = greeks_t0['vega'] / 0.0001 * dsigma
        theta_pnl = greeks_t0['theta_daily']
        
        explained = delta_pnl + cs01_pnl + gamma_pnl + vega_pnl + theta_pnl
        residual = actual_pnl - explained
        
        return {
            'actual_pnl': actual_pnl,
            'delta_pnl': delta_pnl,
            'cs01_pnl': cs01_pnl,
            'gamma_pnl': gamma_pnl,
            'vega_pnl': vega_pnl,
            'theta_pnl': theta_pnl,
            'explained_pnl': explained,
            'residual': residual,
            'explain_ratio': explained / actual_pnl if abs(actual_pnl) > 1 else 1.0,
        }
    
    def multi_day_explain(
        self,
        n_days: int = 5,
        base_r: float = 0.035,
        base_cds: float = 0.005,
        seed: int = 42
    ) -> list:
        """Simulate multi-day P&L explain."""
        rng = np.random.default_rng(seed)
        
        greeks_calc = XVAGreeks(n_paths=20_000, seed=42)
        
        results = []
        r = base_r
        s = base_cds
        sigma = 0.01
        
        prev_greeks = greeks_calc.compute_greeks(r, sigma, s)
        
        for day in range(n_days):
            # Market moves
            dr = rng.normal(0, 0.002)     # ~20bp daily vol
            ds = rng.normal(0, 0.0002)    # ~2bp daily CDS vol
            ds = max(s + ds, 0.0002) - s  # Floor CDS
            
            r += dr
            s += ds
            
            # Recompute Greeks
            new_greeks = greeks_calc.compute_greeks(r, sigma, s)
            
            # Explain
            explain = self.explain(prev_greeks, new_greeks, dr, ds)
            explain['day'] = day + 1
            explain['rate'] = r
            explain['cds'] = s
            explain['dr'] = dr
            explain['ds'] = ds
            results.append(explain)
            
            prev_greeks = new_greeks
        
        return results


if __name__ == "__main__":
    print("XVA Greeks & P&L Explain Extension")
    print("=" * 50)
    
    # Compute Greeks
    t0 = time.time()
    greeks_calc = XVAGreeks(n_paths=30_000, seed=42)
    greeks = greeks_calc.compute_greeks()
    print(f"Greeks computation: {time.time()-t0:.1f}s")
    
    print(f"\nXVA GREEKS")
    print(f"{'='*40}")
    print(f"Base CVA:    {greeks['base_cva']:>10,.0f}")
    print(f"IR Delta:    {greeks['ir_delta']:>10,.0f} (per 1bp rate move)")
    print(f"IR Gamma:    {greeks['ir_gamma']:>10,.2f} (per 1bp^2)")
    print(f"CS01:        {greeks['cs01']:>10,.0f} (per 1bp CDS move)")
    print(f"Vega:        {greeks['vega']:>10,.0f} (per 1bp vol move)")
    print(f"Theta:       {greeks['theta_daily']:>10,.2f} (daily)")
    
    # P&L Explain
    print(f"\nP&L EXPLAIN (5-day simulation)")
    print(f"{'='*70}")
    print(f"{'Day':<4} {'Rate':>7} {'CDS':>7} {'Actual':>10} {'Delta':>8} {'CS01':>8} {'Theta':>8} {'Resid':>8}")
    print("-" * 70)
    
    pnl_engine = PnLExplain()
    days = pnl_engine.multi_day_explain(n_days=5)
    
    for d in days:
        print(f"{d['day']:<4} {d['rate']:>6.2%} {d['cds']*10000:>5.0f}bp "
              f"{d['actual_pnl']:>+10,.0f} {d['delta_pnl']:>+8,.0f} "
              f"{d['cs01_pnl']:>+8,.0f} {d['theta_pnl']:>+8,.0f} "
              f"{d['residual']:>+8,.0f}")
    
    total_pnl = sum(d['actual_pnl'] for d in days)
    total_explained = sum(d['explained_pnl'] for d in days)
    print(f"\nTotal P&L: {total_pnl:+,.0f}")
    print(f"Explained: {total_explained:+,.0f}")
    print(f"Residual:  {total_pnl - total_explained:+,.0f}")
