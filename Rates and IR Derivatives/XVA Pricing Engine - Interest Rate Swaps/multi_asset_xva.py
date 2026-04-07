"""
Extension 1: Multi-Asset XVA — Cross-Currency Swap
====================================================
Extends the XVA engine to a cross-currency interest rate swap (XCCY),
which has both IR and FX exposure.

A XCCY swap exchanges:
  - Fixed/floating payments in currency 1 (e.g. EUR)
  - Fixed/floating payments in currency 2 (e.g. USD)
  - Notional exchange at maturity (FX risk)

The exposure profile is fundamentally different from a single-currency IRS:
  - FX component creates large exposure near maturity (notional exchange)
  - IR and FX risks interact, creating fatter tails
  - Wrong-way risk is natural: EM currency depreciation + credit deterioration

Simulation: correlated Hull-White rates (2 currencies) + GBM FX.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import time


@dataclass
class XCCYSwap:
    """Cross-currency interest rate swap."""
    notional_dom: float = 10_000_000   # Domestic (EUR) notional
    notional_for: float = 11_000_000   # Foreign (USD) notional
    fixed_rate_dom: float = 0.030      # EUR fixed rate
    fixed_rate_for: float = 0.040      # USD fixed rate
    maturity_years: float = 10.0
    payment_frequency: int = 2
    fx_initial: float = 1.10           # EUR/USD spot
    
    @property
    def payment_times(self) -> np.ndarray:
        n = int(self.maturity_years * self.payment_frequency)
        return np.linspace(1/self.payment_frequency, self.maturity_years, n)
    
    @property
    def dt(self) -> float:
        return 1.0 / self.payment_frequency


@dataclass
class MultiCurrencyParams:
    """Parameters for 2-currency + FX simulation."""
    # Domestic (EUR) HW params
    a_dom: float = 0.03
    sigma_dom: float = 0.008
    r0_dom: float = 0.030
    
    # Foreign (USD) HW params
    a_for: float = 0.04
    sigma_for: float = 0.010
    r0_for: float = 0.040
    
    # FX (EUR/USD)
    fx_vol: float = 0.08
    fx_spot: float = 1.10
    
    # Correlations
    corr_dom_for: float = 0.40     # EUR rate vs USD rate
    corr_dom_fx: float = -0.20     # EUR rate vs EUR/USD
    corr_for_fx: float = 0.30      # USD rate vs EUR/USD


class MultiCurrencyEngine:
    """
    Simulate correlated domestic rate, foreign rate, and FX.
    
    3-factor model:
      dr_dom = [theta_dom - a_dom * r_dom] dt + sigma_dom dW1
      dr_for = [theta_for - a_for * r_for] dt + sigma_for dW2
      dS/S   = (r_dom - r_for) dt + sigma_fx dW3
    
    with correlated Brownians via Cholesky.
    """
    
    def __init__(self, params: MultiCurrencyParams, seed: int = 42):
        self.p = params
        self.rng = np.random.default_rng(seed)
        
        # Build correlation matrix
        corr = np.array([
            [1.0, params.corr_dom_for, params.corr_dom_fx],
            [params.corr_dom_for, 1.0, params.corr_for_fx],
            [params.corr_dom_fx, params.corr_for_fx, 1.0]
        ])
        self.L = np.linalg.cholesky(corr)
    
    def simulate(
        self,
        n_paths: int,
        time_grid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            r_dom: (n_paths, n_times) domestic short rates
            r_for: (n_paths, n_times) foreign short rates
            fx:    (n_paths, n_times) FX rate (DOM/FOR)
        """
        p = self.p
        n_times = len(time_grid)
        
        r_dom = np.zeros((n_paths, n_times))
        r_for = np.zeros((n_paths, n_times))
        log_fx = np.zeros((n_paths, n_times))
        
        r_dom[:, 0] = p.r0_dom
        r_for[:, 0] = p.r0_for
        log_fx[:, 0] = np.log(p.fx_spot)
        
        for i in range(1, n_times):
            dt = time_grid[i] - time_grid[i-1]
            sqrt_dt = np.sqrt(dt)
            
            eps = self.rng.standard_normal((n_paths, 3))
            Z = eps @ self.L.T
            
            # Domestic rate
            theta_dom = p.a_dom * p.r0_dom
            r_dom[:, i] = r_dom[:, i-1] + (theta_dom - p.a_dom * r_dom[:, i-1]) * dt + p.sigma_dom * sqrt_dt * Z[:, 0]
            
            # Foreign rate
            theta_for = p.a_for * p.r0_for
            r_for[:, i] = r_for[:, i-1] + (theta_for - p.a_for * r_for[:, i-1]) * dt + p.sigma_for * sqrt_dt * Z[:, 1]
            
            # FX (risk-neutral drift = r_dom - r_for)
            drift_fx = (r_dom[:, i-1] - r_for[:, i-1] - 0.5 * p.fx_vol**2) * dt
            log_fx[:, i] = log_fx[:, i-1] + drift_fx + p.fx_vol * sqrt_dt * Z[:, 2]
        
        fx = np.exp(log_fx)
        return r_dom, r_for, fx
    
    def xccy_swap_value(
        self,
        r_dom: np.ndarray,
        r_for: np.ndarray,
        fx: np.ndarray,
        t: float,
        swap: XCCYSwap,
        time_idx: int
    ) -> np.ndarray:
        """
        Value XCCY swap at time t.
        
        MtM = PV(domestic leg) - FX(t) * PV(foreign leg) + notional exchange
        """
        remaining = swap.payment_times[swap.payment_times > t + 1e-10]
        if len(remaining) == 0:
            return np.zeros(r_dom.shape[0])
        
        n_paths = r_dom.shape[0]
        p = self.p
        
        # Domestic leg PV (receive fixed EUR)
        pv_dom = np.zeros(n_paths)
        for T_i in remaining:
            tau = T_i - t
            B_dom = (1 - np.exp(-p.a_dom * tau)) / p.a_dom
            P_dom = np.exp(-B_dom * r_dom[:, time_idx] - p.r0_dom * tau + B_dom * p.r0_dom)
            pv_dom += swap.dt * swap.fixed_rate_dom * swap.notional_dom * P_dom
        
        # Add domestic notional at maturity
        tau_mat = swap.maturity_years - t
        if tau_mat > 0:
            B_mat = (1 - np.exp(-p.a_dom * tau_mat)) / p.a_dom
            P_mat_dom = np.exp(-B_mat * r_dom[:, time_idx] - p.r0_dom * tau_mat + B_mat * p.r0_dom)
            pv_dom += swap.notional_dom * P_mat_dom
        
        # Foreign leg PV (pay fixed USD, converted to EUR)
        pv_for = np.zeros(n_paths)
        for T_i in remaining:
            tau = T_i - t
            B_for = (1 - np.exp(-p.a_for * tau)) / p.a_for
            P_for = np.exp(-B_for * r_for[:, time_idx] - p.r0_for * tau + B_for * p.r0_for)
            pv_for += swap.dt * swap.fixed_rate_for * swap.notional_for * P_for
        
        # Foreign notional at maturity
        if tau_mat > 0:
            B_mat_for = (1 - np.exp(-p.a_for * tau_mat)) / p.a_for
            P_mat_for = np.exp(-B_mat_for * r_for[:, time_idx] - p.r0_for * tau_mat + B_mat_for * p.r0_for)
            pv_for += swap.notional_for * P_mat_for
        
        # Convert foreign PV to domestic via current FX
        fx_t = fx[:, time_idx]
        pv_for_in_dom = pv_for / fx_t  # USD -> EUR
        
        # Net: receive DOM, pay FOR
        mtm = pv_dom - pv_for_in_dom
        
        return mtm


def run_xccy_xva():
    """Run full XCCY XVA analysis."""
    from xva_pricer import CreditCurve, XVACalculator
    
    swap = XCCYSwap()
    params = MultiCurrencyParams()
    engine = MultiCurrencyEngine(params, seed=42)
    
    n_paths = 50_000
    time_grid = np.linspace(0, swap.maturity_years, 100)
    
    print("Multi-Asset XVA — Cross-Currency Swap")
    print("=" * 50)
    print(f"DOM: EUR {swap.notional_dom:,.0f} at {swap.fixed_rate_dom:.1%}")
    print(f"FOR: USD {swap.notional_for:,.0f} at {swap.fixed_rate_for:.1%}")
    print(f"FX: EUR/USD = {swap.fx_initial}")
    
    t0 = time.time()
    r_dom, r_for, fx = engine.simulate(n_paths, time_grid)
    print(f"\nSimulation: {time.time()-t0:.2f}s ({n_paths:,} paths)")
    
    # Compute MtM at each time
    mtm = np.zeros((n_paths, len(time_grid)))
    for i, t in enumerate(time_grid):
        mtm[:, i] = engine.xccy_swap_value(r_dom, r_for, fx, t, swap, i)
    
    ee = np.mean(np.maximum(mtm, 0), axis=0)
    ene = np.mean(np.minimum(mtm, 0), axis=0)
    pfe = np.percentile(np.maximum(mtm, 0), 97.5, axis=0)
    
    print(f"\nEXPOSURE PROFILES (XCCY)")
    print(f"Peak EE:       {np.max(ee):>12,.0f}")
    print(f"Peak PFE 97.5: {np.max(pfe):>12,.0f}")
    print(f"EPE:           {np.mean(ee):>12,.0f}")
    
    # XVA
    exposures = {'ee': ee, 'ene': ene}
    cpty = CreditCurve("Cpty", 0.005, 0.40)
    own = CreditCurve("Own", 0.003, 0.40)
    xva_calc = XVACalculator(cpty, own, funding_spread=0.002)
    xva = xva_calc.compute_all(exposures, time_grid)
    
    N = swap.notional_dom
    print(f"\nXVA RESULTS (XCCY)")
    print(f"CVA:  {xva['cva']:>10,.0f} ({xva['cva']/N*10000:.1f} bps)")
    print(f"DVA:  {xva['dva']:>10,.0f} ({xva['dva']/N*10000:.1f} bps)")
    print(f"FVA:  {xva['fva']:>10,.0f} ({xva['fva']/N*10000:.1f} bps)")
    print(f"Total:{xva['total_xva']:>10,.0f} ({xva['total_xva']/N*10000:.1f} bps)")
    
    # Compare with single-currency IRS
    print(f"\nNote: XCCY exposure is much larger than IRS due to")
    print(f"notional exchange at maturity creating FX risk.")
    
    return mtm, ee, ene, pfe, xva, time_grid, fx


if __name__ == "__main__":
    run_xccy_xva()
