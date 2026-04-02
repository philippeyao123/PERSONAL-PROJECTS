"""
Extension 2: Wrong-Way Risk (WWR)
===================================
Models the correlation between counterparty exposure and default probability.

Wrong-way risk: exposure increases when the counterparty is more likely to default.
Right-way risk: exposure decreases when the counterparty is more likely to default.

Examples:
  - Bank sells protection on sovereign CDS, counterparty is in same country (WWR)
  - EM currency swap: FX depreciates when EM counterparty credit deteriorates (WWR)
  - Receiving fixed from a bank: rates rise -> exposure rises, bank credit worsens (WWR)

Implementation:
  1. Stochastic hazard rate correlated with market factors
  2. Hull-White hazard rate: dh = a_h*(theta_h - h)dt + sigma_h*dW_h
  3. corr(dW_h, dW_r) = rho_wr (wrong-way risk correlation)

CVA_wwr = E[LGD * DF(tau) * V(tau)+ * 1_{tau<T}]
        ≠ E[LGD * DF(t)] * E[V(t)+]  (no independence!)

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import time


@dataclass
class WWRParams:
    """Wrong-way risk parameters."""
    # Hazard rate dynamics (CIR-like for positivity)
    h0: float = 0.008            # Initial hazard rate (~50bps CDS / 0.6 LGD)
    a_h: float = 0.5             # Mean reversion of hazard rate
    theta_h: float = 0.008       # Long-run hazard rate
    sigma_h: float = 0.003       # Vol of hazard rate
    
    # Wrong-way risk correlation
    rho_wr: float = 0.3          # Correlation between rate and hazard rate
    # Positive rho_wr = when rates rise (exposure up), hazard rises (WWR)
    
    recovery: float = 0.40
    
    @property
    def lgd(self) -> float:
        return 1.0 - self.recovery


class WrongWayRiskEngine:
    """
    Jointly simulate interest rates and counterparty hazard rates.
    
    The key insight: under wrong-way risk, we cannot separate the
    exposure calculation from the credit calculation. We must simulate
    them jointly and compute CVA path-by-path.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def simulate(
        self,
        n_paths: int,
        time_grid: np.ndarray,
        r0: float = 0.035,
        a_r: float = 0.03,
        sigma_r: float = 0.01,
        wwr: WWRParams = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Jointly simulate rates and hazard rates.
        
        Returns:
            rates: (n_paths, n_times)
            hazard_rates: (n_paths, n_times)
            survival_probs: (n_paths, n_times)
        """
        if wwr is None:
            wwr = WWRParams()
        
        n_times = len(time_grid)
        
        # Cholesky for (rate, hazard) correlation
        corr = np.array([[1.0, wwr.rho_wr], [wwr.rho_wr, 1.0]])
        L = np.linalg.cholesky(corr)
        
        rates = np.zeros((n_paths, n_times))
        hazard = np.zeros((n_paths, n_times))
        cum_hazard = np.zeros((n_paths, n_times))
        
        rates[:, 0] = r0
        hazard[:, 0] = wwr.h0
        
        for i in range(1, n_times):
            dt = time_grid[i] - time_grid[i-1]
            sqrt_dt = np.sqrt(dt)
            
            eps = self.rng.standard_normal((n_paths, 2))
            Z = eps @ L.T
            
            # Rate (HW)
            theta_r = a_r * r0
            rates[:, i] = rates[:, i-1] + (theta_r - a_r * rates[:, i-1]) * dt + sigma_r * sqrt_dt * Z[:, 0]
            
            # Hazard rate (CIR-like, floor at 0)
            h_prev = np.maximum(hazard[:, i-1], 1e-8)
            dh = wwr.a_h * (wwr.theta_h - h_prev) * dt + wwr.sigma_h * np.sqrt(h_prev) * sqrt_dt * Z[:, 1]
            hazard[:, i] = np.maximum(hazard[:, i-1] + dh, 1e-8)
            
            # Cumulative hazard (for survival probability)
            cum_hazard[:, i] = cum_hazard[:, i-1] + 0.5 * (hazard[:, i-1] + hazard[:, i]) * dt
        
        survival = np.exp(-cum_hazard)
        
        return rates, hazard, survival
    
    def compute_cva_with_wwr(
        self,
        mtm: np.ndarray,
        survival: np.ndarray,
        hazard: np.ndarray,
        time_grid: np.ndarray,
        lgd: float,
        risk_free_rate: float = 0.035
    ) -> dict:
        """
        Compute CVA with wrong-way risk (no independence assumption).
        
        CVA = sum_i E[LGD * DF(t_i) * max(V(t_i), 0) * (SP(t_{i-1}) - SP(t_i))]
        
        Under WWR, we compute the expectation jointly across paths.
        """
        n_paths, n_times = mtm.shape
        
        cva_wwr = 0.0
        cva_independent = 0.0
        
        for i in range(1, n_times):
            t = time_grid[i]
            dt = time_grid[i] - time_grid[i-1]
            df = np.exp(-risk_free_rate * t)
            
            exposure = np.maximum(mtm[:, i], 0)
            
            # Marginal default probability per path
            pd_path = survival[:, i-1] - survival[:, i]
            pd_path = np.maximum(pd_path, 0)
            
            # WWR CVA: joint expectation (correct)
            cva_wwr += lgd * df * np.mean(exposure * pd_path)
            
            # Independent CVA: E[exposure] * E[pd] (incorrect under WWR)
            cva_independent += lgd * df * np.mean(exposure) * np.mean(pd_path)
        
        wwr_ratio = cva_wwr / max(cva_independent, 1e-10)
        
        return {
            'cva_wwr': cva_wwr,
            'cva_independent': cva_independent,
            'wwr_ratio': wwr_ratio,
            'wwr_adjustment': cva_wwr - cva_independent,
        }


def wwr_correlation_scan(n_paths: int = 30_000) -> dict:
    """Scan CVA across different WWR correlations."""
    from xva_pricer import IRSwap, HullWhiteModel, HullWhiteParams
    
    swap = IRSwap()
    hw = HullWhiteModel(HullWhiteParams())
    time_grid = np.linspace(0, swap.maturity_years, 80)
    
    rho_range = np.arange(-0.5, 0.55, 0.1)
    results = {}
    
    for rho in rho_range:
        wwr_params = WWRParams(rho_wr=rho)
        engine = WrongWayRiskEngine(seed=42)
        
        rates, hazard, survival = engine.simulate(
            n_paths, time_grid,
            r0=0.035, a_r=0.03, sigma_r=0.01,
            wwr=wwr_params
        )
        
        # Compute MtM
        mtm = np.zeros((n_paths, len(time_grid)))
        for i, t in enumerate(time_grid):
            mtm[:, i] = hw.swap_value(rates[:, i], t, swap)
        
        cva_result = engine.compute_cva_with_wwr(
            mtm, survival, hazard, time_grid,
            lgd=wwr_params.lgd
        )
        
        results[rho] = cva_result
    
    return results


if __name__ == "__main__":
    from xva_pricer import IRSwap, HullWhiteModel, HullWhiteParams
    
    print("Wrong-Way Risk Extension")
    print("=" * 50)
    
    swap = IRSwap()
    hw = HullWhiteModel(HullWhiteParams())
    time_grid = np.linspace(0, swap.maturity_years, 80)
    
    n_paths = 50_000
    
    # Base case: rho = 0.3 (wrong-way)
    wwr_params = WWRParams(rho_wr=0.3)
    engine = WrongWayRiskEngine(seed=42)
    
    t0 = time.time()
    rates, hazard, survival = engine.simulate(
        n_paths, time_grid, r0=0.035, a_r=0.03, sigma_r=0.01, wwr=wwr_params
    )
    print(f"Simulation: {time.time()-t0:.2f}s")
    
    # MtM
    mtm = np.zeros((n_paths, len(time_grid)))
    for i, t in enumerate(time_grid):
        mtm[:, i] = hw.swap_value(rates[:, i], t, swap)
    
    result = engine.compute_cva_with_wwr(
        mtm, survival, hazard, time_grid, lgd=wwr_params.lgd
    )
    
    print(f"\nWWR RESULTS (rho = {wwr_params.rho_wr})")
    print(f"CVA (with WWR):     {result['cva_wwr']:>10,.0f}")
    print(f"CVA (independent):  {result['cva_independent']:>10,.0f}")
    print(f"WWR ratio:          {result['wwr_ratio']:>10.2f}x")
    print(f"WWR adjustment:     {result['wwr_adjustment']:>10,.0f}")
    
    # Correlation scan
    print(f"\nWWR CORRELATION SCAN")
    print(f"{'rho':<8} {'CVA_wwr':>10} {'CVA_indep':>10} {'Ratio':>8}")
    print("-" * 40)
    
    scan = wwr_correlation_scan(n_paths=30_000)
    for rho, res in sorted(scan.items()):
        print(f"{rho:<8.1f} {res['cva_wwr']:>10,.0f} {res['cva_independent']:>10,.0f} {res['wwr_ratio']:>7.2f}x")
