"""
Extension 3: Netting & CSA (Credit Support Annex)
====================================================
Models the effect of netting and collateral on exposure and XVA.

Netting: multiple trades with the same counterparty are netted,
reducing exposure from sum(max(V_i, 0)) to max(sum(V_i), 0).

CSA (collateral agreement):
  - Threshold: exposure below this level is not collateralised
  - MTA (Minimum Transfer Amount): minimum collateral call
  - MPOR (Margin Period of Risk): delay between collateral call and receipt
  - Independent Amount (IA): additional collateral buffer

Collateralised exposure:
  E_coll = max(V(t) - C(t-MPOR), 0)

where C is the collateral amount, subject to threshold and MTA.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import time


@dataclass
class CSATerms:
    """Credit Support Annex parameters."""
    threshold: float = 0.0           # Below this, no collateral required
    mta: float = 50_000             # Minimum Transfer Amount
    rounding: float = 10_000        # Rounding amount
    mpor_days: int = 10             # Margin Period of Risk (business days)
    independent_amount: float = 0.0 # Additional buffer
    collateral_currency: str = "USD"
    rehypothecation: bool = True    # Can reuse received collateral
    
    @property
    def mpor_years(self) -> float:
        return self.mpor_days / 252
    
    def describe(self) -> str:
        return (
            f"CSA: Threshold={self.threshold:,.0f}, "
            f"MTA={self.mta:,.0f}, MPOR={self.mpor_days}d, "
            f"IA={self.independent_amount:,.0f}"
        )


@dataclass
class NettingSet:
    """A netting set of trades with the same counterparty."""
    trades: list                     # List of trade MtM arrays
    trade_names: list               # Names for reporting
    csa: Optional[CSATerms] = None  # CSA if collateralised
    
    @property
    def n_trades(self) -> int:
        return len(self.trades)
    
    @property
    def is_collateralised(self) -> bool:
        return self.csa is not None


class NettingEngine:
    """
    Compute netted and collateralised exposure profiles.
    
    Key concepts:
    1. Without netting: exposure = sum of individual positive exposures
    2. With netting: exposure = max(sum of all MtMs, 0)
    3. With CSA: exposure = max(netted_MtM - collateral, 0)
    """
    
    def compute_exposure_no_netting(
        self,
        trades_mtm: List[np.ndarray]
    ) -> np.ndarray:
        """
        Exposure without netting: sum of individual positive exposures.
        E = sum_i max(V_i, 0)
        """
        total = np.zeros_like(trades_mtm[0])
        for mtm in trades_mtm:
            total += np.maximum(mtm, 0)
        return total
    
    def compute_exposure_with_netting(
        self,
        trades_mtm: List[np.ndarray]
    ) -> np.ndarray:
        """
        Exposure with netting: max(sum of all MtMs, 0).
        E = max(sum_i V_i, 0)
        """
        netted = np.zeros_like(trades_mtm[0])
        for mtm in trades_mtm:
            netted += mtm
        return np.maximum(netted, 0)
    
    def compute_collateral(
        self,
        netted_mtm: np.ndarray,
        csa: CSATerms,
        time_grid: np.ndarray
    ) -> np.ndarray:
        """
        Compute collateral held at each time step.
        
        Collateral is called when netted MtM exceeds threshold + MTA.
        There is a delay (MPOR) between call and receipt.
        """
        n_paths, n_times = netted_mtm.shape
        collateral = np.zeros_like(netted_mtm)
        
        # MPOR lag in time steps
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1/252
        mpor_steps = max(1, int(csa.mpor_years / dt))
        
        for i in range(mpor_steps, n_times):
            # Collateral based on MtM at t - MPOR
            lagged_mtm = netted_mtm[:, i - mpor_steps]
            
            # Collateral = max(MtM - threshold, 0), rounded, subject to MTA
            target = np.maximum(lagged_mtm - csa.threshold, 0)
            
            # Apply MTA: only transfer if change exceeds MTA
            delta = target - collateral[:, i-1]
            transfer = np.where(np.abs(delta) >= csa.mta, delta, 0)
            
            collateral[:, i] = collateral[:, i-1] + transfer
            
            # Add independent amount
            collateral[:, i] += csa.independent_amount
            
            # Floor at 0 (can't have negative collateral from our perspective)
            collateral[:, i] = np.maximum(collateral[:, i], 0)
        
        return collateral
    
    def compute_collateralised_exposure(
        self,
        netted_mtm: np.ndarray,
        csa: CSATerms,
        time_grid: np.ndarray
    ) -> dict:
        """
        Compute collateralised exposure = max(netted_MtM - collateral, 0).
        
        Returns exposure profiles with and without collateral for comparison.
        """
        n_paths, n_times = netted_mtm.shape
        
        # Collateral
        collateral = self.compute_collateral(netted_mtm, csa, time_grid)
        
        # Collateralised exposure
        coll_exposure = np.maximum(netted_mtm - collateral, 0)
        
        # Uncollateralised for comparison
        uncoll_exposure = np.maximum(netted_mtm, 0)
        
        return {
            'collateralised_ee': np.mean(coll_exposure, axis=0),
            'uncollateralised_ee': np.mean(uncoll_exposure, axis=0),
            'collateral': np.mean(collateral, axis=0),
            'reduction_pct': 1 - np.mean(coll_exposure) / max(np.mean(uncoll_exposure), 1),
            'ee_coll': np.mean(coll_exposure, axis=0),
            'pfe_coll': np.percentile(coll_exposure, 97.5, axis=0),
        }
    
    def netting_benefit(
        self,
        trades_mtm: List[np.ndarray]
    ) -> dict:
        """Quantify the benefit of netting."""
        no_net = self.compute_exposure_no_netting(trades_mtm)
        with_net = self.compute_exposure_with_netting(trades_mtm)
        
        ee_no_net = np.mean(no_net, axis=0)
        ee_with_net = np.mean(with_net, axis=0)
        
        return {
            'ee_no_netting': ee_no_net,
            'ee_with_netting': ee_with_net,
            'netting_benefit': 1 - np.mean(ee_with_net) / max(np.mean(ee_no_net), 1),
            'peak_reduction': 1 - np.max(ee_with_net) / max(np.max(ee_no_net), 1),
        }


def run_netting_analysis():
    """Full netting and CSA analysis."""
    from xva_pricer import (
        IRSwap, HullWhiteParams, HullWhiteModel,
        ExposureEngine, CreditCurve, XVACalculator
    )
    
    print("Netting & CSA Extension")
    print("=" * 50)
    
    hw = HullWhiteModel(HullWhiteParams())
    time_grid = np.linspace(0, 10, 100)
    n_paths = 50_000
    rates = hw.simulate_rates(n_paths, time_grid, seed=42)
    
    # Create 3 trades in a netting set
    swap1 = IRSwap(notional=10_000_000, fixed_rate=0.035, maturity_years=10, is_payer=True)
    swap2 = IRSwap(notional=8_000_000, fixed_rate=0.030, maturity_years=7, is_payer=False)
    swap3 = IRSwap(notional=5_000_000, fixed_rate=0.040, maturity_years=5, is_payer=True)
    
    # Compute MtM for each trade
    trades_mtm = []
    for swap, name in [(swap1, "10Y Payer"), (swap2, "7Y Receiver"), (swap3, "5Y Payer")]:
        mtm = np.zeros((n_paths, len(time_grid)))
        for i, t in enumerate(time_grid):
            if t < swap.maturity_years:
                mtm[:, i] = hw.swap_value(rates[:, i], t, swap)
        trades_mtm.append(mtm)
        print(f"Trade: {name} {swap.notional/1e6:.0f}M, EE peak: {np.max(np.mean(np.maximum(mtm, 0), axis=0)):,.0f}")
    
    engine = NettingEngine()
    
    # Netting benefit
    netting = engine.netting_benefit(trades_mtm)
    print(f"\nNETTING BENEFIT")
    print(f"EPE without netting:  {np.mean(netting['ee_no_netting']):>12,.0f}")
    print(f"EPE with netting:     {np.mean(netting['ee_with_netting']):>12,.0f}")
    print(f"Netting benefit:      {netting['netting_benefit']:>11.1%}")
    print(f"Peak EE reduction:    {netting['peak_reduction']:>11.1%}")
    
    # CSA analysis
    netted_mtm = sum(trades_mtm)
    
    csa_configs = [
        CSATerms(threshold=0, mta=50_000, mpor_days=10),
        CSATerms(threshold=500_000, mta=100_000, mpor_days=10),
        CSATerms(threshold=1_000_000, mta=250_000, mpor_days=14),
        CSATerms(threshold=0, mta=50_000, mpor_days=10, independent_amount=200_000),
    ]
    
    print(f"\nCSA COMPARISON")
    print(f"{'Config':<35} {'EPE':>10} {'Reduction':>10}")
    print("-" * 57)
    
    # Uncollateralised baseline
    uncoll_epe = np.mean(np.mean(np.maximum(netted_mtm, 0), axis=0))
    print(f"{'No CSA':<35} {uncoll_epe:>10,.0f} {'':>10}")
    
    for csa in csa_configs:
        result = engine.compute_collateralised_exposure(netted_mtm, csa, time_grid)
        coll_epe = np.mean(result['ee_coll'])
        label = f"Thr={csa.threshold/1e3:.0f}K MTA={csa.mta/1e3:.0f}K MPOR={csa.mpor_days}d"
        if csa.independent_amount > 0:
            label += f" IA={csa.independent_amount/1e3:.0f}K"
        print(f"{label:<35} {coll_epe:>10,.0f} {result['reduction_pct']:>9.1%}")
    
    # CVA impact
    print(f"\nCVA IMPACT OF NETTING & CSA")
    cpty = CreditCurve("Cpty", 0.005, 0.40)
    own = CreditCurve("Own", 0.003, 0.40)
    
    # No netting
    no_net_exp = {'ee': netting['ee_no_netting'], 'ene': np.zeros(len(time_grid))}
    xva_no_net = XVACalculator(cpty, own).compute_all(no_net_exp, time_grid)
    
    # With netting
    net_exp = {'ee': netting['ee_with_netting'], 'ene': np.mean(np.minimum(netted_mtm, 0), axis=0)}
    xva_net = XVACalculator(cpty, own).compute_all(net_exp, time_grid)
    
    # With netting + CSA
    csa = CSATerms(threshold=0, mta=50_000, mpor_days=10)
    coll_result = engine.compute_collateralised_exposure(netted_mtm, csa, time_grid)
    coll_exp = {'ee': coll_result['ee_coll'], 'ene': np.zeros(len(time_grid))}
    xva_coll = XVACalculator(cpty, own).compute_all(coll_exp, time_grid)
    
    print(f"{'Setup':<25} {'CVA':>10} {'Reduction':>10}")
    print("-" * 47)
    print(f"{'No netting':<25} {xva_no_net['cva']:>10,.0f} {'':>10}")
    print(f"{'With netting':<25} {xva_net['cva']:>10,.0f} {1-xva_net['cva']/xva_no_net['cva']:>9.1%}")
    print(f"{'Netting + CSA (0 thr)':<25} {xva_coll['cva']:>10,.0f} {1-xva_coll['cva']/xva_no_net['cva']:>9.1%}")
    
    return netting, csa_configs, trades_mtm, time_grid


if __name__ == "__main__":
    run_netting_analysis()
