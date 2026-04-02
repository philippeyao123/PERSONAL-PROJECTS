"""
XVA Pricing Engine for Interest Rate Swaps
============================================
Full post-trade valuation adjustment framework:

  CVA  = Credit Valuation Adjustment (counterparty default risk)
  DVA  = Debit Valuation Adjustment (own default risk)
  FVA  = Funding Valuation Adjustment (cost of funding uncollateralised exposure)
  ColVA = Collateral Valuation Adjustment (cost of posting collateral)
  KVA  = Capital Valuation Adjustment (cost of regulatory capital)
  MVA  = Margin Valuation Adjustment (cost of initial margin / SIMM)

Pipeline:
  1. Simulate interest rate paths (Hull-White 1F)
  2. Price the IRS at each future time step on each path
  3. Compute exposure profiles (EE, ENE, EPE, PFE)
  4. Overlay credit, funding, and capital to get XVA

Relevant for: UBS XVA/SIMM, GS FICCS Strats, JPM MRM, any desk strat role.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
from scipy.stats import norm
import time


# ============================================================================
# Interest Rate Swap
# ============================================================================

@dataclass
class IRSwap:
    """Plain vanilla interest rate swap specification."""
    notional: float = 10_000_000     # 10M notional
    fixed_rate: float = 0.035        # 3.5% fixed
    maturity_years: float = 10.0     # 10Y swap
    payment_frequency: int = 2       # Semi-annual
    is_payer: bool = True            # Payer = pay fixed, receive floating
    
    @property
    def n_payments(self) -> int:
        return int(self.maturity_years * self.payment_frequency)
    
    @property
    def payment_times(self) -> np.ndarray:
        return np.linspace(
            1 / self.payment_frequency,
            self.maturity_years,
            self.n_payments
        )
    
    @property
    def dt(self) -> float:
        return 1.0 / self.payment_frequency
    
    def describe(self) -> str:
        direction = "Payer (pay fixed)" if self.is_payer else "Receiver (receive fixed)"
        return (
            f"IRS: {direction}\n"
            f"Notional:    {self.notional:,.0f}\n"
            f"Fixed rate:  {self.fixed_rate:.2%}\n"
            f"Maturity:    {self.maturity_years}Y\n"
            f"Frequency:   {'Semi-annual' if self.payment_frequency == 2 else 'Quarterly'}\n"
            f"Payments:    {self.n_payments}\n"
        )


# ============================================================================
# Hull-White 1-Factor Model
# ============================================================================

@dataclass
class HullWhiteParams:
    """Hull-White 1-factor short rate model parameters."""
    a: float = 0.03          # Mean reversion speed
    sigma: float = 0.01      # Short rate volatility
    r0: float = 0.035        # Initial short rate
    
    # Initial yield curve (flat for simplicity, can be extended)
    flat_curve: float = 0.035


class HullWhiteModel:
    """
    Hull-White 1-factor short rate model:
        dr(t) = [theta(t) - a*r(t)] dt + sigma dW(t)
    
    For a flat initial curve:
        theta(t) = a * r0 + sigma^2 / (2*a) * (1 - exp(-2*a*t))
    
    Provides:
    - Short rate simulation
    - Zero-coupon bond pricing: P(t,T) = A(t,T) * exp(-B(t,T)*r(t))
    - Swap valuation at future dates
    """
    
    def __init__(self, params: HullWhiteParams):
        self.p = params
    
    def B(self, t: float, T: float) -> float:
        """B(t,T) in the HW bond price formula."""
        a = self.p.a
        if abs(a) < 1e-10:
            return T - t
        return (1 - np.exp(-a * (T - t))) / a
    
    def A(self, t: float, T: float) -> float:
        """A(t,T) in the HW bond price formula (flat curve)."""
        a = self.p.a
        sigma = self.p.sigma
        r0 = self.p.flat_curve
        
        B_val = self.B(t, T)
        tau = T - t
        
        # For flat initial curve
        P_0_T = np.exp(-r0 * T)
        P_0_t = np.exp(-r0 * t) if t > 0 else 1.0
        
        log_A = np.log(P_0_T / P_0_t) + B_val * r0 * t / max(t, 1e-10) * 0
        # Simplified: use exact formula for flat curve
        log_A = (-r0 * tau + r0 * B_val 
                 - sigma**2 / (4 * a) * B_val**2 * (1 - np.exp(-2 * a * t)))
        
        return np.exp(log_A)
    
    def zero_coupon_bond(self, r: np.ndarray, t: float, T: float) -> np.ndarray:
        """Price of zero-coupon bond P(t,T) given short rate r(t)."""
        B_val = self.B(t, T)
        
        # Use simplified affine formula
        a = self.p.a
        sigma = self.p.sigma
        r0 = self.p.flat_curve
        tau = T - t
        
        # Forward rate component
        f_0_t = r0  # Flat curve
        
        # Variance component
        var_term = sigma**2 / (2 * a**2) * (1 - np.exp(-a * tau))**2 * (1 - np.exp(-2 * a * t)) / (2 * a) if t > 0 else 0
        
        return np.exp(-B_val * r - r0 * tau + B_val * r0 - var_term)
    
    def simulate_rates(
        self,
        n_paths: int,
        time_grid: np.ndarray,
        seed: int = 42
    ) -> np.ndarray:
        """
        Simulate short rate paths under the risk-neutral measure.
        
        Returns:
            rates: (n_paths, n_times) short rates
        """
        rng = np.random.default_rng(seed)
        n_times = len(time_grid)
        
        rates = np.zeros((n_paths, n_times))
        rates[:, 0] = self.p.r0
        
        a = self.p.a
        sigma = self.p.sigma
        r0 = self.p.flat_curve
        
        for i in range(1, n_times):
            dt = time_grid[i] - time_grid[i-1]
            sqrt_dt = np.sqrt(dt)
            
            # Theta for flat curve
            theta = a * r0 + sigma**2 / (2 * a) * (1 - np.exp(-2 * a * time_grid[i]))
            
            # Euler discretisation
            dW = rng.standard_normal(n_paths)
            rates[:, i] = (rates[:, i-1] 
                          + (theta - a * rates[:, i-1]) * dt 
                          + sigma * sqrt_dt * dW)
        
        return rates
    
    def swap_value(
        self,
        r: np.ndarray,
        t: float,
        swap: IRSwap
    ) -> np.ndarray:
        """
        Value an IRS at time t given short rates r(t).
        
        MtM = Notional * sum_{i: T_i > t} [delta * (forward_rate - K) * P(t, T_i)]
        
        Simplified: use discount factors from HW model.
        """
        remaining_payments = swap.payment_times[swap.payment_times > t + 1e-10]
        if len(remaining_payments) == 0:
            return np.zeros_like(r)
        
        dt_pay = swap.dt
        K = swap.fixed_rate
        N = swap.notional
        
        # Annuity (sum of discount factors)
        annuity = np.zeros_like(r)
        for T_i in remaining_payments:
            # ZCB price at each path
            B_val = self.B(t, T_i)
            tau = T_i - t
            P = np.exp(-B_val * r - self.p.flat_curve * tau + B_val * self.p.flat_curve)
            annuity += dt_pay * P
        
        # Par swap rate (approximate)
        T_first = remaining_payments[0]
        T_last = remaining_payments[-1]
        
        B_first = self.B(t, T_first)
        B_last = self.B(t, T_last)
        
        P_first = np.exp(-B_first * r - self.p.flat_curve * (T_first - t) + B_first * self.p.flat_curve)
        P_last = np.exp(-B_last * r - self.p.flat_curve * (T_last - t) + B_last * self.p.flat_curve)
        
        # Swap rate = (P_first - P_last) / Annuity
        swap_rate = np.where(annuity > 1e-12, (P_first - P_last) / annuity, K)
        
        # MtM
        if swap.is_payer:
            mtm = N * (swap_rate - K) * annuity
        else:
            mtm = N * (K - swap_rate) * annuity
        
        return mtm


# ============================================================================
# Exposure Profiles
# ============================================================================

class ExposureEngine:
    """
    Compute exposure profiles from simulated swap MtM paths.
    
    Key profiles:
    - EE (Expected Exposure): E[max(V, 0)]
    - ENE (Expected Negative Exposure): E[min(V, 0)]
    - EPE (Expected Positive Exposure): time-averaged EE
    - PFE (Potential Future Exposure): quantile of max(V, 0)
    """
    
    def __init__(self, hw_model: HullWhiteModel, swap: IRSwap):
        self.model = hw_model
        self.swap = swap
    
    def compute_profiles(
        self,
        rates: np.ndarray,
        time_grid: np.ndarray
    ) -> dict:
        """Compute all exposure profiles."""
        n_paths, n_times = rates.shape
        
        mtm = np.zeros((n_paths, n_times))
        for i, t in enumerate(time_grid):
            mtm[:, i] = self.model.swap_value(rates[:, i], t, self.swap)
        
        # Exposure = max(MtM, 0) from our perspective
        positive_exposure = np.maximum(mtm, 0)
        negative_exposure = np.minimum(mtm, 0)
        
        # Profiles
        ee = np.mean(positive_exposure, axis=0)     # Expected Exposure
        ene = np.mean(negative_exposure, axis=0)     # Expected Negative Exposure
        pfe_97_5 = np.percentile(positive_exposure, 97.5, axis=0)
        pfe_99 = np.percentile(positive_exposure, 99, axis=0)
        
        # EPE (time-average of EE)
        epe = np.mean(ee)
        
        # Effective EPE (non-decreasing EE)
        eepe = np.maximum.accumulate(ee)
        effective_epe = np.mean(eepe)
        
        return {
            'mtm': mtm,
            'ee': ee,
            'ene': ene,
            'epe': epe,
            'effective_epe': effective_epe,
            'pfe_97_5': pfe_97_5,
            'pfe_99': pfe_99,
            'positive_exposure': positive_exposure,
            'negative_exposure': negative_exposure,
            'eepe': eepe,
        }


# ============================================================================
# Credit Model (Hazard Rates)
# ============================================================================

@dataclass
class CreditCurve:
    """CDS-implied hazard rate curve."""
    name: str = "Counterparty"
    cds_spread: float = 0.005    # 50bps flat CDS spread
    recovery: float = 0.40
    
    @property
    def lgd(self) -> float:
        return 1.0 - self.recovery
    
    @property
    def hazard_rate(self) -> float:
        return self.cds_spread / self.lgd
    
    def survival_prob(self, t: float) -> float:
        return np.exp(-self.hazard_rate * t)
    
    def default_prob(self, t1: float, t2: float) -> float:
        """Marginal default probability between t1 and t2."""
        return self.survival_prob(t1) - self.survival_prob(t2)


# ============================================================================
# XVA Calculator
# ============================================================================

class XVACalculator:
    """
    Compute all valuation adjustments.
    
    CVA  = sum_i DF(t_i) * LGD_c * SP_c(t_{i-1}) * E[max(V(t_i), 0)]
    DVA  = sum_i DF(t_i) * LGD_o * SP_o(t_{i-1}) * E[max(-V(t_i), 0)]
    FVA  = sum_i DF(t_i) * s_f * dt * E[V(t_i)]
    KVA  = sum_i DF(t_i) * h_k * K(t_i) * dt
    MVA  = sum_i DF(t_i) * s_f * IM(t_i) * dt
    """
    
    def __init__(
        self,
        counterparty: CreditCurve,
        own_credit: CreditCurve,
        funding_spread: float = 0.002,  # 20bps funding spread
        capital_hurdle: float = 0.10,    # 10% return on capital
        risk_free_rate: float = 0.035
    ):
        self.cpty = counterparty
        self.own = own_credit
        self.s_f = funding_spread
        self.h_k = capital_hurdle
        self.r = risk_free_rate
    
    def compute_all(
        self,
        exposures: dict,
        time_grid: np.ndarray,
        simm_profile: np.ndarray = None,
        capital_profile: np.ndarray = None
    ) -> dict:
        """Compute all XVA measures."""
        ee = exposures['ee']
        ene = exposures['ene']
        n_times = len(time_grid)
        
        cva = 0.0
        dva = 0.0
        fva = 0.0
        kva = 0.0
        mva = 0.0
        
        cva_profile = np.zeros(n_times)
        dva_profile = np.zeros(n_times)
        fva_profile = np.zeros(n_times)
        
        for i in range(1, n_times):
            t = time_grid[i]
            t_prev = time_grid[i-1]
            dt = t - t_prev
            df = np.exp(-self.r * t)
            
            # CVA: counterparty default risk on positive exposure
            pd_cpty = self.cpty.default_prob(t_prev, t)
            cva_increment = df * self.cpty.lgd * pd_cpty * ee[i]
            cva += cva_increment
            cva_profile[i] = cva_increment
            
            # DVA: own default risk on negative exposure
            pd_own = self.own.default_prob(t_prev, t)
            dva_increment = df * self.own.lgd * pd_own * abs(ene[i])
            dva += dva_increment
            dva_profile[i] = dva_increment
            
            # FVA: funding cost on net exposure
            net_exposure = ee[i] + ene[i]  # Can be positive or negative
            fva_increment = df * self.s_f * net_exposure * dt
            fva += fva_increment
            fva_profile[i] = fva_increment
            
            # KVA: cost of holding regulatory capital
            if capital_profile is not None:
                kva += df * self.h_k * capital_profile[i] * dt
            
            # MVA: cost of posting initial margin
            if simm_profile is not None:
                mva += df * self.s_f * simm_profile[i] * dt
        
        total_xva = cva - dva + fva + kva + mva
        
        return {
            'cva': cva,
            'dva': dva,
            'fva': fva,
            'kva': kva,
            'mva': mva,
            'total_xva': total_xva,
            'cva_profile': cva_profile,
            'dva_profile': dva_profile,
            'fva_profile': fva_profile,
            'bilateral_cva': cva - dva,
        }


# ============================================================================
# ISDA SIMM (Simplified)
# ============================================================================

class SIMMCalculator:
    """
    Simplified ISDA SIMM (Standard Initial Margin Model).
    
    SIMM computes initial margin based on sensitivities (delta, vega, curvature).
    
    For a vanilla IRS, the dominant component is IR Delta:
        IM = sqrt(sum_i sum_j w_i * w_j * rho_{ij} * s_i * s_j)
    
    where s_i are DV01 sensitivities to tenor buckets and w_i are risk weights.
    
    SIMM risk weights for IR (USD-like, in bps):
        2W: 15, 1M: 18, 3M: 9, 6M: 11, 1Y: 15, 2Y: 20, 3Y: 22,
        5Y: 20, 10Y: 19, 15Y: 15, 20Y: 13, 30Y: 16
    """
    
    # SIMM IR risk weights (simplified, basis points)
    TENORS = np.array([0.04, 0.08, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30])
    RISK_WEIGHTS = np.array([15, 18, 9, 11, 15, 20, 22, 20, 19, 15, 13, 16]) / 10000
    
    # Intra-bucket correlation
    INTRA_CORR = 0.98
    
    def __init__(self):
        pass
    
    def compute_dv01_profile(
        self,
        hw_model: HullWhiteModel,
        swap: IRSwap,
        rates: np.ndarray,
        time_grid: np.ndarray,
        bump: float = 0.0001
    ) -> np.ndarray:
        """
        Compute DV01 at each time step (simplified: total DV01).
        
        DV01 = dV/dr * 1bp
        """
        n_paths, n_times = rates.shape
        dv01_profile = np.zeros(n_times)
        
        for i, t in enumerate(time_grid):
            r = rates[:, i]
            v_base = hw_model.swap_value(r, t, swap)
            v_bump = hw_model.swap_value(r + bump, t, swap)
            dv01 = np.mean(np.abs(v_bump - v_base))
            dv01_profile[i] = dv01
        
        return dv01_profile
    
    def compute_simm(self, dv01: float, remaining_maturity: float) -> float:
        """
        Compute SIMM initial margin from DV01.
        
        Simplified: map DV01 to the nearest tenor bucket and apply risk weight.
        """
        # Find closest tenor
        idx = np.argmin(np.abs(self.TENORS - remaining_maturity))
        rw = self.RISK_WEIGHTS[idx]
        
        # SIMM IM = risk_weight * sensitivity (DV01 scaled)
        # DV01 is per 1bp, scale to absolute terms
        im = rw * dv01 * 10000 * 1.4  # 1.4 = concentration threshold multiplier
        
        return im
    
    def compute_simm_profile(
        self,
        dv01_profile: np.ndarray,
        time_grid: np.ndarray,
        maturity: float
    ) -> np.ndarray:
        """SIMM at each time step."""
        simm = np.zeros(len(time_grid))
        for i, t in enumerate(time_grid):
            remaining = maturity - t
            if remaining > 0:
                simm[i] = self.compute_simm(dv01_profile[i], remaining)
        return simm


# ============================================================================
# Regulatory Capital (SA-CCR Simplified)
# ============================================================================

class CapitalCalculator:
    """
    Simplified SA-CCR (Standardised Approach for Counterparty Credit Risk).
    
    EAD = alpha * (RC + PFE)
    RWA = EAD * risk_weight
    Capital = RWA * 8%
    
    For IRS:
        RC = max(V - C, 0) (replacement cost, V=MtM, C=collateral)
        PFE = multiplier * AddOn
        AddOn = SF * d * N * MF  (SF=0.5% for IR, d=supervisory duration)
    """
    
    ALPHA = 1.4
    SF_IR = 0.005              # Supervisory factor for IR
    CAPITAL_RATIO = 0.08       # 8% minimum
    
    def compute_ead(
        self,
        mtm: float,
        notional: float,
        maturity: float,
        collateral: float = 0.0
    ) -> float:
        """Compute Exposure at Default under SA-CCR."""
        # Replacement cost
        rc = max(mtm - collateral, 0)
        
        # Supervisory duration
        d = min(max((1 - np.exp(-0.05 * maturity)) / 0.05, 0), maturity)
        
        # Add-on
        addon = self.SF_IR * d * notional
        
        # Multiplier (if MtM is negative)
        if mtm - collateral < 0:
            floor = 0.05
            mult = floor + (1 - floor) * np.exp((mtm - collateral) / (2 * addon)) if addon > 0 else floor
        else:
            mult = 1.0
        
        pfe = mult * addon
        ead = self.ALPHA * (rc + pfe)
        
        return ead
    
    def compute_capital_profile(
        self,
        ee_profile: np.ndarray,
        time_grid: np.ndarray,
        swap: IRSwap,
        counterparty_rw: float = 0.50  # 50% risk weight (corporate)
    ) -> np.ndarray:
        """Capital requirement at each time step."""
        capital = np.zeros(len(time_grid))
        
        for i, t in enumerate(time_grid):
            remaining = swap.maturity_years - t
            if remaining > 0:
                ead = self.compute_ead(ee_profile[i], swap.notional, remaining)
                rwa = ead * counterparty_rw
                capital[i] = rwa * self.CAPITAL_RATIO
        
        return capital


# ============================================================================
# XVA Sensitivity Analysis
# ============================================================================

def xva_sensitivity_analysis(
    swap: IRSwap,
    hw_params: HullWhiteParams,
    n_paths: int = 50_000
) -> dict:
    """Analyse XVA sensitivity to credit spread and funding."""
    
    hw = HullWhiteModel(hw_params)
    time_grid = np.linspace(0, swap.maturity_years, 100)
    rates = hw.simulate_rates(n_paths, time_grid)
    
    exposure_engine = ExposureEngine(hw, swap)
    exposures = exposure_engine.compute_profiles(rates, time_grid)
    
    # Sensitivity to CDS spread
    spreads = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
    cva_by_spread = {}
    
    for s in spreads:
        cpty = CreditCurve("Cpty", cds_spread=s)
        own = CreditCurve("Own", cds_spread=0.003)
        xva_calc = XVACalculator(cpty, own)
        results = xva_calc.compute_all(exposures, time_grid)
        cva_by_spread[s] = results['cva']
    
    # Sensitivity to funding spread
    funding_spreads = [0.001, 0.002, 0.005, 0.01]
    fva_by_funding = {}
    
    for fs in funding_spreads:
        cpty = CreditCurve("Cpty", cds_spread=0.005)
        own = CreditCurve("Own", cds_spread=0.003)
        xva_calc = XVACalculator(cpty, own, funding_spread=fs)
        results = xva_calc.compute_all(exposures, time_grid)
        fva_by_funding[fs] = results['fva']
    
    return {
        'cva_by_spread': cva_by_spread,
        'fva_by_funding': fva_by_funding,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("XVA PRICING ENGINE")
    print("Interest Rate Swap — CVA / DVA / FVA / KVA / MVA / SIMM")
    print("Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE")
    print("=" * 60)
    
    # ── Product ──
    swap = IRSwap(
        notional=10_000_000,
        fixed_rate=0.035,
        maturity_years=10.0,
        payment_frequency=2,
        is_payer=True
    )
    print("\n" + swap.describe())
    
    # ── Hull-White model ──
    hw_params = HullWhiteParams(a=0.03, sigma=0.01, r0=0.035)
    hw = HullWhiteModel(hw_params)
    
    print(f"Hull-White: a={hw_params.a}, sigma={hw_params.sigma}, r0={hw_params.r0:.2%}")
    
    # ── Simulate rates ──
    n_paths = 100_000
    time_grid = np.linspace(0, swap.maturity_years, 120)  # Monthly
    
    print(f"\nSimulating {n_paths:,} rate paths...")
    t0 = time.time()
    rates = hw.simulate_rates(n_paths, time_grid, seed=42)
    print(f"Simulation: {time.time()-t0:.2f}s")
    
    # ── Exposure profiles ──
    print("Computing exposure profiles...")
    t0 = time.time()
    exposure_engine = ExposureEngine(hw, swap)
    exposures = exposure_engine.compute_profiles(rates, time_grid)
    print(f"Exposure: {time.time()-t0:.2f}s")
    
    print(f"\nEXPOSURE PROFILES")
    print(f"{'='*40}")
    print(f"EPE:           {exposures['epe']:>12,.0f}")
    print(f"Effective EPE: {exposures['effective_epe']:>12,.0f}")
    print(f"Peak EE:       {np.max(exposures['ee']):>12,.0f}")
    print(f"Peak ENE:      {np.min(exposures['ene']):>12,.0f}")
    print(f"Peak PFE 97.5: {np.max(exposures['pfe_97_5']):>12,.0f}")
    print(f"Peak PFE 99:   {np.max(exposures['pfe_99']):>12,.0f}")
    
    # ── SIMM ──
    print(f"\nComputing SIMM profile...")
    simm_calc = SIMMCalculator()
    dv01_profile = simm_calc.compute_dv01_profile(hw, swap, rates, time_grid)
    simm_profile = simm_calc.compute_simm_profile(dv01_profile, time_grid, swap.maturity_years)
    
    print(f"Initial SIMM IM: {simm_profile[1]:>12,.0f}")
    print(f"Peak SIMM IM:    {np.max(simm_profile):>12,.0f}")
    
    # ── Capital ──
    cap_calc = CapitalCalculator()
    capital_profile = cap_calc.compute_capital_profile(
        exposures['ee'], time_grid, swap
    )
    
    print(f"Initial capital: {capital_profile[1]:>12,.0f}")
    print(f"Peak capital:    {np.max(capital_profile):>12,.0f}")
    
    # ── XVA ──
    cpty = CreditCurve("Counterparty", cds_spread=0.005, recovery=0.40)
    own = CreditCurve("Own Bank", cds_spread=0.003, recovery=0.40)
    
    print(f"\nCredit: Cpty CDS={cpty.cds_spread*10000:.0f}bps, Own CDS={own.cds_spread*10000:.0f}bps")
    
    xva_calc = XVACalculator(
        cpty, own,
        funding_spread=0.002,
        capital_hurdle=0.10,
        risk_free_rate=hw_params.flat_curve
    )
    
    xva = xva_calc.compute_all(exposures, time_grid, simm_profile, capital_profile)
    
    print(f"\nXVA RESULTS")
    print(f"{'='*40}")
    print(f"CVA:            {xva['cva']:>12,.0f} ({xva['cva']/swap.notional*10000:.1f} bps)")
    print(f"DVA:            {xva['dva']:>12,.0f} ({xva['dva']/swap.notional*10000:.1f} bps)")
    print(f"Bilateral CVA:  {xva['bilateral_cva']:>12,.0f} ({xva['bilateral_cva']/swap.notional*10000:.1f} bps)")
    print(f"FVA:            {xva['fva']:>12,.0f} ({xva['fva']/swap.notional*10000:.1f} bps)")
    print(f"KVA:            {xva['kva']:>12,.0f} ({xva['kva']/swap.notional*10000:.1f} bps)")
    print(f"MVA:            {xva['mva']:>12,.0f} ({xva['mva']/swap.notional*10000:.1f} bps)")
    print(f"{'─'*40}")
    print(f"Total XVA:      {xva['total_xva']:>12,.0f} ({xva['total_xva']/swap.notional*10000:.1f} bps)")
    
    # ── Risk-free price vs XVA-adjusted ──
    mid_mtm = np.mean(exposures['mtm'][:, 0])
    print(f"\nRisk-free MtM:  {mid_mtm:>12,.0f}")
    print(f"XVA-adjusted:   {mid_mtm - xva['total_xva']:>12,.0f}")
    
    # ── Sensitivity analysis ──
    print(f"\n\nCVA SENSITIVITY TO CDS SPREAD")
    print(f"{'CDS (bps)':<12} {'CVA':>12} {'CVA (bps)':>12}")
    print("-" * 38)
    
    sens = xva_sensitivity_analysis(swap, hw_params, n_paths=50_000)
    for spread, cva_val in sens['cva_by_spread'].items():
        print(f"{spread*10000:<12.0f} {cva_val:>12,.0f} {cva_val/swap.notional*10000:>11.1f}")
    
    print(f"\nFVA SENSITIVITY TO FUNDING SPREAD")
    print(f"{'Funding (bps)':<14} {'FVA':>12} {'FVA (bps)':>12}")
    print("-" * 40)
    for fs, fva_val in sens['fva_by_funding'].items():
        print(f"{fs*10000:<14.0f} {fva_val:>12,.0f} {fva_val/swap.notional*10000:>11.1f}")
    
    return exposures, xva, simm_profile, capital_profile, sens


if __name__ == "__main__":
    exposures, xva, simm, capital, sens = main()
