"""
Merton Structural Credit Model
================================
Bridges statistical credit risk (PD, scorecards) with market-implied
credit pricing (CDS, bond spreads).

Merton (1974): Equity = Call option on firm value
  V(t) follows GBM: dV = mu*V*dt + sigma_V*V*dW
  Default at T if V(T) < D (debt barrier)
  Equity = V*N(d1) - D*exp(-rT)*N(d2)

This gives us:
  - Distance to Default (DD): market-implied credit quality metric
  - Risk-neutral PD: from DD, directly comparable to CDS-implied PD
  - CDS spread: from PD and LGD
  - Credit curve: term structure of default probability

Calibration: solve for (V, sigma_V) from observed (E, sigma_E)
using the Merton equations + Ito's lemma relationship.

Extensions:
  1. KMV/Moody's EDF (Expected Default Frequency) mapping
  2. First-passage (Black-Cox): default before maturity
  3. CreditGrades model: stochastic barrier
  4. Portfolio credit risk: correlated defaults via asset correlation

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve, minimize_scalar, minimize
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time


# ============================================================================
# Firm Data
# ============================================================================

@dataclass
class FirmData:
    """Observable firm data."""
    name: str
    equity_value: float           # Market cap (E)
    equity_vol: float             # Annualised equity vol (sigma_E)
    total_debt: float             # Total debt (D) — short-term + 0.5*long-term (KMV convention)
    risk_free_rate: float = 0.04  # Risk-free rate
    debt_maturity: float = 1.0    # Debt maturity (T)
    recovery_rate: float = 0.40   # Recovery rate
    dividend_yield: float = 0.0   # Continuous dividend yield
    
    # Optional: observed CDS spread for calibration
    cds_spread_1y: Optional[float] = None
    cds_spread_5y: Optional[float] = None
    
    @property
    def leverage(self) -> float:
        return self.total_debt / (self.equity_value + self.total_debt)
    
    @property
    def lgd(self) -> float:
        return 1.0 - self.recovery_rate
    
    def describe(self) -> str:
        return (
            f"{self.name}\n"
            f"  Equity: ${self.equity_value:,.0f}M | Vol: {self.equity_vol:.0%}\n"
            f"  Debt:   ${self.total_debt:,.0f}M | Leverage: {self.leverage:.1%}\n"
            f"  r={self.risk_free_rate:.1%} | T={self.debt_maturity}Y | RR={self.recovery_rate:.0%}"
        )


# ============================================================================
# Merton Model
# ============================================================================

class MertonModel:
    """
    Merton (1974) structural credit model.
    
    Core equations:
      E = V*N(d1) - D*exp(-rT)*N(d2)
      sigma_E * E = N(d1) * sigma_V * V     (Ito's lemma)
    
    where:
      d1 = [ln(V/D) + (r + 0.5*sigma_V^2)*T] / (sigma_V * sqrt(T))
      d2 = d1 - sigma_V * sqrt(T)
    
    Distance to Default:
      DD = [ln(V/D) + (mu - 0.5*sigma_V^2)*T] / (sigma_V * sqrt(T))
    
    Risk-neutral PD:
      PD = N(-d2)
    """
    
    def __init__(self):
        pass
    
    def d1_d2(
        self,
        V: float,
        D: float,
        r: float,
        sigma_V: float,
        T: float
    ) -> Tuple[float, float]:
        """Compute d1 and d2."""
        d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)
        return d1, d2
    
    def equity_value(
        self,
        V: float,
        D: float,
        r: float,
        sigma_V: float,
        T: float
    ) -> float:
        """Merton equity = Call(V, D, r, sigma_V, T)."""
        d1, d2 = self.d1_d2(V, D, r, sigma_V, T)
        return V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
    
    def equity_vol_relationship(
        self,
        V: float,
        D: float,
        r: float,
        sigma_V: float,
        T: float,
        E: float
    ) -> float:
        """Implied equity vol from Ito's lemma: sigma_E = N(d1) * sigma_V * V / E."""
        d1, _ = self.d1_d2(V, D, r, sigma_V, T)
        return norm.cdf(d1) * sigma_V * V / E
    
    def calibrate(self, firm: FirmData) -> dict:
        """
        Calibrate (V, sigma_V) from observed (E, sigma_E).
        
        Solve the system:
          E = V*N(d1) - D*exp(-rT)*N(d2)
          sigma_E * E = N(d1) * sigma_V * V
        """
        E = firm.equity_value
        sigma_E = firm.equity_vol
        D = firm.total_debt
        r = firm.risk_free_rate
        T = firm.debt_maturity
        
        def equations(params):
            V, sigma_V = params
            if V <= 0 or sigma_V <= 0:
                return [1e10, 1e10]
            
            d1, d2 = self.d1_d2(V, D, r, sigma_V, T)
            
            # Equation 1: E = V*N(d1) - D*exp(-rT)*N(d2)
            eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
            
            # Equation 2: sigma_E * E = N(d1) * sigma_V * V
            eq2 = norm.cdf(d1) * sigma_V * V - sigma_E * E
            
            return [eq1, eq2]
        
        # Initial guess: V = E + D, sigma_V = sigma_E * E / (E + D)
        V0 = E + D
        sigma_V0 = sigma_E * E / V0
        
        solution = fsolve(equations, [V0, sigma_V0], full_output=True)
        V_star, sigma_V_star = solution[0]
        
        # Ensure positive
        V_star = max(V_star, E + 1)
        sigma_V_star = max(sigma_V_star, 0.001)
        
        # Compute credit metrics
        d1, d2 = self.d1_d2(V_star, D, r, sigma_V_star, T)
        
        # Distance to Default (physical measure, using drift = r for simplicity)
        dd = (np.log(V_star / D) + (r - 0.5 * sigma_V_star**2) * T) / (sigma_V_star * np.sqrt(T))
        
        # Risk-neutral PD
        pd_rn = norm.cdf(-d2)
        
        # Physical PD (using equity risk premium as proxy for drift)
        mu = r + 0.05  # Assume 5% equity risk premium
        dd_physical = (np.log(V_star / D) + (mu - 0.5 * sigma_V_star**2) * T) / (sigma_V_star * np.sqrt(T))
        pd_physical = norm.cdf(-dd_physical)
        
        # CDS spread implied
        cds_implied = pd_rn * firm.lgd / T * 10000  # In basis points (approximate)
        
        # Debt value
        debt_value = D * np.exp(-r * T) * norm.cdf(d2) + V_star * norm.cdf(-d1)
        debt_yield = -np.log(debt_value / D) / T
        credit_spread = debt_yield - r
        
        return {
            'firm_value': V_star,
            'firm_vol': sigma_V_star,
            'd1': d1,
            'd2': d2,
            'distance_to_default': dd,
            'dd_physical': dd_physical,
            'pd_risk_neutral': pd_rn,
            'pd_physical': pd_physical,
            'cds_implied_bps': cds_implied,
            'credit_spread_bps': credit_spread * 10000,
            'debt_value': debt_value,
            'leverage_market': D / V_star,
        }
    
    def term_structure_pd(
        self,
        V: float,
        D: float,
        r: float,
        sigma_V: float,
        maturities: np.ndarray
    ) -> np.ndarray:
        """Compute PD term structure."""
        pds = np.zeros(len(maturities))
        for i, T in enumerate(maturities):
            _, d2 = self.d1_d2(V, D, r, sigma_V, T)
            pds[i] = norm.cdf(-d2)
        return pds
    
    def term_structure_cds(
        self,
        V: float,
        D: float,
        r: float,
        sigma_V: float,
        lgd: float,
        maturities: np.ndarray
    ) -> np.ndarray:
        """Implied CDS spread term structure."""
        pds = self.term_structure_pd(V, D, r, sigma_V, maturities)
        # Approximate: CDS ~ hazard_rate * LGD, hazard ~ -ln(1-PD)/T
        spreads = np.zeros(len(maturities))
        for i, T in enumerate(maturities):
            if pds[i] < 1:
                hazard = -np.log(1 - pds[i]) / T
                spreads[i] = hazard * lgd * 10000
            else:
                spreads[i] = 10000
        return spreads


# ============================================================================
# Extension 1: KMV / EDF Mapping
# ============================================================================

class KMVModel:
    """
    KMV (Moody's) Expected Default Frequency mapping.
    
    KMV modifies Merton by:
    1. Using default point = STD + 0.5*LTD (not full debt)
    2. Mapping DD to EDF via empirical distribution (not Normal)
    3. Incorporating asset growth rate
    
    The empirical mapping captures the fact that actual default rates
    are higher than Normal-implied PDs (fat tails in practice).
    """
    
    @staticmethod
    def default_point(short_term_debt: float, long_term_debt: float) -> float:
        """KMV default point convention."""
        return short_term_debt + 0.5 * long_term_debt
    
    @staticmethod
    def dd_to_edf(dd: float) -> float:
        """
        Map Distance to Default to Expected Default Frequency.
        
        Uses empirical calibration (simplified):
        - For DD > 4: very low default (< 0.05%)
        - For DD ~ 2: moderate risk (~2%)
        - For DD < 1: distressed (>10%)
        
        Empirical mapping is fatter-tailed than Normal.
        """
        # Simplified empirical mapping (logistic-like)
        # Actual KMV uses proprietary database
        if dd > 8:
            return 0.0001  # 1bp
        
        # Empirical: fatter tails than Normal
        # Use a scaled Normal with heavier left tail
        edf_normal = norm.cdf(-dd)
        
        # Empirical adjustment: 2-3x Normal for dd > 2, converging for dd < 0
        if dd > 2:
            adjustment = 2.5
        elif dd > 0:
            adjustment = 2.5 - 1.5 * (2 - dd) / 2
        else:
            adjustment = 1.0
        
        return min(edf_normal * adjustment, 1.0)
    
    @staticmethod
    def rating_from_edf(edf: float) -> str:
        """Map EDF to approximate credit rating."""
        if edf < 0.0004:
            return "AAA"
        elif edf < 0.001:
            return "AA"
        elif edf < 0.003:
            return "A"
        elif edf < 0.01:
            return "BBB"
        elif edf < 0.03:
            return "BB"
        elif edf < 0.08:
            return "B"
        elif edf < 0.20:
            return "CCC"
        else:
            return "CC/D"


# ============================================================================
# Extension 2: First-Passage (Black-Cox)
# ============================================================================

class BlackCoxModel:
    """
    Black-Cox (1976) first-passage model.
    
    Default occurs at the FIRST time V(t) hits the barrier D(t),
    not just at maturity. This is more realistic because firms can
    default at any time, not just when debt matures.
    
    D(t) = D * exp(-gamma * (T-t))  (barrier grows towards maturity)
    
    Survival probability uses the first-passage distribution of GBM.
    """
    
    def survival_probability(
        self,
        V: float,
        D: float,
        r: float,
        sigma_V: float,
        T: float,
        gamma: float = 0.0  # Barrier growth rate
    ) -> float:
        """
        First-passage survival probability.
        
        P(min_{0<t<T} V(t) > D*exp(-gamma*(T-t)))
        
        For gamma=0 (flat barrier):
        P(survive) = N(d2) - (D/V)^(2*mu/sigma^2) * N(d2_reflected)
        """
        mu = r - 0.5 * sigma_V**2
        barrier = D * np.exp(-gamma * T)
        
        if V <= barrier:
            return 0.0
        
        sqrt_T = np.sqrt(T)
        
        d2 = (np.log(V / barrier) + mu * T) / (sigma_V * sqrt_T)
        d2_ref = (np.log(barrier / V) + mu * T) / (sigma_V * sqrt_T)
        
        # Reflection term
        power = 2 * mu / (sigma_V**2) if sigma_V > 0 else 0
        reflection = (barrier / V)**power * norm.cdf(d2_ref)
        
        survival = norm.cdf(d2) - reflection
        return max(min(survival, 1.0), 0.0)
    
    def default_probability(
        self,
        V: float,
        D: float,
        r: float,
        sigma_V: float,
        T: float,
        gamma: float = 0.0
    ) -> float:
        """First-passage default probability."""
        return 1.0 - self.survival_probability(V, D, r, sigma_V, T, gamma)
    
    def term_structure(
        self,
        V: float,
        D: float,
        r: float,
        sigma_V: float,
        maturities: np.ndarray,
        gamma: float = 0.0
    ) -> np.ndarray:
        """First-passage PD term structure."""
        return np.array([
            self.default_probability(V, D, r, sigma_V, T, gamma)
            for T in maturities
        ])


# ============================================================================
# Extension 3: CreditGrades (Stochastic Barrier)
# ============================================================================

class CreditGradesModel:
    """
    CreditGrades (Finger et al., 2002) — Goldman Sachs / JP Morgan model.
    
    Extends Merton with:
    1. Stochastic default barrier (uncertain recovery)
    2. Continuous first-passage
    3. No maturity assumption (perpetual framework)
    
    Barrier: D * L * exp(lambda*Z - 0.5*lambda^2)
    where Z ~ N(0,1) and lambda is the uncertainty of recovery.
    
    This produces more realistic CDS spreads, especially at short tenors
    where Merton gives near-zero spreads.
    """
    
    def __init__(self, lambda_param: float = 0.3):
        self.lambda_param = lambda_param  # Recovery uncertainty
    
    def survival_probability(
        self,
        V: float,
        D: float,
        L: float,  # Expected LGD
        sigma_V: float,
        r: float,
        T: float
    ) -> float:
        """CreditGrades survival probability."""
        lam = self.lambda_param
        
        # Effective barrier
        d_bar = D * L
        
        # Adjusted volatility (includes recovery uncertainty)
        sigma_eff = np.sqrt(sigma_V**2 + lam**2)
        
        if V <= d_bar or sigma_eff < 1e-10 or T < 1e-10:
            return 0.0 if V <= d_bar else 1.0
        
        # First-passage with stochastic barrier
        A = (np.log(V / d_bar) + 0.5 * lam**2) / sigma_eff
        
        sqrt_T = np.sqrt(T)
        
        d1 = A / sqrt_T + 0.5 * sigma_eff * sqrt_T
        d2 = A / sqrt_T - 0.5 * sigma_eff * sqrt_T
        
        # Exponential term
        exp_term = np.exp(-2 * A * (A - lam**2 / sigma_eff) / (sigma_eff if sigma_eff > 0 else 1))
        exp_term = min(exp_term, 1e10)
        
        survival = norm.cdf(d1) - exp_term * norm.cdf(d2)
        return max(min(survival, 1.0), 0.0)
    
    def cds_spread(
        self,
        V: float,
        D: float,
        L: float,
        sigma_V: float,
        r: float,
        T: float
    ) -> float:
        """Implied CDS spread in basis points."""
        pd = 1 - self.survival_probability(V, D, L, sigma_V, r, T)
        if pd >= 1:
            return 10000
        
        # Approximate: hazard -> spread
        if pd > 0 and pd < 1:
            hazard = -np.log(1 - pd) / T
            return hazard * L * 10000
        return 0.0


# ============================================================================
# Extension 4: Portfolio Credit Risk (Correlated Defaults)
# ============================================================================

class PortfolioCreditModel:
    """
    Portfolio credit risk with correlated defaults.
    
    Uses the Gaussian copula (like Basel II IRB):
    - Each firm has asset return: X_i = sqrt(rho)*Z + sqrt(1-rho)*eps_i
    - Default if X_i < N^{-1}(PD_i)
    - Z = systematic factor, eps_i = idiosyncratic
    
    Computes:
    - Portfolio loss distribution
    - Expected loss, unexpected loss
    - VaR and CVaR for credit losses
    - Contribution of each obligor to portfolio risk
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def simulate_defaults(
        self,
        pds: np.ndarray,
        lgds: np.ndarray,
        exposures: np.ndarray,
        correlation: float = 0.20,
        n_simulations: int = 100_000
    ) -> dict:
        """
        Monte Carlo simulation of correlated defaults.
        
        Returns portfolio loss distribution.
        """
        n_obligors = len(pds)
        
        # Default thresholds
        thresholds = norm.ppf(pds)
        
        # Simulate
        losses = np.zeros(n_simulations)
        default_counts = np.zeros(n_simulations)
        
        for sim in range(n_simulations):
            # Systematic factor
            Z = self.rng.standard_normal()
            
            # Individual asset returns
            eps = self.rng.standard_normal(n_obligors)
            X = np.sqrt(correlation) * Z + np.sqrt(1 - correlation) * eps
            
            # Default indicator
            defaults = X < thresholds
            
            # Loss
            loss = np.sum(defaults * lgds * exposures)
            losses[sim] = loss
            default_counts[sim] = np.sum(defaults)
        
        total_exposure = np.sum(exposures)
        
        return {
            'expected_loss': np.mean(losses),
            'unexpected_loss': np.std(losses),
            'var_99': np.percentile(losses, 99),
            'var_999': np.percentile(losses, 99.9),
            'cvar_99': np.mean(losses[losses >= np.percentile(losses, 99)]),
            'max_loss': np.max(losses),
            'avg_defaults': np.mean(default_counts),
            'max_defaults': np.max(default_counts),
            'loss_pct_99': np.percentile(losses, 99) / total_exposure,
            'loss_distribution': losses,
            'el_pct': np.mean(losses) / total_exposure,
        }
    
    def vasicek_analytical(
        self,
        pd: float,
        lgd: float,
        correlation: float,
        confidence: float = 0.999
    ) -> float:
        """
        Vasicek single-factor analytical formula (Basel II IRB).
        
        UL = LGD * N[N^{-1}(PD) * sqrt(1/(1-rho)) + sqrt(rho/(1-rho)) * N^{-1}(confidence)]
        """
        inv_pd = norm.ppf(pd)
        inv_conf = norm.ppf(confidence)
        
        conditional_pd = norm.cdf(
            (inv_pd + np.sqrt(correlation) * inv_conf) / np.sqrt(1 - correlation)
        )
        
        return lgd * conditional_pd


# ============================================================================
# CDS Calibration
# ============================================================================

class CDSCalibrator:
    """
    Calibrate Merton model parameters to match observed CDS spreads.
    
    Given: equity_value, equity_vol, observed CDS spread
    Find: firm_value, firm_vol that match the CDS spread
    """
    
    def __init__(self):
        self.merton = MertonModel()
    
    def calibrate_to_cds(
        self,
        firm: FirmData,
        target_cds_bps: float,
        model: str = 'merton'
    ) -> dict:
        """
        Calibrate to match target CDS spread.
        
        Adjusts firm vol to match the observed CDS spread.
        """
        base_result = self.merton.calibrate(firm)
        
        # Optimise: find sigma_V that produces the target CDS
        def objective(sigma_V_adj):
            if sigma_V_adj <= 0.01:
                return 1e10
            
            # Create adjusted firm data
            d1, d2 = self.merton.d1_d2(
                base_result['firm_value'], firm.total_debt,
                firm.risk_free_rate, sigma_V_adj, firm.debt_maturity
            )
            pd = norm.cdf(-d2)
            
            # CDS spread
            if pd < 1 and pd > 0:
                hazard = -np.log(1 - pd) / firm.debt_maturity
                cds = hazard * firm.lgd * 10000
            else:
                cds = 10000 if pd >= 1 else 0
            
            return (cds - target_cds_bps)**2
        
        result = minimize_scalar(objective, bounds=(0.01, 1.5), method='bounded')
        sigma_V_calibrated = result.x
        
        # Recompute with calibrated vol
        d1, d2 = self.merton.d1_d2(
            base_result['firm_value'], firm.total_debt,
            firm.risk_free_rate, sigma_V_calibrated, firm.debt_maturity
        )
        pd_calibrated = norm.cdf(-d2)
        dd_calibrated = (np.log(base_result['firm_value'] / firm.total_debt) 
                        + (firm.risk_free_rate - 0.5 * sigma_V_calibrated**2) * firm.debt_maturity
                        ) / (sigma_V_calibrated * np.sqrt(firm.debt_maturity))
        
        if pd_calibrated < 1 and pd_calibrated > 0:
            hazard = -np.log(1 - pd_calibrated) / firm.debt_maturity
            cds_fitted = hazard * firm.lgd * 10000
        else:
            cds_fitted = 0
        
        return {
            'sigma_V_base': base_result['firm_vol'],
            'sigma_V_calibrated': sigma_V_calibrated,
            'pd_base': base_result['pd_risk_neutral'],
            'pd_calibrated': pd_calibrated,
            'dd_calibrated': dd_calibrated,
            'cds_target': target_cds_bps,
            'cds_fitted': cds_fitted,
            'fit_error_bps': abs(cds_fitted - target_cds_bps),
            'rating': KMVModel.rating_from_edf(pd_calibrated),
        }


# ============================================================================
# Multi-Firm Analysis
# ============================================================================

def create_sample_firms() -> List[FirmData]:
    """Create a diverse set of firms for analysis."""
    return [
        FirmData("Apple (Tech)", equity_value=3000000, equity_vol=0.25, total_debt=120000,
                 cds_spread_5y=0.0020, recovery_rate=0.40),
        FirmData("JPMorgan (Bank)", equity_value=550000, equity_vol=0.28, total_debt=3200000,
                 cds_spread_5y=0.0045, recovery_rate=0.40),
        FirmData("Ford (Auto)", equity_value=45000, equity_vol=0.45, total_debt=140000,
                 cds_spread_5y=0.0180, recovery_rate=0.35),
        FirmData("Netflix (Growth)", equity_value=250000, equity_vol=0.40, total_debt=15000,
                 cds_spread_5y=0.0060, recovery_rate=0.40),
        FirmData("Boeing (Industrial)", equity_value=120000, equity_vol=0.38, total_debt=58000,
                 cds_spread_5y=0.0150, recovery_rate=0.30),
        FirmData("EM Telecom (HY)", equity_value=8000, equity_vol=0.55, total_debt=12000,
                 cds_spread_5y=0.0450, recovery_rate=0.25),
    ]


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("MERTON STRUCTURAL CREDIT MODEL")
    print("Calibration, CDS Pricing, KMV, Black-Cox, Portfolio Credit")
    print("Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE")
    print("=" * 65)
    
    merton = MertonModel()
    firms = create_sample_firms()
    
    # ── Calibrate all firms ──
    print(f"\nMERTON CALIBRATION")
    print(f"{'Firm':<22} {'V($M)':>10} {'σ_V':>8} {'DD':>8} {'PD(RN)':>8} {'PD(Phys)':>8} {'CDS(bp)':>8} {'Sprd(bp)':>8}")
    print("-" * 88)
    
    results = {}
    for firm in firms:
        result = merton.calibrate(firm)
        results[firm.name] = result
        print(f"{firm.name:<22} {result['firm_value']/1000:>9.0f} {result['firm_vol']:>7.1%} "
              f"{result['distance_to_default']:>7.2f} {result['pd_risk_neutral']:>7.2%} "
              f"{result['pd_physical']:>7.2%} {result['cds_implied_bps']:>7.0f} {result['credit_spread_bps']:>7.0f}")
    
    # ── KMV / EDF Mapping ──
    print(f"\nKMV / EDF MAPPING")
    print(f"{'Firm':<22} {'DD':>8} {'PD(Normal)':>10} {'EDF(KMV)':>10} {'Rating':>8}")
    print("-" * 62)
    
    kmv = KMVModel()
    for firm in firms:
        r = results[firm.name]
        edf = kmv.dd_to_edf(r['distance_to_default'])
        rating = kmv.rating_from_edf(edf)
        print(f"{firm.name:<22} {r['distance_to_default']:>7.2f} {r['pd_risk_neutral']:>9.2%} "
              f"{edf:>9.2%} {rating:>8}")
    
    # ── First-Passage (Black-Cox) ──
    print(f"\nFIRST-PASSAGE (BLACK-COX) vs MERTON")
    print(f"{'Firm':<22} {'PD Merton':>10} {'PD BC(γ=0)':>10} {'PD BC(γ=2%)':>12} {'Ratio':>8}")
    print("-" * 65)
    
    bc = BlackCoxModel()
    for firm in firms:
        r = results[firm.name]
        V = r['firm_value']
        sigma = r['firm_vol']
        
        pd_merton = r['pd_risk_neutral']
        pd_bc_0 = bc.default_probability(V, firm.total_debt, firm.risk_free_rate, sigma, firm.debt_maturity)
        pd_bc_2 = bc.default_probability(V, firm.total_debt, firm.risk_free_rate, sigma, firm.debt_maturity, gamma=0.02)
        
        ratio = pd_bc_0 / max(pd_merton, 1e-10)
        print(f"{firm.name:<22} {pd_merton:>9.2%} {pd_bc_0:>9.2%} {pd_bc_2:>11.2%} {ratio:>7.1f}x")
    
    # ── CreditGrades ──
    print(f"\nCREDITGRADES MODEL")
    print(f"{'Firm':<22} {'CDS Merton':>10} {'CDS CG':>10} {'Obs CDS':>10}")
    print("-" * 55)
    
    cg = CreditGradesModel(lambda_param=0.3)
    for firm in firms:
        r = results[firm.name]
        V = r['firm_value']
        sigma = r['firm_vol']
        
        cds_merton = r['cds_implied_bps']
        cds_cg = cg.cds_spread(V, firm.total_debt, firm.lgd, sigma, firm.risk_free_rate, 5.0)
        obs = firm.cds_spread_5y * 10000 if firm.cds_spread_5y else '-'
        
        obs_str = f"{obs:>9.0f}" if isinstance(obs, float) else f"{obs:>9}"
        print(f"{firm.name:<22} {cds_merton:>9.0f} {cds_cg:>9.0f} {obs_str}")
    
    # ── CDS Calibration ──
    print(f"\nCDS CALIBRATION")
    print(f"{'Firm':<22} {'Target':>8} {'Fitted':>8} {'Error':>8} {'σ_V adj':>8} {'PD adj':>8} {'Rating':>8}")
    print("-" * 72)
    
    calibrator = CDSCalibrator()
    for firm in firms:
        if firm.cds_spread_5y:
            target = firm.cds_spread_5y * 10000
            cal = calibrator.calibrate_to_cds(firm, target)
            print(f"{firm.name:<22} {target:>7.0f} {cal['cds_fitted']:>7.0f} {cal['fit_error_bps']:>7.1f} "
                  f"{cal['sigma_V_calibrated']:>7.1%} {cal['pd_calibrated']:>7.2%} {cal['rating']:>8}")
    
    # ── Term Structure ──
    print(f"\nCDS TERM STRUCTURE (selected firms)")
    maturities = np.array([1, 2, 3, 5, 7, 10])
    print(f"{'Firm':<22}" + "".join(f"  {m}Y" for m in maturities))
    print("-" * 62)
    
    for firm in [firms[0], firms[2], firms[5]]:
        r = results[firm.name]
        spreads = merton.term_structure_cds(
            r['firm_value'], firm.total_debt, firm.risk_free_rate,
            r['firm_vol'], firm.lgd, maturities
        )
        row = f"{firm.name:<22}"
        for s in spreads:
            row += f" {s:>4.0f}"
        print(row)
    
    # ── Portfolio Credit Risk ──
    print(f"\nPORTFOLIO CREDIT RISK (Gaussian Copula)")
    print("-" * 50)
    
    pds = np.array([results[f.name]['pd_risk_neutral'] for f in firms])
    lgds = np.array([f.lgd for f in firms])
    exposures = np.array([10, 50, 20, 15, 30, 10], dtype=float)  # $M per name
    
    portfolio = PortfolioCreditModel(seed=42)
    
    for corr_name, corr in [("Low (10%)", 0.10), ("Medium (20%)", 0.20), ("High (40%)", 0.40)]:
        port_result = portfolio.simulate_defaults(pds, lgds, exposures, corr, n_simulations=100_000)
        print(f"\nCorrelation: {corr_name}")
        print(f"  Expected Loss:    ${port_result['expected_loss']:>8.2f}M ({port_result['el_pct']:.2%})")
        print(f"  Unexpected Loss:  ${port_result['unexpected_loss']:>8.2f}M")
        print(f"  VaR 99%:          ${port_result['var_99']:>8.2f}M ({port_result['loss_pct_99']:.2%})")
        print(f"  CVaR 99%:         ${port_result['cvar_99']:>8.2f}M")
        print(f"  Avg defaults:     {port_result['avg_defaults']:>8.2f}")
    
    # Vasicek analytical comparison
    print(f"\nVASICEK ANALYTICAL (IRB Formula)")
    avg_pd = np.mean(pds)
    avg_lgd = np.mean(lgds)
    for corr in [0.10, 0.20, 0.40]:
        ul = portfolio.vasicek_analytical(avg_pd, avg_lgd, corr, 0.999)
        print(f"  rho={corr:.0%}: UL = {ul:.2%} of exposure")
    
    return results, firms


if __name__ == "__main__":
    results, firms = main()
