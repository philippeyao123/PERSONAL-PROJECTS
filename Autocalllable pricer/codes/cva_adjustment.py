"""
Extension 4: CVA (Credit Valuation Adjustment)
================================================
Computes the counterparty credit risk adjustment for the autocallable.

CVA = E[ LGD * DF(tau) * max(V(tau), 0) * 1_{tau < T} ]

where:
- tau: default time of the issuer (modelled via hazard rate)
- LGD: loss given default
- V(tau): mark-to-market value at default
- DF: discount factor

For structured notes, the investor is exposed to issuer default risk.
If the issuer defaults when the note has positive MTM (e.g. after a
market rally but before autocall), the investor loses.

Implementation:
1. Simulate default times from a CDS-implied hazard rate
2. For each path where default occurs before maturity/autocall,
   compute the exposure at default
3. CVA = expected loss

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CreditParams:
    """Counterparty credit parameters."""
    cds_spread: float = 0.005      # 5Y CDS spread (e.g. 50bps for investment grade)
    recovery_rate: float = 0.40    # Recovery rate
    
    @property
    def lgd(self) -> float:
        """Loss given default."""
        return 1.0 - self.recovery_rate
    
    @property
    def hazard_rate(self) -> float:
        """Constant hazard rate implied by CDS spread."""
        return self.cds_spread / self.lgd
    
    def survival_probability(self, t: float) -> float:
        """Probability issuer survives to time t."""
        return np.exp(-self.hazard_rate * t)
    
    def default_probability(self, t: float) -> float:
        """Probability of default by time t."""
        return 1.0 - self.survival_probability(t)
    
    def describe(self) -> str:
        return (
            f"CDS: {self.cds_spread*10000:.0f}bps, "
            f"Recovery: {self.recovery_rate:.0%}, "
            f"LGD: {self.lgd:.0%}, "
            f"Hazard rate: {self.hazard_rate:.4f}, "
            f"5Y default prob: {self.default_probability(5):.2%}"
        )


class CVACalculator:
    """
    Compute CVA for the autocallable structured note.
    
    Method:
    1. For each MC path, determine the redemption time (autocall or maturity)
    2. Simulate a default time tau from exponential distribution
    3. If tau < redemption time and the exposure is positive, compute loss
    4. CVA = average discounted loss across all paths
    """
    
    def __init__(self, credit: CreditParams, risk_free_rate: float = 0.04):
        self.credit = credit
        self.r = risk_free_rate
    
    def compute_cva(
        self,
        payoffs: np.ndarray,
        redemption_times: np.ndarray,
        notional: float,
        n_simulations: int = 50_000,
        seed: int = 99
    ) -> dict:
        """
        Compute CVA from pre-computed payoff results.
        
        Args:
            payoffs: Discounted payoffs per path
            redemption_times: Time of cash flow for each path
            notional: Note notional
            n_simulations: Number of default time simulations per path
            seed: Random seed
        
        Returns:
            Dictionary with CVA results
        """
        rng = np.random.default_rng(seed)
        n_paths = len(payoffs)
        
        # Simulate default times (exponential distribution)
        default_times = rng.exponential(
            1.0 / self.credit.hazard_rate,
            size=n_paths
        )
        
        # Exposure at default: positive MTM at default
        # Approximation: use the final payoff as proxy for MTM
        # (in practice, you'd need intermediate MTM via nested MC)
        mtm = payoffs - notional * np.exp(-self.r * redemption_times)
        exposure = np.maximum(mtm, 0.0)
        
        # Default occurs before redemption
        default_before_redemption = default_times < redemption_times
        
        # Discounted loss
        df_at_default = np.exp(-self.r * default_times)
        losses = np.where(
            default_before_redemption,
            self.credit.lgd * df_at_default * exposure,
            0.0
        )
        
        cva = np.mean(losses)
        cva_std = np.std(losses) / np.sqrt(n_paths)
        
        # Expected Positive Exposure profile
        time_grid = np.linspace(0.1, max(redemption_times), 20)
        epe_profile = []
        for t in time_grid:
            # Paths still alive at time t
            alive = redemption_times > t
            if np.sum(alive) > 0:
                epe = np.mean(np.maximum(payoffs[alive] - notional * np.exp(-self.r * t), 0))
            else:
                epe = 0.0
            epe_profile.append(epe)
        
        return {
            'cva': cva,
            'cva_bps': cva / notional * 10000,
            'cva_std': cva_std,
            'cva_pct': cva / notional,
            'default_prob': np.mean(default_before_redemption),
            'avg_exposure': np.mean(exposure),
            'max_exposure': np.max(exposure),
            'epe_times': time_grid,
            'epe_profile': np.array(epe_profile),
            'risk_free_price': np.mean(payoffs),
            'cva_adjusted_price': np.mean(payoffs) - cva,
        }
    
    def cva_by_credit_quality(
        self,
        payoffs: np.ndarray,
        redemption_times: np.ndarray,
        notional: float
    ) -> dict:
        """Compute CVA across different credit qualities."""
        spreads = {
            'AAA (10bps)': 0.001,
            'AA (25bps)': 0.0025,
            'A (50bps)': 0.005,
            'BBB (100bps)': 0.01,
            'BB (200bps)': 0.02,
            'B (400bps)': 0.04,
        }
        
        results = {}
        for name, spread in spreads.items():
            credit = CreditParams(cds_spread=spread, recovery_rate=self.credit.recovery_rate)
            calc = CVACalculator(credit, self.r)
            res = calc.compute_cva(payoffs, redemption_times, notional)
            results[name] = res
        
        return results


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
    results = AutocallablePayoff(note, market).evaluate(paths)
    
    print("CVA Extension")
    print("=" * 50)
    
    # Base case: A-rated issuer (50bps CDS)
    credit = CreditParams(cds_spread=0.005, recovery_rate=0.40)
    print(f"Credit: {credit.describe()}\n")
    
    cva_calc = CVACalculator(credit, market.risk_free_rate)
    cva_results = cva_calc.compute_cva(
        results['payoff_distribution'],
        results['redemption_times'],
        note.notional
    )
    
    print(f"Risk-free price:    {cva_results['risk_free_price']:,.0f}")
    print(f"CVA:                {cva_results['cva']:,.0f} ({cva_results['cva_bps']:.1f} bps)")
    print(f"CVA-adjusted price: {cva_results['cva_adjusted_price']:,.0f}")
    print(f"Default probability: {cva_results['default_prob']:.2%}")
    print(f"Avg exposure:       {cva_results['avg_exposure']:,.0f}")
    
    # Credit quality comparison
    print(f"\nCVA BY CREDIT QUALITY")
    print("-" * 50)
    quality_results = cva_calc.cva_by_credit_quality(
        results['payoff_distribution'],
        results['redemption_times'],
        note.notional
    )
    
    print(f"{'Rating':<16} {'CVA':>10} {'CVA (bps)':>10} {'Adj Price':>12}")
    print("-" * 50)
    for name, res in quality_results.items():
        print(f"{name:<16} {res['cva']:>10,.0f} {res['cva_bps']:>9.1f} {res['cva_adjusted_price']:>12,.0f}")
