"""Risk engine for swaptions: bucketed DV01, vega ladder, and scenarios.

All sensitivities are computed by bump-and-revalue, which is model-agnostic and
matches how a desk's risk system actually reports. The pricing function passed
in can be either the semi-analytic European or the LSM Bermudan price.

    - Bucketed DV01: bump each curve pillar by 1bp, reprice, report dPV per
      bucket. The sum approximates parallel DV01; the buckets show where the
      curve risk lives (key-rate durations).
    - Vega: bump the calibrated normal vol surface and re-calibrate, or bump
      the model vols directly; here we bump model sigma/eta and report the
      total vega (dPV per 1bp of normal vol).
    - Scenarios: parallel shifts, steepeners/flatteners, vol shocks.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from swaptions_engine.curve.yield_curve import Curve
from swaptions_engine.models.g2pp import G2pp, G2ppParams

PriceFn = Callable[[G2pp], float]


@dataclass
class RiskReport:
    pv: float
    bucketed_dv01: dict[float, float]  # pillar time -> dPV per 1bp
    total_dv01: float
    vega: float                        # dPV per 1bp of normal vol (approx)
    scenarios: dict[str, float]        # scenario name -> PV change


class SwaptionRisk:
    """Bump-and-revalue risk for a swaption priced under G2++."""

    def __init__(self, curve: Curve, params: G2ppParams) -> None:
        self.curve = curve
        self.params = params

    def compute(self, price_fn: PriceFn, bump_bp: float = 1.0) -> RiskReport:
        bump = bump_bp * 1e-4
        base_model = G2pp(self.curve, self.params)
        pv = price_fn(base_model)

        # --- Bucketed DV01: bump each curve pillar up by 1bp ---
        buckets: dict[float, float] = {}
        for i, t in enumerate(self.curve.t):
            up = G2pp(self.curve.bump_pillar(i, bump), self.params)
            buckets[float(t)] = (price_fn(up) - pv) / bump_bp
        total_dv01 = sum(buckets.values())

        # --- Vega: bump both factor vols to mimic a +1bp normal-vol shift ---
        # Scale sigma/eta by a small relative bump that maps ~1bp of ATM vol.
        rel = 0.01
        bumped = G2ppParams(
            a=self.params.a, sigma=self.params.sigma * (1 + rel),
            b=self.params.b, eta=self.params.eta * (1 + rel),
            rho=self.params.rho,
        )
        pv_vol_up = price_fn(G2pp(self.curve, bumped))
        # Express per-1bp by dividing by the vol move in bp (approx).
        approx_vol_move_bp = rel * (self.params.sigma + self.params.eta) * 1e4
        vega = (pv_vol_up - pv) / max(approx_vol_move_bp, 1e-9)

        # --- Scenarios ---
        scenarios = {
            "parallel_+50bp": self._scenario_parallel(price_fn, pv, 50e-4),
            "parallel_-50bp": self._scenario_parallel(price_fn, pv, -50e-4),
            "steepener_+25bp": self._scenario_tilt(price_fn, pv, 25e-4),
            "flattener_-25bp": self._scenario_tilt(price_fn, pv, -25e-4),
            "vol_+10pct": pv_vol_up_scenario(self, price_fn, pv, 0.10),
        }

        return RiskReport(
            pv=pv, bucketed_dv01=buckets, total_dv01=total_dv01,
            vega=vega, scenarios=scenarios,
        )

    def _scenario_parallel(self, price_fn, pv, shift):
        m = G2pp(self.curve.shifted(shift), self.params)
        return price_fn(m) - pv

    def _scenario_tilt(self, price_fn, pv, slope):
        """Linear tilt around the 5y point: short end down, long end up."""
        t = self.curve.t
        pivot = 5.0
        new_r = self.curve.r + slope * (t - pivot) / (t[-1] - t[0])
        m = G2pp(Curve(t.copy(), new_r), self.params)
        return price_fn(m) - pv


def pv_vol_up_scenario(risk: SwaptionRisk, price_fn, pv, rel):
    bumped = G2ppParams(
        a=risk.params.a, sigma=risk.params.sigma * (1 + rel),
        b=risk.params.b, eta=risk.params.eta * (1 + rel), rho=risk.params.rho,
    )
    return price_fn(G2pp(risk.curve, bumped)) - pv
