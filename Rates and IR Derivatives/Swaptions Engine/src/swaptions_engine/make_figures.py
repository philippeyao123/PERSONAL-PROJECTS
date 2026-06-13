"""Regenerate all README figures: calibration fit, Bermudan premium, DV01.

    python -m swaptions_engine.make_figures
"""
from __future__ import annotations

import logging

import numpy as np

from swaptions_engine.calibration.calibrate import G2ppCalibrator, SurfacePoint
from swaptions_engine.curve.yield_curve import Curve
from swaptions_engine.instruments.swap import Swaption, VanillaSwap
from swaptions_engine.models.g2pp import G2pp
from swaptions_engine.plots import (
    plot_bermudan_premium,
    plot_bucketed_dv01,
    plot_calibration_fit,
)
from swaptions_engine.pricing.bermudan import BermudanLSM
from swaptions_engine.risk.engine import SwaptionRisk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("make_figures")
FIG = "docs/figures"


def main() -> None:
    curve = Curve.flat(0.025)
    surface = [
        SurfacePoint(1, 5, 0.0075), SurfacePoint(2, 5, 0.0080),
        SurfacePoint(5, 5, 0.0078), SurfacePoint(1, 10, 0.0072),
        SurfacePoint(2, 10, 0.0076), SurfacePoint(5, 10, 0.0074),
        SurfacePoint(3, 7, 0.0079), SurfacePoint(7, 7, 0.0070),
        SurfacePoint(10, 10, 0.0065),
    ]
    logger.info("Calibrating G2++...")
    res = G2ppCalibrator(curve).calibrate(surface)
    plot_calibration_fit(res.fit_table, res.rmse_vol, f"{FIG}/calibration_fit.png")

    model = G2pp(curve, res.params)
    sw0 = VanillaSwap(1.0, 5.0, 0.0)
    fwd = sw0.forward_swap_rate(curve)
    swap = VanillaSwap(1.0, 5.0, fwd)
    euro = model.european_swaption(swap, payer=True)
    lsm = BermudanLSM(model)
    logger.info("Pricing Bermudans...")
    ns, prices = [1, 2, 3, 5], []
    for n in ns:
        ex = np.linspace(1.0, 5.0, n) if n > 1 else np.array([1.0])
        style = "bermudan" if n > 1 else "european"
        prices.append(
            lsm.price(Swaption(swap, ex, style), n_paths=30_000, seed=5)["price"]
        )
    plot_bermudan_premium(ns, prices, euro, f"{FIG}/bermudan_premium.png")

    sw2 = VanillaSwap(2.0, 5.0, VanillaSwap(2.0, 5.0, 0.0).forward_swap_rate(curve))
    rep = SwaptionRisk(curve, res.params).compute(
        lambda m: m.european_swaption(sw2, True) * 1e6
    )
    plot_bucketed_dv01(
        {k: v for k, v in rep.bucketed_dv01.items() if abs(v) > 0.1},
        f"{FIG}/bucketed_dv01.png",
    )
    logger.info("Figures written to %s/", FIG)


if __name__ == "__main__":
    main()
