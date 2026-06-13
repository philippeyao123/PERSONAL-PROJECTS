"""Regenerate all README figures: python -m vol_arb.make_figures"""
from __future__ import annotations

import logging

from vol_arb.data.loader import VolDataLoader, forward_realized_vol
from vol_arb.pipeline import run_dispersion, run_vrp
from vol_arb.plots import (
    plot_correlation,
    plot_equity_curves,
    plot_iv_vs_rv,
    plot_return_distribution,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
FIG = "docs/figures"


def main() -> None:
    data = VolDataLoader().load(use_cache=True)
    core = data["core"]
    iv = core["VIX"] / 100.0
    fwd = forward_realized_vol(core["SPX"], 21)
    plot_iv_vs_rv(iv, fwd, f"{FIG}/iv_vs_rv.png")

    vrp = run_vrp(data)
    plot_equity_curves(vrp["equity"], vrp["short_equity"],
                       f"{FIG}/equity_curves.png")
    plot_return_distribution(vrp["bt"]["pnl_net"].iloc[::21],
                             f"{FIG}/pnl_distribution.png")

    disp = run_dispersion(data)
    plot_correlation(disp["implied_corr"], disp["realized_corr"],
                     f"{FIG}/correlation.png")
    logging.info("Figures written to %s/", FIG)


if __name__ == "__main__":
    main()
