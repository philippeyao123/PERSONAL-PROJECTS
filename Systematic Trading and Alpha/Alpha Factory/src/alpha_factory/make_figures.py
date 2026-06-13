"""Regenerate all figures used in the README and reports.

Usage::

    python -m alpha_factory.make_figures

Writes PNGs to docs/figures/. Uses the cached parquet data if present so the
script runs offline and deterministically.
"""
from __future__ import annotations

import logging

from alpha_factory.diagnostics.plots import (
    plot_capacity,
    plot_equity_curve,
    plot_factor_ic,
    plot_tsmom_decay,
)
from alpha_factory.diagnostics.tsmom_replication import (
    TimeSeriesMomentum,
    load_tsmom_proxies,
)
from alpha_factory.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("make_figures")

FIG_DIR = "docs/figures"


def main() -> None:
    logger.info("Running synthetic pipeline for demo figures...")
    out = run_pipeline(n_trials=50)
    res = out["result"]

    plot_equity_curve(
        res.gross_returns, res.net_returns,
        f"{FIG_DIR}/equity_curve.png",
        "Multi-Asset Alpha Factory (synthetic, planted signal)",
    )
    plot_factor_ic(out["ic_summary"], f"{FIG_DIR}/factor_ic.png")
    plot_capacity(out["capacity"], f"{FIG_DIR}/capacity.png")

    logger.info("Running TSMOM replication for decay figure...")
    tres = TimeSeriesMomentum().run(load_tsmom_proxies())
    plot_tsmom_decay(tres.by_period, f"{FIG_DIR}/tsmom_decay.png")

    logger.info("All figures written to %s/", FIG_DIR)


if __name__ == "__main__":
    main()
