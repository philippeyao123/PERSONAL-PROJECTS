"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from systematic_research.config import ExperimentConfig
from systematic_research.examples.flagship import run_flagship
from systematic_research.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sysresearch")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the flagship point-in-time experiment")
    run.add_argument("--config", default="configs/flagship.yaml")
    run.add_argument("--output", default="reports/flagship")
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    options = build_parser().parse_args(arguments)
    logger = configure_logging()
    if options.command == "run":
        config = ExperimentConfig.from_yaml(options.config)
        logger.info(
            "starting experiment",
            extra={"context": {"experiment_id": config.experiment_id, "seed": config.seed}},
        )
        report = run_flagship(config, Path(options.output))
        logger.info("report complete", extra={"context": {"report": str(report.markdown)}})
        print(report.markdown)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
