#!/usr/bin/env python3
"""Run the complete flagship experiment from the repository root."""

from pathlib import Path

from systematic_research.config import ExperimentConfig
from systematic_research.examples.flagship import run_flagship

if __name__ == "__main__":
    configuration = ExperimentConfig.from_yaml("configs/flagship.yaml")
    report = run_flagship(configuration, Path("reports/flagship"))
    print(f"Report: {report.markdown}")
