"""Tables, charts and a compact PM-ready research summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from systematic_research.backtest.engine import BacktestResult
from systematic_research.reporting.export import export_frame, export_mapping
from systematic_research.risk import drawdown_series


@dataclass(frozen=True)
class ResearchReport:
    directory: Path
    markdown: Path
    equity_chart: Path
    drawdown_chart: Path
    metrics_json: Path


def _plot_series(series: pd.Series, title: str, ylabel: str, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(series.index, series.values, color="#155EEF", linewidth=1.5)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def generate_report(
    result: BacktestResult,
    metrics: Mapping[str, Any],
    output_directory: Union[str, Path],
    *,
    capacity: Optional[pd.DataFrame] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ResearchReport:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    daily = result.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    equity = (1.0 + daily.set_index("date")["net_return"]).cumprod()
    drawdown = drawdown_series(daily.set_index("date")["net_return"])
    equity_chart = output / "equity_curve.png"
    drawdown_chart = output / "drawdown.png"
    _plot_series(equity, "Net equity curve", "Growth of 1", equity_chart)
    _plot_series(drawdown, "Drawdown", "Drawdown", drawdown_chart)
    export_frame(daily, output / "daily_results.csv")
    export_frame(result.positions, output / "positions.csv")
    if capacity is not None:
        export_frame(capacity, output / "capacity.csv")
    metrics_json = export_mapping(metrics, output / "metrics.json")
    if metadata is not None:
        export_mapping(metadata, output / "metadata.json")
    metric_rows = "\n".join(
        f"| {name} | {value:.6f} |" if isinstance(value, (float, int)) else f"| {name} | {value} |"
        for name, value in sorted(metrics.items())
    )
    capacity_section = ""
    if capacity is not None:
        capacity_section = (
            "\n## Capacity\n\n" + capacity.to_markdown(index=False, floatfmt=".4f") + "\n"
        )
    markdown = output / "report.md"
    body = f"""# Systematic Research Report

## Executive summary

This report was generated from a point-in-time experiment with explicit execution lag,
transaction costs and square-root market impact. Development and final test windows are separated.

## Performance and risk

| Metric | Value |
|---|---:|
{metric_rows}

## Equity and drawdown

![Net equity curve](equity_curve.png)

![Drawdown](drawdown.png)
{capacity_section}
## Reproducibility

The exact configuration hash, data hash, seed, runtime versions and platform are stored in
`metadata.json`. Daily accounting and positions are exported with stable schemas.
"""
    markdown.write_text(body, encoding="utf-8")
    return ResearchReport(output, markdown, equity_chart, drawdown_chart, metrics_json)
