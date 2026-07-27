# Systematic Research

A **point-in-time** Python quantitative research library designed to turn a hypothesis into a
reproducible and auditable result. It covers data, features and signals, portfolio construction,
cost-aware backtesting, walk-forward validation, risk, capacity, attribution, and reporting.

This repository provides research infrastructure, not a collection of historical performance
claims. The examples use deterministic synthetic data only.

## Core guarantees

- normalized long-form data with `date`, `asset`, `price`, `volume`, and `available_at`;
- historical investment universes that retain removed assets and prevent survivorship bias;
- features and normalizations estimated using information available in the past only;
- explicit execution lag and signal-availability controls;
- linear costs, square-root impact, turnover, participation, and capacity curves;
- gross/net leverage and concentration limits, volatility, VaR/CVaR, and drawdowns;
- rolling or expanding validation with train/validation/test windows, purging, and embargo;
- PSR, DSR, IC, IR, IC decay, benchmarks, and placebo tests;
- deterministic experiment tracking through configuration, seeds, data hashes, and versions;
- typed package, 40 tests, more than 80% branch coverage, Ruff, mypy, and CI.

## Installation

Python 3.9 through 3.12 is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

To reproduce the exact Python 3.9 validation environment:

```bash
python -m pip install -r requirements/lock-py39.txt
python -m pip install -e . --no-deps
```

## Flagship experiment

```bash
sysresearch run \
  --config configs/flagship.yaml \
  --output reports/flagship
```

The pipeline generates a synthetic market that includes a delisting, calculates a lagged
momentum signal, constructs a constrained long/short portfolio, applies costs and market impact,
creates walk-forward folds, and exports:

- `report.md`, `equity_curve.png`, and `drawdown.png`;
- `metrics.json` and `metadata.json`;
- `daily_results.csv`, `positions.csv`, and `capacity.csv`.

An equivalent Python entry point is available in `examples/run_flagship.py`. The
`systematic_research.examples.tsmom` and `systematic_research.examples.stat_arb` modules
demonstrate the APIs with time-series momentum and convergence signals.

## Minimal usage

```python
from systematic_research.backtest.engine import VectorizedBacktester
from systematic_research.features.factors import Momentum
from systematic_research.signals.pipeline import SignalPipeline

feature = Momentum(lookback=63, lag=1)
targets = SignalPipeline(feature, normalization="cross_sectional_rank").run(market_data)
result = VectorizedBacktester(execution_lag=1).run(returns, targets)
result.assert_reconciled()
```

`targets` must contain `date`, `asset`, `target_weight`, and `available_at`. A target for which
`available_at > date` is rejected.

## Quality checks

```bash
ruff format --check .
ruff check .
mypy src/systematic_research
pytest --cov=systematic_research --cov-report=term-missing
python -m build
```

Alternatively, run `scripts/run_all.sh`.

## Documentation

- [Architecture](docs/architecture.md)
- [Point-in-time contract](docs/point-in-time.md)
- [Statistical validation](docs/validation.md)
- [Performance and complexity](docs/performance.md)
- [Technical note and limitations](docs/technical-note.md)
- [Export schemas](docs/schemas.md)
- [Release checklist](docs/release-checklist.md)

## Status

Version `0.1.0`, alpha API. Models and estimates produced by this software are intended for
research and do not constitute investment advice.

