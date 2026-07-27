#!/usr/bin/env bash
set -euo pipefail

ruff format --check .
ruff check .
mypy src/systematic_research
pytest --cov=systematic_research --cov-report=term-missing
python -m build

