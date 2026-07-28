#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"
"$PYTHON_BIN" src/qa.py
"$PYTHON_BIN" src/models.py
"$PYTHON_BIN" src/trading.py
"$PYTHON_BIN" src/research.py
"$PYTHON_BIN" src/make_figures.py
"$PYTHON_BIN" scripts/generate_paper_artifacts.py
"$PYTHON_BIN" -m pytest -q
bash scripts/build_paper.sh
