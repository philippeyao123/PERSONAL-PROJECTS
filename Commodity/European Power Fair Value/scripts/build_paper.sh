#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TECTONIC_BIN="${TECTONIC_BIN:-tectonic}"

"$PYTHON_BIN" "$ROOT/scripts/generate_paper_artifacts.py"
mkdir -p "$ROOT/output/pdf"
rm -f \
  "$ROOT/output/pdf/main.aux" \
  "$ROOT/output/pdf/main.bbl" \
  "$ROOT/output/pdf/main.blg" \
  "$ROOT/output/pdf/main.log" \
  "$ROOT/output/pdf/main.out" \
  "$ROOT/output/pdf/main.pdf"

(
  cd "$ROOT/paper"
  "$TECTONIC_BIN" main.tex \
    --outdir "$ROOT/output/pdf" \
    --keep-logs \
    --keep-intermediates
)

cp "$ROOT/output/pdf/main.pdf" \
  "$ROOT/output/pdf/european-power-fair-value-paper.pdf"
cp "$ROOT/output/pdf/main.bbl" "$ROOT/paper/main.bbl"
echo "paper: $ROOT/output/pdf/european-power-fair-value-paper.pdf"
