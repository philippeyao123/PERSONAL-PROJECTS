#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/european-power-arxiv.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/generated" "$STAGE/figures" "$ROOT/output/pdf"
cp "$ROOT/paper/main.tex" "$STAGE/main.tex"
cp "$ROOT/paper/references.bib" "$STAGE/references.bib"
cp "$ROOT/paper/main.bbl" "$STAGE/main.bbl"
cp "$ROOT"/paper/generated/*.tex "$STAGE/generated/"
cp "$ROOT"/paper/figures/*.pdf "$STAGE/figures/"

ARCHIVE="$ROOT/output/pdf/european-power-fair-value-arxiv-source.tar.gz"
COPYFILE_DISABLE=1 tar \
  --exclude='._*' \
  --exclude='.DS_Store' \
  -czf "$ARCHIVE" \
  -C "$STAGE" .
echo "arXiv source archive: $ARCHIVE"
