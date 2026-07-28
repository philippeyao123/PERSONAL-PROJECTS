# Changelog

## 1.0.0 - 2026-07-28

- Expanded the walk-forward test from 180 to 365 local delivery days.
- Made the primary model deterministic and restricted it to documented D-1
  price, TSO renewable-forecast, and calendar inputs.
- Reclassified stitched historical weather as a non-primary sensitivity
  because it does not preserve a fixed day-ahead lead time.
- Added delivery-day bootstrap intervals and HAC loss-differential tests.
- Added prequential 90% residual intervals with hourly coverage diagnostics.
- Added feature-family ablations, seasonal/hourly regimes, and price-tail
  evaluation.
- Corrected right-censoring in the prompt-proxy experiment and added a
  moving-block bootstrap plus threshold/window sensitivity.
- Added ten executable tests, fourteen vector figures, generated LaTeX
  tables, an arXiv manuscript, citation metadata, and source packaging.
