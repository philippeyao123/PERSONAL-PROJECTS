# Changelog

## 1.1.0 - 2026-07-29

- Refocused the manuscript on auditability, leakage control, and conditional
  failure maps rather than a generic model-performance claim.
- Decoupled the primary complete-case sample from the optional weather
  sensitivity and documented the Energy-Charts endpoint semantics.
- Added a stronger 24-model hour-specific Ridge baseline, first-90-day
  sensitivity, and prequential bias correction.
- Replaced independent day resampling with a seven-day moving-block bootstrap
  and added Harvey-Leybourne-Newbold finite-sample corrections.
- Added formal hour-by-hour coverage tests with Holm adjustment, interval
  score and pinball diagnostics, and moving-block signal uncertainty.
- Bounded the prompt-proxy result with no-signal, naive, linear-model, and
  infeasible perfect-information references.
- Replaced manuscript result literals with generated macros, expanded the
  literature, switched the paper to A4, and hardened arXiv packaging against
  AppleDouble files.
- Expanded the executable test suite from ten to thirteen tests.

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
