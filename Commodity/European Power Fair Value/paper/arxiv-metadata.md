# arXiv submission metadata

## Title

Auditable Day-Ahead Electricity Price Forecasting: Leakage Control,
Conditional Failure Maps, and Walk-Forward Evidence from Germany-Luxembourg

## Authors

Bathaix Philippe-Emmanuel Yao

## Abstract

This paper presents an auditable day-ahead electricity-price forecasting study
for the Germany-Luxembourg market. It uses a strict one-year expanding-window
experiment with weekly-naive, pooled Ridge, hour-specific Ridge, and
deterministic LightGBM forecasts. Inference uses delivery-day moving-block
intervals and autocorrelation-robust loss comparisons with a finite-sample
correction. The artifact reports feature ablations, regime and tail
diagnostics, proper interval scores, and formal hour-conditional coverage
tests. The primary specification excludes a weather archive that does not
preserve a fixed D-1 vintage. A separate prompt-proxy analysis is bounded by
no-signal, model, and infeasible perfect-information references and is not
presented as tradable forward-market P&L. Frozen evidence, tests, figures, and
a self-contained arXiv source archive accompany the manuscript.

## Categories

- Primary: q-fin.ST
- Cross-list candidates: stat.ML, econ.EM

## Comments

24 pages, 14 figures, 11 tables. Python research artifact and frozen evidence
included in the companion repository.

## Keywords

electricity price forecasting; day-ahead market; DE-LU; LightGBM; renewable
forecasts; conformal intervals; walk-forward validation; reproducible
research.

## Submission checklist

- Compile `main.tex` from the source archive in an empty directory.
- Confirm the title, author spelling, abstract, categories, and comments.
- Select the arXiv license during submission.
- Replace repository metadata with a versioned release URL after publication.
- Add the arXiv identifier and DOI to `CITATION.cff` after acceptance.
