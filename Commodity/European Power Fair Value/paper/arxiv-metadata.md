# arXiv submission metadata

## Title

European Power Fair Value: Reproducible Day-Ahead Forecasting and
Prompt-Proxy Diagnostics for the Germany-Luxembourg Market

## Authors

Bathaix Philippe-Emmanuel Yao

## Abstract

This paper presents a reproducible day-ahead electricity-price forecasting
study for the Germany-Luxembourg market. It uses a strict one-year
expanding-window experiment with weekly naive, Ridge, and deterministic
LightGBM forecasts, delivery-day uncertainty intervals, autocorrelation-robust
loss comparisons, feature-family ablations, regime and tail diagnostics, and
strictly prequential residual intervals. The primary specification excludes a
weather archive that does not preserve a fixed D-1 vintage. A separate
prompt-proxy analysis is reported as a descriptive directional diagnostic,
not as tradable forward-market P&L. Source code, frozen CSV evidence, tests,
figures, and an autonomous arXiv source archive accompany the manuscript.

## Categories

- Primary: q-fin.ST
- Cross-list candidates: stat.ML, econ.EM

## Comments

22 pages, 14 figures, 10 tables. Python research artifact and frozen evidence
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
