# Suggested arXiv metadata

## Title

Reproducible Numerical Validation of G2++ Swaption Pricing: Quadrature, Monte
Carlo, Calibration, and Bermudan Exercise

## Author

Bathaix Philippe-Emmanuel Yao

## Categories

- Primary: `q-fin.CP` — Computational Finance
- Secondary: `q-fin.PR` — Pricing of Securities

The final category assignment is decided by arXiv moderation. A first-time
submission may also require endorsement.

## Comments

34 pages, 16 figures, 13 tables. Accompanying open-source C++20 artifact with
34 automated tests and fully reproducible numerical results.

## License

Recommended arXiv submission license: CC BY 4.0 for the manuscript. The software
remains MIT licensed. Choose the arXiv license explicitly during submission.

## Abstract

Numerical implementations of the same interest-rate model can disagree because
of integration rules, discretization, conventions, calibration choices, or
exercise-policy estimation. This paper presents a reproducible validation
protocol for G2++ swaption pricing based on three-dimensional Gauss-Hermite
quadrature, independently time-stepped Monte Carlo, and QuantLib. The external
comparison spans 135 European payer swaptions across five parameter regimes,
three expiries, three tenors, and three moneyness levels. Among the 123 cells
worth at least one basis point of notional in QuantLib, the absolute relative
price difference has median 0.184% and 95th percentile 2.486%; the maximum
absolute price gap over the full grid is 4.559 basis points of notional.
Fixed-strike curve DV01 and a controlled joint G2++ volatility bump agree within
4.69% and 1.38%, respectively, over four representative cases. Coupled paths
isolate time-discretization bias, while path-count experiments report Monte
Carlo confidence intervals. Eight calibration starts expose local-solution and
parameter-identification risk, and Longstaff-Schwartz policies are evaluated on
paths independent of their training samples. The accompanying qf-rates C++20
artifact regenerates all tests, CSV evidence, tables, and figures. The
contribution is a transparent error-budget and cross-engine validation study,
not a new short-rate model.

## Before submission

1. Create a public GitHub release and archive it with Zenodo.
2. Insert the resulting software DOI into `CITATION.cff`, `.zenodo.json`, and the paper.
3. Confirm the author name, manuscript license, and category.
4. Upload `output/pdf/qf-rates-arxiv-source.tar.gz` to arXiv and inspect arXiv's compiled preview.
5. Add the assigned arXiv identifier back to the README and citation metadata.

No DOI or arXiv identifier is fabricated in this repository.
