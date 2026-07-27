# Statistical validation

Validation separates three functions:

- **train**: estimate parameters;
- **validation**: choose among candidates;
- **test**: evaluate the selected candidate.

`walk_forward_splits` creates rolling or expanding windows. `purge_periods` removes the end of the
training window when labels extend beyond it. `embargo_periods` enforces a gap before the test
window. `nested_select` chooses exclusively on validation data and calls the evaluator once on
the test data.

Raw Sharpe ratio is never sufficient. The report includes:

- PSR against a declared benchmark;
- DSR against the expected maximum across attempted trials;
- cross-sectional IC, IC information ratio, and decay by horizon;
- turnover, drawdown, VaR/CVaR, exposures, and concentration;
- parameter sensitivity and subperiod performance;
- delayed, randomized, and permuted-label placebos;
- cash, market, and equal-weight benchmarks.

DSR depends on the number and dispersion of the trials actually attempted. Those trials must be
recorded rather than reconstructed after selection. The flagship experiment's synthetic results
validate the pipeline; they do not demonstrate investable performance.

