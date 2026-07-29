# Frozen numerical results

These CSV files are the machine-readable inputs for the paper figures and tables.

- `variance_reduction.csv`, `lsm_convergence.csv`,
  `g2pp_monte_carlo_convergence.csv`, `g2pp_time_step_convergence.csv`,
  `g2pp_stress_grid.csv`, `g2pp_multistart_calibration.csv`,
  `g2pp_calibration_residuals.csv`, `lsm_out_of_sample.csv`,
  `wrong_way_risk.csv`, and `wrong_way_risk_grid.csv` are emitted by
  `qf_rates_validation`. The LSM out-of-sample file includes exercise-date and
  non-exercise probabilities; the WWR grid retains every beta and exposure
  profile used in the detailed sensitivity study.
- `quantlib_validation.csv`, `quantlib_g2pp_grid.csv`,
  `g2pp_risk_validation.csv`, and `calibration_validation.csv` are emitted by
  the deterministic Python/QuantLib cross-check implemented in
  `scripts/python_reference.py`. The G2++ comparison grid contains five
  parameter regimes, three expiries, three underlying tenors, and three
  moneyness levels (135 cells). Its summary treats a QuantLib price of at least
  one basis point of notional as material when reporting relative errors.
  Curve DV01 is a fixed-strike parallel one-basis-point curve bump. The
  model-volatility sensitivity is the price change from jointly increasing
  G2++ \(\sigma\) and \(\eta\) by one basis point; it is a controlled
  cross-engine diagnostic, not a market-quote vega.

All prices are present values in the notional currency. Normal volatilities are in
absolute units. Seeds, path counts, model parameters, instruments, and numerical
settings are fixed in the corresponding source programs. Recreate the complete
directory with `scripts/reproduce_paper.sh`.
