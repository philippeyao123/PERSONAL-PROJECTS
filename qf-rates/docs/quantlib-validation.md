# QuantLib validation report

## Environment and command

The cross-check was executed on 2026-07-29 with Python 3.9, QuantLib-Python 1.43, pybind11 3.0.4 and
the Release qf-rates Python module:

```bash
PYTHONPATH=build-py python3 scripts/python_reference.py \
  --require-quantlib --require-bindings
```

The script contains assertions, so a tolerance breach returns a non-zero process status. The
standalone and monorepo GitHub Actions workflows run the same command.

## Analytic options

| Instrument | qf-rates/Python reference | QuantLib | Difference |
|---|---:|---:|---:|
| Black-76 ATM call | 7.567289082636 | 7.567289082636 | < \(10^{-12}\) |
| Bachelier ATM call | 0.003869740120 | 0.003869740120 | < \(10^{-12}\) |

## Vanilla swap

The test uses a continuously compounded flat 2.5% curve, 2Y start, 7Y maturity, annual fixed
payments, semiannual floating payments, a 3% payer fixed coupon and notional 1,000,000. QuantLib
uses `SimpleDayCounter`, `NullCalendar` and unadjusted schedules to align its date conventions with
qf-rates year-fraction schedules.

| Metric | qf-rates | QuantLib | Absolute difference |
|---|---:|---:|---:|
| Par rate | 0.025315120524 | 0.025315120524 | < \(10^{-12}\) |
| NPV | -20,684.88039279 | -20,684.88039279 | < \(10^{-7}\) |

This is a direct validation of fixed-leg accruals, floating forwards, discounting, direction and
notional scaling.

## G2++ European swaption

Both libraries use \(a=0.10\), \(b=0.30\), \(\sigma=0.01\), \(\eta=0.015\) and
\(\rho=-0.70\) on the same flat curve and ATM 2Y×5Y payer swaption.

| Engine | Present value |
|---|---:|
| qf-rates order-8 joint Gaussian quadrature | 13,527.73806635 |
| QuantLib `G2SwaptionEngine` (range 7, 64 intervals) | 13,430.98886655 |
| Relative difference | 0.720343% |

The implementations do not share scheduling or numerical integration code, so
exact equality is not expected. This single anchor is supplemented by the
systematic grid below and by qf-rates' independent Monte Carlo
confidence-interval test.

## Systematic G2++ grid

The cross-engine surface contains 135 payer swaptions: five parameter regimes,
expiries 1Y/2Y/5Y, tenors 2Y/5Y/10Y, and moneyness -100/0/+100 basis points.
For relative-error summaries, a material cell is declared as a QuantLib price
of at least one basis point of notional.

| Regime | Cells | Material | Median relative difference | 95th percentile | Maximum absolute difference (bp notional) |
|---|---:|---:|---:|---:|---:|
| Low volatility | 27 | 21 | 0.176% | 1.351% | 1.173 |
| Base | 27 | 27 | 0.292% | 2.867% | 2.170 |
| High volatility | 27 | 27 | 0.184% | 1.120% | 4.559 |
| Fast mean reversion | 27 | 21 | 0.199% | 5.214% | 0.678 |
| Weak correlation | 27 | 27 | 0.169% | 1.147% | 1.519 |

Across the 123 material cells, the median and 95th-percentile absolute relative
differences are 0.184% and 2.486%. The maximum is 9.518%; the maximum absolute
difference over all 135 cells is 4.559 basis points of notional. Raw,
non-rounded results are in `paper/data/quantlib_g2pp_grid.csv`.

## Propagation to risk

Four representative ATM cases are revalued after a fixed-strike parallel
one-basis-point downward curve shift and after jointly increasing G2++
`sigma` and `eta` by one basis point. Maximum cross-engine relative differences
are 4.69% for curve DV01 and 1.38% for the joint model-volatility bump. The
latter is a controlled model-parameter diagnostic, not market-quote vega and
not a recalibrated sensitivity. Raw results are in
`paper/data/g2pp_risk_validation.csv`.

## Calibration

Both engines calibrate to six absolute normal-volatility quotes:

| Expiry | Tenor | Strike | Normal volatility |
|---:|---:|---:|---:|
| 1Y | 5Y | 0.026 | 0.0060 |
| 2Y | 5Y | 0.026 | 0.0065 |
| 3Y | 5Y | 0.026 | 0.0068 |
| 1Y | 10Y | 0.026 | 0.0064 |
| 2Y | 10Y | 0.026 | 0.0069 |
| 3Y | 10Y | 0.026 | 0.0072 |

| Engine | Volatility RMSE |
|---|---:|
| qf-rates bounded coordinate search | 0.0001868800 |
| QuantLib Levenberg–Marquardt | 0.0003867077 |

| Engine | \(a\) | \(b\) | \(\sigma\) | \(\eta\) | \(\rho\) |
|---|---:|---:|---:|---:|---:|
| qf-rates | 0.01374512 | 0.27060742 | 0.01000000 | 0.01336102 | -0.78869141 |
| QuantLib | 0.00013715 | 0.00946185 | 0.00013779 | 0.00946185 | -0.76345702 |

Both RMSE values are below the declared 0.001 acceptance threshold, while parameters differ
materially. This is expected on a sparse six-point surface: G2++ parameters are weakly identifiable
and optimizer-dependent. The validation conclusion is therefore about price/volatility fit and
parameter bounds, not equality of parameter vectors.
