# Validation matrix

| Component | Independent reference/property | Tolerance |
|---|---|---:|
| Flat curve | \(P(0,T)=e^{-rT}\) | \(10^{-14}\) |
| Interpolation | every supplied node is reproduced | \(10^{-14}\) |
| Swap | par coupon has zero NPV; floating leg telescopes | \(10^{-8}\) currency |
| Black-76 | published closed form; put-call parity | \(10^{-6}\), \(10^{-12}\) |
| Bachelier | ATM closed form and finite-difference signs | \(10^{-12}\) |
| Numerics | \(\sqrt 2\), integral of sine, quadratic minimizer | \(10^{-9}\) |
| G2++ bonds | initial curve reproduction for 0.25–20Y | \(10^{-14}\) |
| G2++ covariance | non-negative variances/determinant | \(10^{-15}\) floor |
| European swaption | independent time-stepped MC | 5 standard errors + 2 bp PV |
| Calibration | finite RMSE and constrained parameters | parameter bounds |
| LSM | one date equals European; Bermudan ≥ European | \(10^{-14}\) |
| DV01 | sum of nodes vs parallel bump | 2% |
| Vega | weighted buckets vs global bump | 0.1% |
| XVA | non-negativity; netting cannot raise EPE | numerical epsilon |

## Python/QuantLib cross-check

`scripts/python_reference.py` independently computes the curve, swap, Black-76 and Bachelier
reference cases using Python's standard library. If QuantLib-Python is installed, the script also
prints QuantLib values for the same analytic options. This keeps the default build lightweight
while making an external library check one command away.

Reference values frozen for non-regression:

| Case | qf-rates expected |
|---|---:|
| Black ATM, F=K=100, 20%, T=1, DF=.95 | 7.5672890826 |
| Bachelier ATM, F=K=.03, vol=.01, T=1, DF=.97 | .0038697401 |
| Flat 3% curve, P(0,5) | .8607079764 |

On 2026-07-27 the script was run against QuantLib-Python 1.43. Black-76 and Bachelier matched
QuantLib to all 12 printed decimal places (`7.567289082636` and `0.003869740120` respectively).
The optional pybind11 3.0.4 module was also compiled and imported with Python 3.9; its Black-76
price printed `7.567289082636`.

Monte-Carlo tests use fixed seeds. Their assertions include statistical uncertainty rather than
requiring bitwise equality across standard-library implementations.
