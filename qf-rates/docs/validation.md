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
| LSM | one date equals European; convergence over paths, seeds and bases | \(10^{-14}\), statistical |
| DV01 | sum of nodes vs parallel bump | 2% |
| Vega | weighted buckets vs global bump | 0.1% |
| Volatility scenario | up shock raises and down shock lowers payer value | strict sign |
| XVA | non-negativity; netting cannot raise EPE; positive WWR beta raises CVA | numerical epsilon |

## Python/QuantLib cross-check

`scripts/python_reference.py` independently computes curve, Black-76 and Bachelier reference cases.
With the optional bindings and QuantLib-Python it compares a 2Y×5Y swap, the G2++ European price and
a six-quote normal-volatility calibration. CI requires these optional dependencies and fails when a
tolerance is breached.

Reference values frozen for non-regression:

| Case | qf-rates expected |
|---|---:|
| Black ATM, F=K=100, 20%, T=1, DF=.95 | 7.5672890826 |
| Bachelier ATM, F=K=.03, vol=.01, T=1, DF=.97 | .0038697401 |
| Flat 3% curve, P(0,5) | .8607079764 |

On 2026-07-27 the script was run against QuantLib-Python 1.43 and pybind11 3.0.4. Analytic options
matched to all 12 printed decimal places. Swap par rate and NPV matched at machine precision. The
G2++ European price differed by 0.720343%, inside the declared 1% cross-implementation tolerance.
The qf-rates and QuantLib calibration volatility RMSEs were respectively 0.0001868800 and
0.0003867077. Full inputs, outputs and interpretation are in
[`quantlib-validation.md`](quantlib-validation.md).

## Measured stochastic validation

The `qf_rates_validation` executable freezes a reproducible comparison workload:

| Estimator | Standard error | Variance ratio vs plain |
|---|---:|---:|
| Plain | 0.04170130 | 1.00000000 |
| Antithetic | 0.03328838 | 0.63721512 |
| Control variate | 0.01898809 | 0.20733054 |

The control variate reduces measured variance by 79.27%; antithetic sampling reduces it by 36.28%.
These are workload-specific measured results, not universal claims.

The same executable evaluates LSM at 2,000, 5,000 and 10,000 paths, seeds 7 and 42, and both linear
and quadratic bases. Standard error falls from approximately 0.00036 at 2,000 paths to 0.00016 at
10,000 paths. Across the 10,000-path cases, prices span 0.01366320–0.01384655, making seed and basis
sensitivity explicit.

For the deterministic WWR test workload, independent exposure produces CVA 1312.45468831 and proxy
WWR with beta 20 produces CVA 1636.07844166. The test also checks the pathwise exposure relationship
instead of relying only on the aggregated CVA value.

Monte-Carlo tests use fixed seeds. Their assertions include statistical uncertainty rather than
requiring bitwise equality across standard-library implementations.
