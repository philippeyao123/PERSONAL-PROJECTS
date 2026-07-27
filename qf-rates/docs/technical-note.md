# qf-rates v0.1.0: architecture, calibration and validation

## Executive summary

This note describes a compact C++20 rates library designed to make pricing assumptions visible and
validation reproducible. The release implements the complete path from a discount curve to vanilla
swaps, European and Bermudan swaptions, calibrated G2++ dynamics, market risk and a simplified
counterparty-risk extension. The design emphasizes small immutable interfaces, explicit units and
two independent numerical routes for the most material model price.

The central validation result is structural: the G2++ deterministic shift reproduces the input
curve by construction, the European swaption quadrature integrates the joint Gaussian state
directly, and an independently time-stepped Monte-Carlo engine checks that deterministic result
within simulation uncertainty. Longstaff–Schwartz reduces exactly to the European engine when only
one exercise date is supplied. Risk is defined as transparent bump-and-revalue rather than an
implicit analytic approximation.

## 1. Scope

Version 0.1.0 includes continuous-time year-fraction maturities, flat and node curves, vanilla
fixed-for-floating swaps, Black-76 and Bachelier options, root finding, quadrature, constrained
optimization, reproducible pseudo-random sampling, generic Monte-Carlo aggregation, G2++ bonds and
swaptions, normal-volatility calibration, Bermudan LSM, DV01, vega, curve scenarios, exposure,
netting, proxy wrong-way risk, CVA/DVA/FVA and simplified SIMM/MVA.

Calendar dates, business-day adjustment, collateral agreements, multi-curve bootstrapping and
regulatory SIMM are intentionally outside the release. This boundary is important: the code is a
transparent research library, not a replacement for a production trade capture or risk system.

## 2. Domain conventions

Time is measured in years from valuation. A continuously compounded zero rate \(z(T)\) maps to
\(P(0,T)=\exp[-z(T)T]\). A simple forward for a coupon period \([T_i,T_{i+1}]\) is

\[
L_i = \frac{P_f(0,T_i)/P_f(0,T_{i+1})-1}{\alpha_i}.
\]

Discount and forward curves may differ at the swap API boundary, although examples use a single
curve. Fixed coupons and normal volatilities are absolute decimals. A payer swap pays fixed and
receives floating; a payer swaption is represented by a call. NPV signs are always from the
instrument holder's perspective.

## 3. Curves and interpolation

`YieldCurve` exposes only discount, zero and forward queries. `FlatYieldCurve` is the analytic base
case. `InterpolatedYieldCurve` accepts strictly increasing maturities and positive discount
factors. Linear discount interpolation is

\[
P(T)=(1-w)P(T_i)+wP(T_{i+1}),
\]

while log-linear interpolation applies the same relation to \(\log P\). Log-linear discount
interpolation produces piecewise constant continuous forward rates and preserves positivity.
Beyond the last node, the terminal continuously compounded zero rate is held flat. Every supplied
node is reproduced to machine precision.

## 4. Vanilla swaps

Each leg owns an immutable schedule, notional and pay/receive direction. The fixed-leg present value
before direction is

\[
PV_{\text{fixed}}=N K \sum_i \alpha_i P_d(0,T_i)=NKA.
\]

The floating leg discounts curve-implied simple forwards. The par rate is the unsigned floating
leg value divided by \(NA\). With a single curve and no spread, the floating coupons telescope to
\(N[P(0,T_0)-P(0,T_n)]\). These two identities are regression tests and catch sign, accrual and
payment-date defects.

## 5. Analytic option engines

Black-76 prices a call as

\[
DF\,N[F\Phi(d_1)-K\Phi(d_2)],\qquad
d_{1,2}=\frac{\log(F/K)\pm\frac12\sigma^2T}{\sigma\sqrt T}.
\]

Bachelier prices a call as

\[
DF\,N[(F-K)\Phi(d)+\sigma_N\sqrt T\,\phi(d)],\qquad
d=\frac{F-K}{\sigma_N\sqrt T}.
\]

Put prices use a sign transformation rather than duplicated formulas. Delta, gamma and vega are
returned with the price. Degenerate zero-expiry or zero-volatility cases return intrinsic value and
well-defined one-sided delta. Tests cover known values, put-call parity, non-negativity and
monotonicity.

## 6. Numerical building blocks

Bisection requires a finite sign-changing bracket and reports convergence explicitly. Adaptive
Simpson integration recursively controls the local error estimate. The online statistics class
uses Welford's recurrence, avoiding the cancellation in sum-of-squares formulas. Calibration uses
a bounded coordinate search. Although less sophisticated than Levenberg–Marquardt, it is
deterministic, cannot leave parameter bounds and behaves predictably for a five-parameter
demonstration.

Linear regression solves the six-dimensional normal equations with pivoted Gauss–Jordan
elimination and a small diagonal ridge. Production work should prefer QR or SVD; the current
approach remains adequate for the scaled two-factor polynomial basis and reports singular systems.

## 7. G2++ model

The short rate is

\[
r_t=\varphi(t)+x_t+y_t,
\quad dx_t=-a x_tdt+\sigma dW_t^x,
\quad dy_t=-b y_tdt+\eta dW_t^y,
\]

with \(dW_t^x dW_t^y=\rho dt\). Parameters satisfy \(a,b>0\), non-negative volatilities and
\(-1<\rho<1\).

Define \(B(k,\tau)=(1-e^{-k\tau})/k\). The time-\(t\) bond is

\[
P(t,T)=A(t,T)\exp[-B(a,T-t)x_t-B(b,T-t)y_t],
\]

where

\[
A(t,T)=\frac{P(0,T)}{P(0,t)}
\exp\left[\frac12\{V(T-t)-V(T)+V(t)\}\right].
\]

\(V(t)\) is the variance of \(\int_0^t(x_s+y_s)ds\), implemented in closed form. At \(t=0\),
both factor loadings multiply zero states and the variance adjustment cancels, so the market
discount curve is reproduced exactly.

Factor transitions are sampled exactly over each step. The innovation correlation is derived from
the covariance over that step; it is not incorrectly assumed equal to the Brownian correlation when
mean reversions differ.

## 8. European swaption quadrature

At expiry \(T_e\), the payer swap value per notional is

\[
1-P(T_e,T_n)-K\sum_i\alpha_iP(T_e,T_i).
\]

The present value requires the stochastic discount integral together with terminal factors. Under
the risk-neutral measure the vector

\[
\left(\int_0^{T_e}(x_s+y_s)ds,\ x_{T_e},\ y_{T_e}\right)
\]

is jointly Gaussian. The engine builds its covariance matrix from closed-form factor moments and
independently integrated cross-covariances, obtains a three-by-three Cholesky factor and evaluates
the discounted positive payoff with tensor order-8 Gauss–Hermite quadrature. This is deterministic,
uses 512 nodes, handles payer and receiver exercise, and includes the exact deterministic shift
through \(P(0,T_e)\exp[-V(T_e)/2]\).

An order other than eight is rejected rather than silently pretending that a configurable rule was
used. A later release may generate arbitrary Hermite rules from an eigenvalue algorithm.

## 9. Monte-Carlo validation

The independent engine advances both OU factors with exact marginal transitions, integrates the
short rate by the trapezoid rule and evaluates the swap payoff from terminal model bonds.
Antithetic innovations are enabled by default. Welford aggregation returns price, standard error
and a two-sided 95% confidence interval.

Time discretization affects only the path discount integral; factor endpoints are exact. Validation
therefore combines statistical error with a small deterministic allowance. A fixed seed makes
results reproducible, while assertions remain statistical because C++ standard libraries may map a
Mersenne Twister stream to normal draws differently.

The generic one-factor Monte-Carlo function also supports an estimated optimal control coefficient.
Antithetic pairs reduce odd payoff noise. Variance-reduction effectiveness is workload-dependent,
so callers should compare reported standard errors rather than assume improvement.

## 10. Calibration

Market inputs are expiry, underlying tenor, strike, absolute normal volatility and weight. For each
candidate parameter vector, the engine:

1. builds the expiry-to-maturity annual swap schedule;
2. computes the G2++ quadrature price;
3. inverts the Bachelier formula by bracketed bisection;
4. minimizes weighted volatility RMSE.

The bounds are \(a\in[0.005,1]\), \(b\in[0.01,1.5]\),
\(\sigma,\eta\in[10^{-4},0.1]\), and \(\rho\in[-0.95,0.95]\). The result includes convergence,
iterations, RMSE and an expiry/tenor table of market volatility, model volatility and basis-point
error. Bounds avoid invalid correlations and near-zero mean reversions.

Calibration is deliberately deterministic and easy to audit. Coordinate search can settle in a
local optimum and G2++ parameters are not separately identifiable on sparse grids. Multiple
starting points and regularization should be used before interpreting parameters economically.

## 11. Bermudan Longstaff–Schwartz

The LSM engine simulates states and cumulative discount factors at ordered exercise dates. At each
backward date it regresses discounted realized continuation on

\[
(1,x,y,x^2,xy,y^2)
\]

using in-the-money paths. Exercise occurs when intrinsic value exceeds estimated continuation.
Values are rolled backward with path-specific discount ratios.

With one exercise date the function delegates to the validated European quadrature engine; this is
an exact invariant rather than a noisy coincidence. For multiple dates the reported estimate is
floored at the comparable first-date European value. This protects the no-arbitrage ordering from
finite-sample regression bias but should be disclosed when the floor binds. Exercise probabilities
and standard error support convergence studies across paths, seeds, basis choices and date grids.

## 12. Market risk

DV01 uses a one-basis-point downward continuously compounded node or parallel shift:

\[
DV01=PV(z-1\text{bp})-PV(z).
\]

Both discount and forward curves are bumped in the single-curve helper. Bucketed changes should sum
to the parallel change to first order; nonlinearity creates a small residual. Scenario functions
also produce parallel up/down and linear steepener/flattener shapes.

Vega aggregates a weighted normal-volatility surface proxy. Each bucket bump changes the aggregate
volatility by its normalized weight. The sum of infinitesimal bucket vegas therefore reconciles
with the global bump, with only second-order finite-bump differences.

## 13. Exposure and XVA extension

Exposure simulation revalues a portfolio of swaps along G2++ factor paths. With netting enabled,
positive and negative exposure are taken after portfolio aggregation; without netting, trade-level
positive and negative values are summed. Identical random seeds make the netting comparison
pathwise.

A transparent proxy for wrong-way risk multiplies positive exposure by
\(\exp[\beta(x_t+y_t)]\). It is a sensitivity scenario, not a calibrated credit model. Deterministic
hazard rates produce default probability increments. The discretized adjustments are

\[
CVA=(1-R)\sum_i DF_i EPE_i(S^c_{i-1}-S^c_i),
\]
\[
DVA=(1-R)\sum_i DF_i ENE_i(S^o_{i-1}-S^o_i),
\qquad
FVA=\sum_i DF_i s_f EPE_i\Delta t_i.
\]

The simplified SIMM proxy combines weighted absolute DV01 and vega in Euclidean norm. MVA funds
that initial margin over a horizon using an average discount. These formulas are labeled
“simplified” throughout because regulatory SIMM requires official risk buckets, correlations,
concentration thresholds and current parameter files.

## 14. Software quality

The build enables `-Wall -Wextra -Wpedantic`; CI turns warnings into errors. ASan and UBSan exercise
the full tests on Linux. Catch2 is version-pinned by CMake when unavailable locally. Eigen is
detected as an integration dependency without leaking it into the ABI. Targets export under the
`qf::` namespace and a generated version file supports `find_package`.

Tests are divided by component and mix exact unit values, algebraic properties, convergence,
statistical comparison and frozen non-regression values. Exceptions carry actionable messages and
invalid inputs never silently return NaN.

## 15. Performance and reproducibility

The deterministic European engine performs 512 payoff evaluations per price and has constant
memory use. Calibration cost is number of quotes times objective evaluations times 512. Monte-Carlo
and LSM scale linearly in paths; LSM stores factors and discounts at exercise dates, giving memory
of order paths times dates.

`benchmarks/pricing_benchmark.cpp` reports elapsed seconds, throughput and a checksum. Benchmark
results should always record compiler, optimization mode, CPU and operating system. The example and
all stochastic APIs use explicit seeds. No notebooks, hidden state or downloaded market data are
required to reproduce the release.

## 16. Model-risk conclusions

The library's strongest features are auditability, exact initial-curve fit, independent European
engines and explicit numerical uncertainty. Its main risks are simplified schedules, sparse-grid
parameter identifiability, normal-equation regression, time-discretized discounting and proxy XVA
assumptions. These are acceptable for the stated educational and recruitment-demo scope and are
clearly isolated behind interfaces that can accept stronger implementations.

Before production use, add dated calendars and day counts, bootstrapped multi-curves, collateral,
robust optimization with analytic derivatives, QR/SVD regression, low-discrepancy simulation,
portfolio parallelism, market-data governance and independent golden datasets owned outside this
repository.

