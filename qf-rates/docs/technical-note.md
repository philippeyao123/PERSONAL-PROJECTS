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

## 17. Architecture and ownership boundaries

The source tree is intentionally divided into a small numerical core and a rates domain library.
`qf_core` owns algorithms that do not know what a curve or swaption is: bisection, adaptive
integration, online statistics, regression, bounded optimization and random-number utilities.
`qf_rates` owns the financial meaning: curves, schedules, instruments, models, engines and risk
reports. This direction of dependency is one-way. A numerical routine can be tested against a
quadratic or a sine integral without constructing a market object, while a pricing engine can
replace a numerical implementation without changing the instrument representation.

Public headers use standard-library types and small value structures. Eigen is detected and linked
privately, so its types do not cross the installed ABI. Catch2 is a test dependency only, pybind11
is compiled only when bindings are requested, and QuantLib is never linked into the C++ library.
The latter is a validation dependency in Python CI, not a pricing dependency. This separation
matters for independent verification: a defect cannot be hidden by qf-rates delegating the
calculation under test to QuantLib.

The principal domain interfaces are deliberately narrow. A `YieldCurve` answers discount, zero and
forward queries. An `InterestRateSwap` owns its legs and accepts discount and forward curves at
valuation. `G2ppModel` owns the initial curve and its five parameters. European and Bermudan
swaptions are plain value objects consumed by separate engines. Risk functions accept the same
instrument and market objects used by pricing. There is no global evaluation date, singleton
market, implicit currency or thread-local configuration.

This design makes ownership and lifetime visible. Curves shared by model instances use
`std::shared_ptr<const interface>` semantics through the public alias; temporary pricing state
remains on the stack. Stochastic routines allocate path-local arrays within a call and return
aggregate results rather than exposing mutable internal buffers. Seed, path count, time steps and
basis choice live in configuration structures. A caller can therefore reproduce a result from
serialized inputs without recovering hidden process state.

The library is maintained in two publication forms. Its development location is the
`qf-rates/` subtree of the `PERSONAL-PROJECTS` monorepo, where the root workflow is the active
integration gate. A subtree split publishes the same commit history as an autonomous
`philippeyao123/qf-rates` repository. In that repository the nested workflow becomes the root
workflow. The split is not a copied or divergent implementation: its commit is produced from the
monorepo prefix, tagged independently and released with GitHub-generated source archives.

## 18. Calibration governance and identifiability

A calibration result is not validated merely because an optimizer stops. The release separates
four questions: whether inputs are valid, whether the objective was minimized inside permitted
bounds, whether fitted prices reproduce quoted volatilities, and whether the resulting parameters
are economically stable. Only the first three can be established on the small demonstration
surface. Economic stability would require more dates, multiple market snapshots and explicit
regularization.

Every quote supplies expiry, tenor, strike, absolute normal volatility and positive weight. Invalid
times, negative strikes or volatilities, and non-positive weights are rejected before optimization.
Candidate parameters are bounded to keep mean reversions positive, factor volatilities non-negative
and correlation away from singular endpoints. For each objective evaluation, the model price is
converted back to an implied Bachelier volatility by bracketed bisection. Optimizing in quoted
volatility space prevents a high-annuity or long-expiry instrument from dominating solely because
its currency price is larger.

The objective is weighted root-mean-square volatility error:

\[
\operatorname{RMSE}(\theta)=
\sqrt{\frac{\sum_i w_i(\sigma_i^{model}(\theta)-\sigma_i^{market})^2}
{\sum_i w_i}}.
\]

Diagnostics retain each expiry, tenor, market volatility, model volatility and error in basis
points. That table is more informative than the scalar objective because localized bias can cancel
in an average. Bounds and initial parameters are also part of the model-risk record. A future
optimizer should not change them silently: doing so changes the admissible model family even if the
instrument formulas are identical.

The external QuantLib exercise illustrates identifiability risk. On the same six normal-volatility
quotes, qf-rates reaches RMSE 0.0001868800 while QuantLib reaches 0.0003867077. Both pass the
0.001 acceptance threshold, yet the parameter vectors differ substantially. qf-rates returns
\((0.01374512,0.27060742,0.01000000,0.01336102,-0.78869141)\); QuantLib returns approximately
\((0.00013715,0.00946185,0.00013779,0.00946185,-0.76345702)\). This does not establish that either
parameter vector is “true.” It establishes that multiple factor dynamics can fit a sparse surface.

Consequently, downstream risk should be tested across plausible calibrations rather than relying on
a single optimum. Sensible production extensions include multiple starting points, penalties on
extreme mean reversions, temporal regularization against the previous calibration, withheld-quote
validation and a parameter-stability dashboard. Price fit remains the primary acceptance measure;
parameter narratives are secondary and require stronger evidence.

## 19. Independent implementation validation

Validation uses a hierarchy. Exact identities are preferred where available. Independent closed
forms are next, then independent library implementations, then stochastic confidence intervals,
and finally frozen regression values. A frozen number alone is the weakest evidence because code
and expected value can preserve the same error. The test suite therefore combines several levels
for material components.

For curves, the strongest checks are analytic: flat discounts equal \(\exp(-rT)\), supplied
interpolation nodes are reproduced exactly, and G2++ bonds at time zero equal the initial discount
curve. For vanilla swaps, a par-coupon swap has zero NPV and the single-curve floating leg
telescopes. Black-76 and Bachelier are checked against direct formulas and QuantLib. These cases
detect units, sign conventions, discount placement and limiting behavior.

The QuantLib swap comparison deliberately aligns conventions rather than accepting an unexplained
difference. Both sides use a continuously compounded flat 2.5% curve, a 2Y start, 7Y maturity,
annual fixed payments, semiannual floating payments and unadjusted exact-year periods. Under those
conditions qf-rates and QuantLib produce par rate 0.025315120524 and NPV
-20,684.88039279 for a 3% payer swap with notional one million. The par-rate tolerance is
\(10^{-12}\), and the currency NPV tolerance is \(10^{-7}\).

For the ATM 2Y×5Y G2++ payer swaption, qf-rates order-8 joint Gaussian quadrature returns
13,527.73806635 and QuantLib's G2 engine returns 13,430.98886655. The 0.720343% difference is within
the declared 1% tolerance. The difference is retained rather than tuned away because the libraries
use different schedule machinery and numerical algorithms. Within qf-rates, the deterministic
price is also compared with a separately time-stepped Monte Carlo engine using its reported
standard error. External agreement and internal algorithmic independence answer different
questions, so both are required.

The cross-check is an executable assertion suite, not a manually transcribed table.
`scripts/python_reference.py --require-quantlib --require-bindings` fails if either module is
missing or any tolerance is breached. CI builds the binding from the same source revision before
running the script. The report in `docs/quantlib-validation.md` records the environment, inputs,
values and interpretation so that a future change has an explicit review baseline.

## 20. Formal variance-reduction evidence

Antithetic sampling and a control variate are optional mechanisms, not labels attached to an
engine. Their benefit must be measured on a common workload using comparable path budgets and a
fixed seed. The validation executable prices the same expectation three ways and reports standard
error plus the ratio of estimator variance to the plain estimator variance.

| Estimator | Standard error | Variance ratio |
|---|---:|---:|
| Plain | 0.04170130 | 1.00000000 |
| Antithetic | 0.03328838 | 0.63721512 |
| Control variate | 0.01898809 | 0.20733054 |

Because variance is the square of standard error for a fixed effective sample count, the
antithetic workload reduces measured variance by 36.28% and the control variate by 79.27%. The
control coefficient is estimated from the sample covariance rather than chosen after inspecting
the final price. Unit tests assert a material reduction with deliberately conservative thresholds,
while the benchmark prints exact observed ratios for review.

The numbers should not be generalized to every payoff. Antithetic sampling is most effective when
the payoff is sufficiently monotone and opposing shocks cancel odd components. A poorly correlated
control can add estimation noise, especially with few paths. The generic engine therefore returns
standard error whether or not variance reduction is enabled. A consumer should treat lower
reported error at the same workload as evidence, not assume that a named technique guarantees
improvement.

Formal comparison also prevents an accounting mistake: two antithetic draws must not be presented
as two independent samples if they are aggregated as a correlated pair. The validation workload
uses consistent estimator definitions and calculates ratios from the returned uncertainty. In a
production performance study, the next measure would be variance times wall-clock cost, since a
more expensive estimator can reduce variance but still be less efficient per unit of compute.

## 21. LSM convergence and exercise policy

Longstaff–Schwartz contains three interacting error sources: Monte Carlo sampling, regression
approximation and exercise-time discretization. Increasing paths addresses only the first.
Changing the polynomial basis probes the second, while changing exercise dates probes the third.
The release therefore exposes a `LsmBasis` choice and records a grid over path counts, seeds and
bases instead of reporting one favorable run.

| Paths | Seed | Basis | Price | Standard error |
|---:|---:|---|---:|---:|
| 2,000 | 7 | Linear | 0.01373992 | 0.00035943 |
| 2,000 | 7 | Quadratic | 0.01374178 | 0.00035948 |
| 2,000 | 42 | Linear | 0.01423549 | 0.00036417 |
| 2,000 | 42 | Quadratic | 0.01415695 | 0.00036237 |
| 5,000 | 7 | Linear | 0.01346868 | 0.00022505 |
| 5,000 | 7 | Quadratic | 0.01347971 | 0.00022489 |
| 5,000 | 42 | Linear | 0.01385239 | 0.00022829 |
| 5,000 | 42 | Quadratic | 0.01388149 | 0.00022813 |
| 10,000 | 7 | Linear | 0.01366320 | 0.00015946 |
| 10,000 | 7 | Quadratic | 0.01367462 | 0.00015934 |
| 10,000 | 42 | Linear | 0.01383001 | 0.00015982 |
| 10,000 | 42 | Quadratic | 0.01384655 | 0.00015895 |

The approximately \(1/\sqrt{N}\) reduction is visible: increasing paths from 2,000 to 10,000
reduces standard error from about 0.00036 to 0.00016. At 10,000 paths, changing seed shifts the
price by roughly one reported standard error, while linear and quadratic bases are much closer for
each seed. This is evidence of reasonable stability for the demonstration, not proof that the
chosen basis is adequate for every trade.

Regression is performed only on in-the-money paths. The linear basis is
\((1,x,y)\); the quadratic basis adds \((x^2,xy,y^2)\). A small ridge protects the normal equations,
and a minimum in-the-money sample count prevents an underdetermined regression. When the exercise
set contains one date, the implementation delegates to the validated European engine, producing an
exact limiting invariant. Multiple exercise dates are checked for reproducibility and the expected
Bermudan-versus-European ordering.

Exercise-grid tests compare alternative ordered date sets. Adding a possible exercise date should
not economically reduce value, although finite-sample LSM estimates can violate this locally. The
release uses a disclosed European floor to protect the essential lower bound. Production work
should add out-of-sample regression, QR or SVD, richer normalized state variables and confidence
intervals for policy differences. A dual upper bound would materially strengthen validation by
bracketing the true Bermudan value.

## 22. Market scenarios and wrong-way risk

Risk reports distinguish rate bumps from volatility shocks. DV01 bumps continuously compounded
zero rates and revalues both discount and forwarding when using the single-curve helper. The
volatility scenario function instead applies an absolute normal-volatility shock to a Bachelier
option input and returns scenario name, shocked volatility, price and change from base. For an
otherwise fixed payer option, the up shock must raise value and the down shock must lower it. Tests
assert those signs and reject shocks that would create a negative volatility.

An absolute shock is used because normal volatility is expressed in rate units. A shock of 0.0010
means ten basis points of annualized normal volatility, not a ten-percent relative change. This
choice is explicit in the API and documentation. Relative, smile-shaped or expiry-dependent shocks
can be layered later, but should be separate scenario definitions to avoid unit ambiguity.

Wrong-way risk is tested in a controlled deterministic comparison. With identical paths and market
inputs, beta zero is the independent-exposure baseline and positive beta multiplies positive
exposure by \(\exp[\beta(x+y)]\). The validation workload reports CVA 1312.45468831 for the baseline
and 1636.07844166 for beta 20. Tests verify that the stressed exposure and aggregated CVA rise in
the designed scenario. They also retain the existing netting property: aggregating trades before
positive/negative decomposition cannot increase pathwise positive exposure relative to the sum of
standalone positive exposures.

The proxy is intentionally not presented as a full joint credit/rates model. It has no stochastic
hazard process, copula calibration, collateral thresholds, margin period of risk or path-dependent
default. Its purpose is sensitivity analysis: identify whether exposure concentration in adverse
rate states could materially change CVA. The parameter beta has no standalone regulatory meaning.
Production WWR would require joint calibration, default-time simulation or an equivalent
measure-change framework, and validation on counterparty-specific historical or market evidence.

## 23. Python boundary and research workflow

The first binding exposed only Black-76 and Bachelier, which was insufficient for independent
research. Version 0.1.0 exposes the coherent workflow: flat and interpolated curves, schedules,
swaps, G2++ parameters and bonds, European quadrature and Monte Carlo, Bermudan LSM, six-quote
calibration, DV01 and volatility scenarios. Result objects retain diagnostics such as standard
error, confidence interval, exercise probabilities and calibration rows instead of returning a
bare scalar.

Python owns high-level orchestration while C++ owns calculation. Shared curve lifetimes are managed
through pybind11 smart-pointer holders. Enumerations preserve option direction, pay/receive,
interpolation and basis choices without magic integers. Configuration structures are mutable in
Python for convenient research setup, but pricing models and C++ algorithms still validate their
inputs.

`scripts/python_bindings_smoke.py` exercises every major family in one short run. It checks finite
swap value, positive par rate, deterministic and Monte Carlo European results, Bermudan value,
calibration diagnostics, DV01 and volatility shock signs. This is both an import test and an API
coherence test. The QuantLib reference script then uses those same bindings for external
comparison, closing the gap between the C++ unit suite and a Python research workflow.

The boundary is deliberately limited. Exposure/XVA path objects, custom payoff callbacks and the
full numerical utility layer are not yet bound. There are no pandas adapters, wheels or stable
pickle formats. Those omissions keep the initial interface reviewable. A future packaging release
should add `pyproject.toml`, manylinux and macOS wheels, Python type stubs and explicit semantic
compatibility guarantees.

## 24. Continuous integration and release reproducibility

Two workflows validate the same subtree in its two repository contexts. At monorepo root,
`.github/workflows/qf-rates-ci.yml` uses path filters and sets `qf-rates` as the working directory.
Inside the autonomous split, `.github/workflows/ci.yml` runs from repository root. Both execute the
same five gates: Linux/macOS Release builds and tests, Linux ASan/UBSan, clang-format,
clang-tidy compilation, and Python/QuantLib validation.

Warnings are errors in normal and sanitizer builds. The installation step verifies generated CMake
package files and public headers. Formatting is checked over public headers, implementations,
tests, examples, benchmarks and bindings. clang-tidy is not merely configured in a file: CMake
invokes it while compiling the library in a dedicated job. The Python job installs explicit
validation dependencies, compiles the extension and runs both smoke and external reference scripts.

Reproducibility has platform limits that are recorded rather than concealed. On macOS, the
available AddressSanitizer reports that `detect_leaks=1` is unsupported, so local ASan/UBSan
validation uses leak detection disabled. The Linux CI job enables leak detection. Random seeds are
fixed, but normal-distribution mappings may differ between standard libraries; stochastic tests
therefore use statistical tolerances. The benchmark records compiler, platform, build type and
checksum.

The release tag belongs to the autonomous subtree history, not to the entire monorepo. This avoids
claiming that unrelated projects share the qf-rates semantic version. GitHub generates immutable
source `.zip` and `.tar.gz` links from the public tag. A clean consumer can clone only qf-rates,
configure CMake, test, install and use `find_package(qf_rates 0.1)` without the parent monorepo.

## 25. Limitations and production roadmap

The largest functional limitation is time representation. Year fractions are convenient for model
research but cannot encode holiday calendars, modified-following adjustment, stubs, end-of-month
rules or heterogeneous day counts. A production instrument layer must introduce actual dates and a
market convention service while retaining year fractions only at the model boundary.

Curve construction is another deliberate omission. The library consumes discount factors or a flat
rate; it does not bootstrap OIS discounting and tenor-specific forwarding curves from deposits,
futures, swaps and basis instruments. It has no market-data timestamps, stale-quote policy or
arbitrage-repair procedure. These are not auxiliary details: they often dominate real desk P&L and
risk differences.

Numerically, coordinate search, normal equations and pseudo-random Monte Carlo favor transparency
over scale. Production extensions should use analytic or automatic derivatives, robust
least-squares optimization, QR/SVD, Sobol sequences with scrambling, deterministic parallel
reduction and portfolio batching. Any replacement must preserve the independent baselines and add
performance evidence rather than weakening acceptance tolerances.

Counterparty risk remains illustrative. Collateral, initial margin schedules, close-out conventions,
funding asymmetry, stochastic credit, recovery dependence and regulatory SIMM parameter sets are
out of scope. The simplified calculations are useful because their assumptions are explicit and
testable, but they should not be reported as accounting CVA or regulatory capital.

Finally, model governance is a process rather than a code feature. A production adoption would need
model inventory, independent ownership of validation datasets, change approval, monitoring
thresholds, backtesting, incident response and decommissioning criteria. Version 0.1.0 supplies a
sound technical base for that conversation: explicit conventions, reproducible engines, quantified
uncertainty, external comparisons, scenario tests, active CI and a traceable public release.
