# qf-rates

[![CI](https://github.com/OWNER/qf-rates/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/qf-rates/actions/workflows/ci.yml)

`qf-rates` is a dependency-light C++20 library for interest-rate pricing and risk. Release
`v0.1.0` covers deterministic yield curves, vanilla swaps, Black-76 and Bachelier options, the
two-factor Gaussian G2++ model, deterministic Gaussian-quadrature and Monte-Carlo European
swaptions, G2++ calibration, Bermudan Longstaff–Schwartz, DV01/vega scenarios and a compact XVA
extension.

The project is educational and suitable for model prototyping and technical demonstrations. It is
not a production trading or regulatory capital system.

## Build

Requirements: a C++20 compiler, CMake 3.20+, Git and network access on the first build so CMake can
fetch pinned Eigen 3.4.0 and Catch2 3.7.1 when they are not installed. The public ABI intentionally
remains dependency-free.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQF_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/qf_rates_demo
```

Sanitizers:

```bash
cmake -S . -B build-san -DQF_ENABLE_SANITIZERS=ON -DQF_WARNINGS_AS_ERRORS=ON
cmake --build build-san --parallel
ctest --test-dir build-san --output-on-failure
```

Installation and downstream use:

```bash
cmake --install build --prefix "$PWD/install"
```

```cmake
find_package(qf_rates 0.1 REQUIRED)
target_link_libraries(my_pricer PRIVATE qf::qf_rates)
```

Optional Python module:

```bash
python3 -m pip install pybind11
cmake -S . -B build-py -DQF_BUILD_TESTS=OFF -DQF_BUILD_PYTHON=ON
cmake --build build-py --parallel
PYTHONPATH=build-py python3 -c \
  "import qf_rates_python as q; print(q.black76(q.OptionType.Call,100,100,.2,1,.95).price)"
```

## Compact API

The intended workflow is instrument + market + model + engine:

```cpp
auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
auto swap = qf::make_vanilla_swap(2.0, 7.0, 0.03, 1'000'000.0);
double strike = swap.par_rate(*curve, *curve);
qf::G2ppModel model(curve, {.a=.10, .b=.30, .sigma=.01, .eta=.015, .rho=-.70});
qf::EuropeanSwaption option{
    2.0, qf::Schedule(2.0, 7.0, 1.0), strike, 1'000'000.0};
double price = qf::g2pp_european_swaption(model, option);
auto validation = qf::g2pp_european_swaption_mc(
    model, option, {.paths=50'000, .time_steps=120, .seed=42});
```

See [`examples/end_to_end.cpp`](examples/end_to_end.cpp) for curve → calibration → European →
Monte-Carlo → Bermudan → risk → XVA.

## Scope and conventions

- Times are year fractions from valuation time; continuous compounding is used for zero rates.
- Discount factors are strictly positive. Flat extrapolation uses the last continuously compounded
  zero rate.
- Swap fixed and floating legs have explicit pay/receive directions. Floating coupons use simple
  forwards implied by the forward curve.
- Black-76 volatility is lognormal; Bachelier and calibration quotes use absolute normal volatility.
- A swaption call is payer and a put is receiver. Prices are present values in the notional currency.
- DV01 is the value change for a one-basis-point downward zero-rate shift.
- G2++ uses risk-neutral OU factors and a deterministic shift that exactly fits the initial curve.
- Seeds are deterministic. Monte-Carlo results always report standard error and a 95% interval.

## Repository map

```text
include/qf/       public immutable interfaces
src/              implementations
tests/            unit, property, convergence and regression tests
examples/         end-to-end executable
benchmarks/       opt-in micro-benchmark
python/           opt-in pybind11 bindings
docs/             architecture, validation and technical note
scripts/          reproducible local commands and Python cross-check
```

## Quality and limitations

The CI matrix builds on macOS and Linux with warnings-as-errors, runs tests, checks formatting, and
runs ASan/UBSan on Linux. The G2++ European engine uses three-dimensional order-8 Gauss-Hermite
quadrature over the jointly Gaussian discount integral and terminal factors. The Monte-Carlo engine
is an independent discretized validation implementation. LSM uses six polynomial basis functions
with a European lower-bound safeguard.

The XVA/SIMM module is deliberately simplified: deterministic hazard rates, unilateral
discretization, no collateral, proxy WWR scaling, and illustrative SIMM weights. Consult
[`docs/technical-note.md`](docs/technical-note.md) for equations, validation tolerances and model
risk. The reproducible performance baseline is in
[`docs/performance.md`](docs/performance.md).

## Release

The source tree is prepared for `v0.1.0`. Replace `OWNER` in the badge after publishing, then follow
[`docs/release-checklist.md`](docs/release-checklist.md). Changes are recorded in
[`CHANGELOG.md`](CHANGELOG.md).
