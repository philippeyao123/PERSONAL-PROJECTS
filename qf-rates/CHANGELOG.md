# Changelog

All notable changes follow Keep a Changelog. Version numbers follow semantic versioning.

## [0.1.0] - 2026-07-27

### Added

- C++20 `qf_core` and `qf_rates` libraries with CMake install/export support.
- Flat and linearly/log-linearly interpolated yield curves.
- Fixed/floating swaps, NPV, annuity and par rate.
- Black-76 and Bachelier prices and Greeks.
- Bisection, adaptive Simpson integration, bounded optimization and online statistics.
- Deterministic Mersenne Twister sampling, antithetic paths and control-variate infrastructure.
- G2++ curve fitting, bonds, moments, factor evolution and European swaption quadrature.
- Independent G2++ Monte-Carlo validation and confidence intervals.
- Normal-volatility calibration with expiry/tenor diagnostics.
- Bermudan swaption Longstaff–Schwartz engine and convergence controls.
- Linear and quadratic LSM bases with a documented multi-seed convergence grid.
- Parallel/bucketed DV01, bucketed vega, curve scenarios and explicit volatility shocks.
- Exposure simulation, netting, proxy WWR, CVA/DVA/FVA and simplified SIMM/MVA.
- Quantified antithetic/control-variate reduction and deterministic WWR regression tests.
- QuantLib validation for swaps, G2++ swaptions and calibration.
- Catch2 test suite, active monorepo and standalone CI, clang-tidy, sanitizers and benchmarks.
- Python bindings for curves, swaps, G2++, Monte Carlo, LSM, calibration and market risk.
