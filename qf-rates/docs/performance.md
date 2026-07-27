# Performance baseline

The opt-in benchmark prices the same 2Y×10Y G2++ European swaption 1,000 times with the order-8
three-dimensional Gaussian quadrature engine. It prints a checksum so an optimizer cannot remove
the work.

Baseline recorded on 2026-07-27:

| Build | Platform | Compiler | Throughput |
|---|---|---|---:|
| Release (`-O2`/CMake Release) | Apple arm64, macOS | AppleClang 21.0.0 | 1,297 prices/s |

Observed wall time was 0.771 seconds for 1,000 prices. This is a development-machine baseline, not a
cross-platform performance guarantee. Reproduce it with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQF_BUILD_BENCHMARKS=ON
cmake --build build --target qf_rates_bench --parallel
./build/qf_rates_bench
```

Calibration scales approximately with quotes × objective evaluations × 512 payoff nodes.
Monte-Carlo scales linearly with paths × time steps. LSM stores two factor states and one discount
factor per path and exercise date, so its dominant memory complexity is linear in paths × exercise
dates.
