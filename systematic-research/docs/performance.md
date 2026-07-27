# Performance and complexity

For `N` rows and `A` assets, the initial sort costs `O(N log N)` and grouped transformations are
typically `O(N)`. Backtest memory usage is `O(N)` because detailed contributions are retained for
auditability. Neutralization regressions are independent by date, and their cost depends on the
cube of the typically small number of exposures.

Run the reproducible benchmark:

```bash
python scripts/benchmark.py
```

It builds a synthetic panel, runs the pipeline, and prints rows per second, elapsed time, and the
number of simulated days. Results depend on hardware and the pandas version. CI verifies
correctness rather than a fragile speed threshold.

For larger datasets, store inputs in partitioned Parquet files, select only necessary columns
before merges, and split large reports. Do not parallelize before confirming that row ordering
and numerical reductions remain deterministic.

