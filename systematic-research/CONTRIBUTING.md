# Contributing

Create a virtual environment, install `.[dev]`, and work on a short-lived branch. Every change
that affects economic behavior must include:

1. a written convention for decision, availability, and execution timestamps;
2. a test that fails when future information leaks into the result;
3. a P&L reconciliation check when backtest accounting changes;
4. a changelog update when the public API changes.

Run `scripts/run_all.sh` before proposing a change. Do not commit proprietary data, large reports,
or secrets. Breaking changes are reserved for major versions. New parameters must have explicit
defaults.

