# Architecture

The main flow is unidirectional:

```text
raw data
  -> schema and point-in-time availability validation
  -> historical universe and returns
  -> feature
  -> transformation / signal / neutralization
  -> target weights and constraints
  -> lagged execution / costs / impact
  -> reconciled P&L
  -> risk / validation / attribution / capacity
  -> exports and report
```

`data` defines input contracts. `features` calculates quantities observable at the decision time.
`signals` normalizes features and produces scores. `portfolio` converts scores into weights and
enforces limits. `backtest` owns execution chronology and accounting. `validation`, `statistics`,
`risk`, and `capacity` analyze a result without changing the simulation. `reporting` produces
stable artifacts.

The vectorized engine is the current reference implementation. Its grain is one row per date and
asset. Desired positions are forward-filled within each asset and then shifted by the declared
execution lag. Costs are assigned to each weight change before daily aggregation.

Immutable configuration classes form the experiment contract. Their complete serialization is
hashed together with the data and runtime versions. The optional C++ bridge does not introduce a
mandatory build dependency.

