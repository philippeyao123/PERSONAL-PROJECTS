# Architecture

Dependencies point inward:

```text
examples / bindings / tests
            |
          qf_rates
   curves instruments models engines risk xva
            |
          qf_core
      types numerics random errors
```

`qf_core` owns general numerical and domain primitives. `qf_rates` owns rate-specific market data,
instruments, models, engines and risk. Public headers contain stable interfaces; implementations
live under `src/`. Yield curves are immutable polymorphic market objects shared by models through
`std::shared_ptr<const YieldCurve>`.

Pricing is deliberately decomposed:

- instrument: contractual schedule, strike, direction and notional;
- market: discount/forward curve and volatility quote;
- model: G2++ parameters and state dynamics;
- engine: analytic formula, Gaussian quadrature, Monte-Carlo or LSM;
- risk: bump definition plus deterministic revaluation.

This separation permits the Monte-Carlo and quadrature engines to validate each other without
sharing payoff integration logic.

