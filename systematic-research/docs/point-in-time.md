# Point-in-time contract

## Clocks

- `date`: economic observation or decision date;
- `available_at`: earliest timestamp at which the information can be used;
- execution: `date + execution_lag` in the simulation calendar.

Data is admissible when `available_at <= date` at the decision time. The engine rejects targets
that violate this rule. Rolling statistics apply `shift(1)` before estimation, so the current
value never enters its own normalization.

## Investment universe

The universe is a time-varying table with eligibility start and end dates. A delisted asset
remains in the historical record. Missing observations are not replaced with unchanged prices:
price forward filling is disabled by default because it fabricates zero returns and hides stale
data.

## Returns

Simple and logarithmic returns are calculated by asset. Futures returns are calculated within
each contract so a contract roll cannot become an economic return. The risk-free rate must be
provided at the same frequency.

## Anti-leakage checklist

- document the source and publication timestamp;
- enforce unique `date/asset` keys;
- prohibit retrospective backfilling in features;
- estimate parameters from past data only;
- apply weights after the execution lag;
- purge labels that overlap the test window;
- inspect the final test set once, after selection.

