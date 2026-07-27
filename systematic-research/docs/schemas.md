# Schemas

## Market data

| Column | Type | Rule |
|---|---|---|
| `date` | UTC timestamp | economic date |
| `asset` | string | non-empty and unique with `date` |
| `price` | float | finite and strictly positive |
| `volume` | float | non-negative |
| `available_at` | UTC timestamp | greater than or equal to `date` |

Optional columns: `return`, `sector`, `adv`, `volatility`, `carry`, and `value`.

## Targets

`date`, `asset`, `target_weight`, and `available_at`. Weights are fractions of capital. A missing
row retains the latest target for the same asset, while execution remains subject to the lag.

## Daily results

`gross_return`, `cash_return`, `linear_cost`, `impact_cost`, `net_return`, `turnover`,
`gross_exposure`, `net_exposure`, `cash_weight`, and `max_participation`.

## Positions

Market inputs and targets are enriched with `desired_weight`, `executed_weight`, `weight_change`,
`gross_contribution`, cost contributions, and `participation`.

CSV exports retain these names. JSON mappings are sorted and serialized with stable indentation.
Future versions will add an explicit schema-version identifier.

