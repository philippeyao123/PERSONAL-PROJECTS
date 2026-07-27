# Technical note

## Tested hypothesis

The flagship experiment tests whether lagged cross-sectional momentum produces an exploitable
signal after constraints, costs, and market impact. The generator includes multiple assets, a
benchmark, and a delisting to exercise point-in-time paths.

## Conventions

- daily frequency and 252-period annualization;
- signal available on the decision date and execution in the following period;
- long/short weights with gross, net, and concentration limits;
- commission, half-spread, and slippage expressed in basis points;
- impact proportional to volatility and the square root of participation;
- zero cash return in the example.

## What the result does not prove

The data is synthetic and does not include complete microstructure or real-world data-quality
issues. Square-root impact is an approximation. The vectorized engine does not simulate order
books, partial fills, or rejections. Linear neutralization does not replace a calibrated risk
model. Historical VaR and CVaR assume that the sample is relevant.

Before real-world use, connect timestamped historical data, qualify calendars, calibrate costs by
market, record every attempted experiment, and independently review every temporal rule.

