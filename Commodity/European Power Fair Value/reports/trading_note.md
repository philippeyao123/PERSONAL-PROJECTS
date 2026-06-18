# Prompt-curve view -- 2026-06-11

**Model fair value (next-day baseload):** 104.22 EUR/MWh
**Prompt proxy (trailing 7d DA baseload):** 90.92 EUR/MWh
**Gap:** +13.30 EUR/MWh (z = +0.68)
**View:** NEUTRAL

## How the view is used
- |z| > 0.75: fair value diverges materially from where the prompt is
  marked -> position front-week in the direction of the model (DA expected
  to print above/below the curve, pulling prompt settlements with it).
- The view sizes with |z| and is re-evaluated every morning after the new
  D-1 fundamentals run.

## Invalidation triggers (flatten / re-run)
1. TSO intraday revision of D+1 wind+solar forecast > 5 GW vs model inputs.
2. REMIT unplanned outage > 2 GW in DE or a connected market.
3. Yesterday's model error > 2x trailing 30d error std (regime break).
4. Fuel/carbon shock (TTF or EUA move > 5%) -- the DA model carries no fuel
   features, so curve moves driven by fuels are out of model scope.

## Out-of-sample evaluation (144 days)
- Active views: 63 (29 long / 34 short)
- Hit rate: 76.2%
- Avg captured spread: +17.23 EUR/MWh per active day
