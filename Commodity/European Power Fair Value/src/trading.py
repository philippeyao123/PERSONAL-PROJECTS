"""Translate the DA fair-value forecast into a prompt-curve view.

Logic
-----
1. Fair value for delivery day D = mean of the 24 forecast hourly prices
   ("forecast baseload", EUR/MWh), produced at D-1 before the auction.
2. Prompt proxy: the trailing PROMPT_WINDOW-day realised DA baseload average.
   Front-week / front-month quotes (EEX) are not freely redistributable, and
   prompt power forwards settle close to expected DA outturn, so a trailing
   DA average is a transparent, reproducible stand-in for where the prompt
   is marked. With a market-data licence the same code plugs EEX front-week
   settlements directly into `prompt_proxy`.
3. Signal: z-score of (fair value - prompt proxy) over a 60-day rolling
   window. |z| > SIGNAL_Z  -> long (cheap prompt) or short (rich prompt)
   front-week; otherwise neutral.
4. Evaluation: each daily view is scored against the *next day's realised*
   DA baseload minus the prompt proxy -- i.e. did the curve converge toward
   our fair value? We report hit rate and average captured spread.

Invalidation (documented in the trading note):
  - wind/solar D-1 forecast revision > 5 GW vs the run used by the model
  - unplanned outage announcements (REMIT) > 2 GW
  - model residual on the last realised day > 2x its trailing std
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import DATA, PROMPT_WINDOW, REPORTS, SIGNAL_Z


def build_views() -> pd.DataFrame:
    res = pd.read_csv(DATA / "predictions_oos.csv", index_col=0)
    res.index = pd.to_datetime(res.index, utc=True)
    local = res.index.tz_convert("Europe/Berlin")

    daily = res.groupby(local.normalize()).agg(
        fair_value=("lgbm", "mean"),
        realised=("y_true", "mean"),
    )
    daily["prompt_proxy"] = daily["realised"].shift(1).rolling(PROMPT_WINDOW).mean()
    daily["gap"] = daily["fair_value"] - daily["prompt_proxy"]
    roll = daily["gap"].rolling(60, min_periods=30)
    daily["gap_z"] = (daily["gap"] - roll.mean()) / roll.std()

    daily["view"] = np.select(
        [daily["gap_z"] > SIGNAL_Z, daily["gap_z"] < -SIGNAL_Z],
        ["LONG prompt", "SHORT prompt"], default="NEUTRAL")

    # score: realised baseload over the *delivered front week* (D..D+6) vs the
    # prompt proxy, signed by the view. This is deliberately harder than
    # scoring day D alone: the model only forecasts D, so a correct view
    # requires the DA-vs-curve divergence to persist across the prompt window.
    fwd_week = daily["realised"][::-1].rolling(7, min_periods=7).mean()[::-1]
    fwd_spread = fwd_week - daily["prompt_proxy"]
    sign = daily["view"].map({"LONG prompt": 1, "SHORT prompt": -1, "NEUTRAL": 0})
    daily["captured"] = sign * fwd_spread
    return daily.dropna(subset=["gap_z"])


def main() -> None:
    daily = build_views()
    active = daily[daily["view"] != "NEUTRAL"]
    stats = {
        "days_evaluated": int(len(daily)),
        "active_days": int(len(active)),
        "hit_rate": round(float((active["captured"] > 0).mean()), 3),
        "avg_captured_eur_mwh": round(float(active["captured"].mean()), 2),
        "median_captured_eur_mwh": round(float(active["captured"].median()), 2),
        "long_days": int((active["view"] == "LONG prompt").sum()),
        "short_days": int((active["view"] == "SHORT prompt").sum()),
    }
    daily.to_csv(DATA / "daily_views.csv")
    (REPORTS / "trading_stats.json").write_text(json.dumps(stats, indent=2))

    last = daily.iloc[-1]
    note = f"""# Prompt-curve view -- {daily.index[-1].date()}

**Model fair value (next-day baseload):** {last['fair_value']:.2f} EUR/MWh
**Prompt proxy (trailing {PROMPT_WINDOW}d DA baseload):** {last['prompt_proxy']:.2f} EUR/MWh
**Gap:** {last['gap']:+.2f} EUR/MWh (z = {last['gap_z']:+.2f})
**View:** {last['view']}

## How the view is used
- |z| > {SIGNAL_Z}: fair value diverges materially from where the prompt is
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

## Out-of-sample evaluation ({stats['days_evaluated']} days)
- Active views: {stats['active_days']} ({stats['long_days']} long / {stats['short_days']} short)
- Hit rate: {stats['hit_rate']:.1%}
- Avg captured spread: {stats['avg_captured_eur_mwh']:+.2f} EUR/MWh per active day
"""
    (REPORTS / "trading_note.md").write_text(note)
    print(json.dumps(stats, indent=2))
    print(f"\nLatest view: {last['view']} (z={last['gap_z']:+.2f})")


if __name__ == "__main__":
    main()
