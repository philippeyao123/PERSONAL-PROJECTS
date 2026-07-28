"""Feature engineering for next-day hourly price forecasting.

Information-set discipline
--------------------------
The DE-LU day-ahead auction for delivery day D clears at 12:00 CET on D-1.
At forecast time (D-1 morning) the following are known:
  - all auction prices up to and including delivery day D-1
    (published at D-2 ~12:45 CET)  -> price lags of 24h, 48h, 168h are valid
  - TSO day-ahead forecasts of wind & solar for day D
  - the calendar
These variables form FEATURES, the primary publication specification.

Open-Meteo's Historical Forecast API stitches early hours from successive
model runs and does not preserve one fixed D-1 lead time for every target
hour.  Weather variables are therefore retained in ALL_FEATURES only for the
explicitly labelled weather-augmented sensitivity.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from config import DATA

DE_HOLIDAYS = holidays.country_holidays("DE")

PRICE_LAGS = (24, 48, 168)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- price history (valid lags only) ---
    for lag in PRICE_LAGS:
        out[f"price_lag{lag}"] = out["price"].shift(lag)
    # daily structure known at auction time: stats over delivery day D-1
    daily = out["price"].shift(24).rolling(24)
    out["price_d1_mean"] = daily.mean()
    out["price_d1_std"] = daily.std()
    out["price_w1_mean"] = out["price"].shift(24).rolling(168).mean()

    # --- ex-ante fundamentals ---
    out["fcst_renewables"] = out["fcst_solar"] + out["fcst_wind_total"]
    # heating/cooling demand proxies from forecast temperature
    out["hdd"] = (16.0 - out["wx_temperature_2m"]).clip(lower=0)
    out["cdd"] = (out["wx_temperature_2m"] - 21.0).clip(lower=0)
    # day-over-day renewables swing (today's DA forecast vs yesterday's level)
    out["renew_chg_24"] = out["fcst_renewables"] - out["fcst_renewables"].shift(24)

    # --- calendar (local time governs behaviour) ---
    local = out.index.tz_convert("Europe/Berlin")
    out["hour"] = local.hour
    out["dow"] = local.dayofweek
    out["month"] = local.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_holiday"] = pd.Series(local.date, index=out.index).map(
        lambda d: int(d in DE_HOLIDAYS))
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out = out.dropna(subset=[f"price_lag{max(PRICE_LAGS)}"])
    return out


PRICE_FEATURES = [
    "price_lag24", "price_lag48", "price_lag168",
    "price_d1_mean", "price_d1_std", "price_w1_mean",
]
RENEWABLE_FEATURES = [
    "fcst_solar", "fcst_wind_total", "fcst_renewables", "renew_chg_24",
]
WEATHER_FEATURES = [
    "wx_temperature_2m", "wx_wind_speed_100m", "wx_shortwave_radiation",
    "hdd", "cdd",
]
CALENDAR_FEATURES = [
    "hour", "dow", "month", "is_weekend", "is_holiday", "hour_sin", "hour_cos",
]
# Primary publication specification: every covariate has a documented D-1
# information set.  Open-Meteo's stitched historical-forecast series is kept
# for the explicitly labelled weather-augmented sensitivity only; it does not
# preserve a fixed day-ahead vintage for every target hour.
FEATURES = PRICE_FEATURES + RENEWABLE_FEATURES + CALENDAR_FEATURES
ALL_FEATURES = FEATURES + WEATHER_FEATURES
TARGET = "price"


def load_features() -> pd.DataFrame:
    df = pd.read_csv(DATA / "dataset.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    feats = build_features(df)
    # forward-fill tiny gaps in exogenous forecasts (never the target)
    feats[ALL_FEATURES] = feats[ALL_FEATURES].ffill(limit=3)
    feats = feats.dropna(subset=ALL_FEATURES + [TARGET])
    return feats


if __name__ == "__main__":
    f = load_features()
    print(f.shape)
    print(f[FEATURES + [TARGET]].describe().T.round(2))
