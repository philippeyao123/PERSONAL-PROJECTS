"""
Assemble the modelling dataset (half-hourly, 2 years).

Two information sets are built:
  A. DAY-AHEAD set  : only data available by D-1 ~16:30 (forecast vintages,
                      weather treated as perfect-foresight forecast, calendar,
                      lagged prices >= 48 settlement periods).
  B. NOWCAST extras : wind forecast error, same-day price lags (2 SPs),
                      for the real-time imbalance stress model.

Target: system (imbalance) price spike, defined against a rolling 90-day
95th percentile computed on PAST data only (shifted), so the label
definition itself is leakage-free and seasonally adaptive.
"""
import pandas as pd
import numpy as np

D = "/home/claude/data"

def load(name):
    df = pd.read_csv(f"{D}/{name}.csv", parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").set_index("time")

sp     = load("system_prices")
dem    = load("demand_actual")
gen    = load("generation")
mid    = load("mid_price")
wfc    = load("wind_fcst")
dfc    = load("demand_fcst")
wx     = load("weather")

# half-hourly master index
df = sp.join(dem).join(gen).join(mid)

# demand forecast is half-hourly; wind fcst & weather are hourly -> ffill to 30min
df = df.join(dfc)
hourly = wfc.join(wx)
hourly = hourly.resample("30min").ffill(limit=1)
df = df.join(hourly)

df = df[~df["system_price"].isna()].copy()

# ----------------------------------------------------------------- weather aggregates
demand_cities = ["london", "birmingham", "manchester", "glasgow"]
df["temp_gb"]  = df[[f"temp_{c}" for c in demand_cities]].mean(axis=1)
df["rh_gb"]    = df[[f"rh_{c}" for c in demand_cities]].mean(axis=1)
df["solar_gb"] = df[[f"solar_{c}" for c in demand_cities]].mean(axis=1)
df["wind_ms_offshore"] = df[["wind_hornsea", "wind_irishsea"]].mean(axis=1)

# Heating / Cooling Degree (base 15.5C / 22C, half-hourly resolution)
df["hdd"] = (15.5 - df["temp_gb"]).clip(lower=0)
df["cdd"] = (df["temp_gb"] - 22.0).clip(lower=0)

# ----------------------------------------------------------------- fundamentals
df["wind_total"] = df["wind_on"] + df["wind_off"]
df["renew_total"] = df["wind_total"] + df["solar"]

# tightness proxies
df["resid_demand_fcst"] = df["nd_fcst_da"] - df["wind_fcst_da"]          # DA vintage
df["renew_pen_fcst"]    = df["wind_fcst_da"] / df["nd_fcst_da"]          # DA vintage
df["resid_demand_act"]  = df["demand_actual"] - df["renew_total"]        # realized (EDA + nowcast)
df["renew_pen_act"]     = df["renew_total"] / df["demand_actual"]
df["margin_proxy"]      = (df["gas"].rolling(48 * 30, min_periods=48).max()
                           + df["nuclear"] + df["biomass"] + df["hydro"]
                           + df["renew_total"] - df["demand_actual"])    # crude realized headroom

# wind forecast error (NOWCAST only - not known at DA stage)
df["wind_fcst_err"] = df["wind_total"] - df["wind_fcst_da"]

# ----------------------------------------------------------------- calendar
idx = df.index
df["hour"] = idx.hour + idx.minute / 60
df["dow"] = idx.dayofweek
df["month"] = idx.month
df["is_weekend"] = (df["dow"] >= 5).astype(int)
df["is_evening_peak"] = ((df["hour"] >= 16) & (df["hour"] <= 19)).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365)
df["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365)

# ----------------------------------------------------------------- lags (DA-safe: >= 48 SPs)
for lag in [48, 96, 336]:
    df[f"sysprice_lag{lag}"] = df["system_price"].shift(lag)
    df[f"niv_lag{lag}"] = df["niv"].shift(lag)
df["sysprice_d1_mean"] = df["system_price"].shift(48).rolling(48).mean()
df["sysprice_d1_max"]  = df["system_price"].shift(48).rolling(48).max()
df["sysprice_w_std"]   = df["system_price"].shift(48).rolling(336).std()
df["spike_rate_30d"]   = (df["system_price"].shift(48)
                          > df["system_price"].shift(48).rolling(48 * 90, min_periods=48 * 30)
                            .quantile(0.95)).rolling(48 * 30).mean()

# nowcast-only short lags
df["sysprice_lag2"] = df["system_price"].shift(2)
df["niv_lag2"] = df["niv"].shift(2)

# ----------------------------------------------------------------- target
roll_q95 = (df["system_price"].shift(1)
            .rolling(48 * 90, min_periods=48 * 30).quantile(0.95))
df["spike_threshold"] = roll_q95
df["spike"] = (df["system_price"] > df["spike_threshold"]).astype(int)

df = df.dropna(subset=["spike_threshold", "nd_fcst_da", "wind_fcst_da",
                       "sysprice_lag336", "temp_gb"])

FEATURES_DA = [
    # forecast fundamentals (DA vintages)
    "nd_fcst_da", "tsd_fcst_da", "wind_fcst_da", "resid_demand_fcst", "renew_pen_fcst",
    # weather (perfect-foresight forecast proxy)
    "temp_gb", "rh_gb", "solar_gb", "wind_ms_offshore", "hdd", "cdd",
    # calendar
    "hour_sin", "hour_cos", "dow", "month", "is_weekend", "is_evening_peak",
    "doy_sin", "doy_cos",
    # DA-safe history
    "sysprice_lag48", "sysprice_lag96", "sysprice_lag336",
    "niv_lag48", "sysprice_d1_mean", "sysprice_d1_max",
    "sysprice_w_std", "spike_rate_30d",
]
FEATURES_NOWCAST = FEATURES_DA + [
    "wind_fcst_err", "resid_demand_act", "renew_pen_act",
    "sysprice_lag2", "niv_lag2",
]

df.to_parquet("/home/claude/data/dataset.parquet")
df.reset_index().to_csv("/home/claude/data/dataset.csv", index=False)

print(f"rows: {len(df)}, span: {df.index.min()} -> {df.index.max()}")
print(f"spike base rate: {df['spike'].mean():.3%}")
print(f"price: mean {df['system_price'].mean():.1f}, p95 {df['system_price'].quantile(.95):.1f}, "
      f"p99 {df['system_price'].quantile(.99):.1f}, max {df['system_price'].max():.0f}")
print("NaN check DA features:", df[FEATURES_DA].isna().sum().sum())
