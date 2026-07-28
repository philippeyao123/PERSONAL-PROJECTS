"""Data ingestion: DE-LU day-ahead prices + ex-ante fundamental drivers.

Sources (all public, no API key required):
  1. energy-charts.info (Fraunhofer ISE, CC BY 4.0)
     - /price                : EPEX day-ahead auction prices, bidding zone DE-LU
     - /public_power_forecast: TSO day-ahead forecasts of solar / wind generation
  2. Open-Meteo Historical Forecast API (CC BY 4.0)
     - a continuous series stitched from the first hours of successive
       operational forecasts.  Because this does not preserve a fixed D-1
       lead time, weather is excluded from the primary publication model and
       retained only for a labelled sensitivity.

All series are stored on a UTC hourly index.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

from config import BZN, CITIES, COUNTRY, DATA, END, START, WEATHER_VARS

EC_BASE = "https://api.energy-charts.info"
OM_BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "power-fv-case-study/1.0"})


MIN_GAP_S = 12.0  # polite pacing between energy-charts calls
_last_call = [0.0]


def _get(url: str, params: dict, retries: int = 6) -> dict:
    """GET with disk cache, pacing and exponential backoff (handles 429s).

    Each (url, params) response is cached under data/raw/ so the pipeline
    is resumable and never re-hits the APIs unnecessarily.
    """
    import hashlib
    import json as _json
    cache_dir = DATA / "raw"
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.md5((url + _json.dumps(params, sort_keys=True)).encode()).hexdigest()
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        return _json.loads(cache_file.read_text())
    pace = MIN_GAP_S if "energy-charts" in url else 0.5
    for attempt in range(retries):
        gap = pace - (time.time() - _last_call[0])
        if gap > 0:
            time.sleep(gap)
        try:
            r = SESSION.get(url, params=params, timeout=120)
            _last_call[0] = time.time()
            r.raise_for_status()
            data = r.json()
            cache_file.write_text(_json.dumps(data))
            return data
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            wait = 10 * 2 ** attempt
            print(f"  retry {attempt + 1} after error: {exc} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _month_chunks(start: str, end: str, months: int = 6):
    """Yield (start, end) date-string pairs spanning `months` calendar months."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    cur = s
    while cur <= e:
        nxt = cur
        for _ in range(months):
            nxt = (nxt.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cur.isoformat(), min(nxt - timedelta(days=1), e).isoformat()
        cur = nxt


def fetch_prices() -> pd.DataFrame:
    """EPEX day-ahead auction price, DE-LU, resampled to hourly EUR/MWh.

    Note: DE-LU moved to 15-minute MTUs in 2025; hourly mean of the four
    quarters equals the hourly-equivalent baseload price.
    """
    frames = []
    for s, e in _month_chunks(START, END):
        print(f"  prices {s} -> {e}")
        d = _get(f"{EC_BASE}/price", {"bzn": BZN, "start": s, "end": e})
        idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
        frames.append(pd.Series(d["price"], index=idx, name="price", dtype="float64"))
    raw = pd.concat(frames).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]  # chunk-boundary overlaps
    hourly = raw.resample("1h").mean().to_frame()
    return hourly


def fetch_renewables_forecast() -> pd.DataFrame:
    """TSO day-ahead forecasts of solar + wind generation (MW), hourly."""
    out = {}
    for pt in ("solar", "wind_onshore", "wind_offshore"):
        frames = []
        for s, e in _month_chunks(START, END):
            d = _get(
                f"{EC_BASE}/public_power_forecast",
                {"country": COUNTRY, "production_type": pt,
                 "forecast_type": "day-ahead", "start": s, "end": e},
            )
            idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
            frames.append(pd.Series(d["forecast_values"], index=idx, dtype="float64"))
        ser = pd.concat(frames).sort_index()
        ser = ser[~ser.index.duplicated(keep="last")].resample("1h").mean()
        out[f"fcst_{pt}"] = ser
        print(f"  renewables DA forecast: {pt} ({len(ser)} h)")
    df = pd.DataFrame(out)
    df["fcst_wind_total"] = df["fcst_wind_onshore"] + df["fcst_wind_offshore"]
    return df


def fetch_weather_forecast() -> pd.DataFrame:
    """Open-Meteo stitched historical forecasts averaged across five cities.

    These national proxies do not preserve a fixed target-hour lead time and
    must not be included in the primary D-1 specification.
    """
    city_frames = []
    for name, (lat, lon) in CITIES.items():
        print(f"  weather forecast: {name}")
        d = _get(
            OM_BASE,
            {"latitude": lat, "longitude": lon,
             "hourly": ",".join(WEATHER_VARS),
             "start_date": START, "end_date": END, "timezone": "UTC"},
        )
        h = d["hourly"]
        idx = pd.to_datetime(h["time"], utc=True)
        city_frames.append(pd.DataFrame({v: h[v] for v in WEATHER_VARS}, index=idx))
    # simple cross-city mean as the national proxy
    panel = pd.concat(city_frames, keys=CITIES.keys())
    natl = panel.groupby(level=1).mean()
    natl.columns = [f"wx_{c}" for c in natl.columns]
    return natl


def build_dataset() -> pd.DataFrame:
    print("Fetching day-ahead prices (energy-charts / EPEX)...")
    px = fetch_prices()
    print("Fetching TSO day-ahead renewables forecasts (energy-charts)...")
    ren = fetch_renewables_forecast()
    print("Fetching weather forecasts as issued (Open-Meteo)...")
    wx = fetch_weather_forecast()

    df = px.join(ren, how="left").join(wx, how="left")
    df.index.name = "ts_utc"
    df = df.loc[df["price"].notna()]
    df.to_csv(DATA / "dataset.csv")
    print(f"Dataset: {df.shape[0]} hourly rows x {df.shape[1]} cols "
          f"({df.index.min()} -> {df.index.max()})")
    return df


if __name__ == "__main__":
    build_dataset()
    sys.exit(0)
