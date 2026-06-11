"""
Data ingestion for UK electricity price spike forecasting.
Pulls 2 years of half-hourly data from Elexon Insights API (free, no auth)
and hourly weather from Open-Meteo archive API.

All forecast series are DAY-AHEAD vintages (published before delivery day)
so the modelling dataset is leakage-free for a day-ahead prediction task.
"""
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import time, sys

BASE = "https://data.elexon.co.uk/bmrs/api/v1"
START = date(2024, 6, 1)
END = date(2026, 5, 31)
OUT = "/home/claude/data"

session = requests.Session()

def get(url, retries=4):
    for i in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            time.sleep(1 + i)
        except Exception:
            time.sleep(1 + i)
    return None

def daterange_chunks(start, end, days):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)

# ---------------------------------------------------------------- system prices (per day)
def fetch_system_prices():
    days = [START + timedelta(d) for d in range((END - START).days + 1)]
    rows = []
    def one(d):
        j = get(f"{BASE}/balancing/settlement/system-prices/{d.isoformat()}")
        return j.get("data", []) if j else []
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(one, d): d for d in days}
        for k, f in enumerate(as_completed(futs)):
            rows.extend(f.result())
            if k % 100 == 0:
                print(f"  system prices {k}/{len(days)}", flush=True)
    df = pd.DataFrame(rows)[["startTime", "systemSellPrice", "netImbalanceVolume", "reserveScarcityPrice"]]
    df.columns = ["time", "system_price", "niv", "scarcity_price"]
    return df

# ---------------------------------------------------------------- demand outturn (14d chunks)
def fetch_demand_outturn():
    rows = []
    chunks = list(daterange_chunks(START, END, 14))
    def one(c):
        j = get(f"{BASE}/demand/outturn?settlementDateFrom={c[0]}&settlementDateTo={c[1]}")
        return j.get("data", []) if j else []
    with ThreadPoolExecutor(8) as ex:
        for f in as_completed({ex.submit(one, c): c for c in chunks}):
            rows.extend(f.result())
    df = pd.DataFrame(rows)[["startTime", "initialTransmissionSystemDemandOutturn"]]
    df.columns = ["time", "demand_actual"]
    return df

# ---------------------------------------------------------------- generation per type (7d chunks)
def fetch_generation():
    rows = []
    chunks = list(daterange_chunks(START, END, 7))
    def one(c):
        j = get(f"{BASE}/generation/actual/per-type?from={c[0]}T00:00Z&to={c[1]}T23:59Z")
        return j.get("data", []) if j else []
    with ThreadPoolExecutor(8) as ex:
        for k, f in enumerate(as_completed({ex.submit(one, c): c for c in chunks})):
            rows.extend(f.result())
    recs = []
    for r in rows:
        d = {x["psrType"]: x["quantity"] for x in r["data"]}
        recs.append({
            "time": r["startTime"],
            "wind_on": d.get("Wind Onshore", 0), "wind_off": d.get("Wind Offshore", 0),
            "solar": d.get("Solar", 0), "gas": d.get("Fossil Gas", 0),
            "nuclear": d.get("Nuclear", 0), "biomass": d.get("Biomass", 0),
            "hydro": d.get("Hydro Run-of-river and poundage", 0) + d.get("Hydro Pumped Storage", 0),
        })
    return pd.DataFrame(recs)

# ---------------------------------------------------------------- market index price (7d chunks)
def fetch_mid():
    rows = []
    chunks = list(daterange_chunks(START, END, 7))
    def one(c):
        j = get(f"{BASE}/balancing/pricing/market-index?from={c[0]}T00:00Z&to={c[1]}T23:59Z&dataProviders=APXMIDP")
        return j.get("data", []) if j else []
    with ThreadPoolExecutor(8) as ex:
        for f in as_completed({ex.submit(one, c): c for c in chunks}):
            rows.extend(f.result())
    df = pd.DataFrame(rows)[["startTime", "price", "volume"]]
    df.columns = ["time", "mid_price", "mid_volume"]
    return df

# ---------------------------------------------------------------- day-ahead forecast vintages (per publish day)
def fetch_da_wind_forecast():
    """Wind forecast published ~16:30 on D-1, keep rows for settlementDate == D."""
    days = [START + timedelta(d) for d in range((END - START).days + 1)]
    rows = []
    def one(d):
        pub = d - timedelta(days=1)
        j = get(f"{BASE}/forecast/generation/wind/history?publishTime={pub.isoformat()}T16:30Z")
        if not j:
            return []
        return [r for r in j.get("data", []) if r.get("settlementDate") == d.isoformat()]
    with ThreadPoolExecutor(8) as ex:
        for k, f in enumerate(as_completed({ex.submit(one, d): d for d in days})):
            rows.extend(f.result())
            if k % 100 == 0:
                print(f"  wind fcst {k}/{len(days)}", flush=True)
    df = pd.DataFrame(rows)[["startTime", "generation"]]
    df.columns = ["time", "wind_fcst_da"]
    return df.groupby("time", as_index=False).mean()

def fetch_da_demand_forecast():
    """National demand forecast published morning of D-1, keep settlementDate == D."""
    days = [START + timedelta(d) for d in range((END - START).days + 1)]
    rows = []
    def one(d):
        pub = d - timedelta(days=1)
        j = get(f"{BASE}/forecast/demand/day-ahead/history?publishTime={pub.isoformat()}T09:00Z")
        if not j:
            return []
        return [r for r in j.get("data", [])
                if r.get("settlementDate") == d.isoformat() and r.get("boundary") == "N"]
    with ThreadPoolExecutor(8) as ex:
        for k, f in enumerate(as_completed({ex.submit(one, d): d for d in days})):
            rows.extend(f.result())
            if k % 100 == 0:
                print(f"  demand fcst {k}/{len(days)}", flush=True)
    df = pd.DataFrame(rows)[["startTime", "transmissionSystemDemand", "nationalDemand"]]
    df.columns = ["time", "tsd_fcst_da", "nd_fcst_da"]
    return df.groupby("time", as_index=False).mean()

# ---------------------------------------------------------------- weather (Open-Meteo archive)
LOCS = {  # demand-weighted population centres + wind belts
    "london":     (51.51, -0.13),
    "birmingham": (52.48, -1.90),
    "manchester": (53.48, -2.24),
    "glasgow":    (55.86, -4.25),
    "hornsea":    (53.88,  1.79),   # North Sea offshore wind
    "irishsea":   (53.80, -3.60),   # Irish Sea offshore wind
}

def fetch_weather():
    frames = []
    for name, (lat, lon) in LOCS.items():
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={START}&end_date={END}"
               f"&hourly=temperature_2m,wind_speed_100m,shortwave_radiation,relative_humidity_2m&timezone=UTC")
        j = get(url)
        h = j["hourly"]
        df = pd.DataFrame({"time": h["time"],
                           f"temp_{name}": h["temperature_2m"],
                           f"wind_{name}": h["wind_speed_100m"],
                           f"solar_{name}": h["shortwave_radiation"],
                           f"rh_{name}": h["relative_humidity_2m"]})
        frames.append(df.set_index("time"))
        print(f"  weather {name} ok", flush=True)
        time.sleep(1)
    return pd.concat(frames, axis=1).reset_index()

if __name__ == "__main__":
    jobs = {
        "system_prices": fetch_system_prices,
        "demand_actual": fetch_demand_outturn,
        "generation": fetch_generation,
        "mid_price": fetch_mid,
        "wind_fcst": fetch_da_wind_forecast,
        "demand_fcst": fetch_da_demand_forecast,
        "weather": fetch_weather,
    }
    for name, fn in jobs.items():
        print(f"== {name}", flush=True)
        df = fn()
        df.to_csv(f"{OUT}/{name}.csv", index=False)
        print(f"   saved {len(df)} rows", flush=True)
