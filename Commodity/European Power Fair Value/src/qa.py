"""Data quality checks on the assembled hourly dataset.

Checks implemented:
  1. completeness      - missing hours on the expected hourly UTC grid
  2. duplicates        - duplicated timestamps
  3. nulls             - per-column null share
  4. range_sanity      - prices within plausible band; renewables non-negative
                         and below installed capacity; weather within physical bounds
  5. negative_prices   - share of negative hours (real feature of DE-LU, flagged not dropped)
  6. spikes            - |robust z-score| > 8 on hourly price changes (flag only)
  7. staleness         - runs of >= 24 identical consecutive prices (stuck feed)
  8. dst_integrity     - hour counts per local calendar day in {23, 24, 25}

Output: reports/qa_report.json and a console summary. The pipeline FAILS
(exit 1) on structural errors (duplicates, broken DST days, >2% missing),
and only WARNS on market features (negative prices, spikes).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from config import DATA, REPORTS

BOUNDS = {
    "price": (-500.0, 4000.0),          # EPEX technical limits
    "fcst_solar": (0.0, 120_000.0),     # MW, vs ~100 GW installed PV
    "fcst_wind_total": (0.0, 80_000.0),
    "wx_temperature_2m": (-30.0, 45.0),
    "wx_wind_speed_100m": (0.0, 60.0),  # m/s... API returns km/h; bound set post-inspection
    "wx_shortwave_radiation": (0.0, 1100.0),
}


def run_qa(df: pd.DataFrame) -> dict:
    rep: dict = {"n_rows": int(len(df)),
                 "span": [str(df.index.min()), str(df.index.max())]}

    # 1. completeness
    full = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    missing = full.difference(df.index)
    rep["completeness"] = {
        "expected_hours": len(full),
        "missing_hours": len(missing),
        "missing_pct": round(100 * len(missing) / len(full), 3),
        "first_missing": [str(t) for t in missing[:5]],
    }

    # 2. duplicates
    rep["duplicates"] = int(df.index.duplicated().sum())

    # 3. nulls
    rep["null_pct"] = {c: round(100 * df[c].isna().mean(), 3) for c in df.columns}

    # 4. range sanity
    viol = {}
    for col, (lo, hi) in BOUNDS.items():
        if col in df:
            bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            viol[col] = bad
    rep["range_violations"] = viol

    # 5. negative prices (informational)
    rep["negative_price_pct"] = round(100 * float((df["price"] < 0).mean()), 2)

    # 6. spikes on price changes (robust z via MAD)
    dpx = df["price"].diff().dropna()
    mad = (dpx - dpx.median()).abs().median()
    rz = 0.6745 * (dpx - dpx.median()) / max(mad, 1e-9)
    spikes = dpx.index[rz.abs() > 8]
    rep["price_spikes"] = {"count": int(len(spikes)),
                           "examples": [str(t) for t in spikes[:5]]}

    # 7. staleness
    runs = (df["price"] != df["price"].shift()).cumsum()
    longest = int(df.groupby(runs)["price"].transform("size").max())
    rep["longest_constant_price_run_h"] = longest

    # 8. DST integrity in local time
    local = df.tz_convert("Europe/Berlin")
    per_day = local.groupby(local.index.date).size()
    bad_days = per_day[~per_day.isin([23, 24, 25])]
    # exclude partial first/last day of the sample
    bad_days = bad_days[~bad_days.index.isin([local.index[0].date(), local.index[-1].date()])]
    rep["dst_bad_days"] = {str(k): int(v) for k, v in bad_days.items()}

    # verdict
    errors = []
    if rep["duplicates"]:
        errors.append("duplicate timestamps")
    if rep["completeness"]["missing_pct"] > 2.0:
        errors.append("missing hours > 2%")
    if rep["dst_bad_days"]:
        errors.append("DST-inconsistent days")
    if any(v > 0 for v in viol.values()):
        errors.append("range violations")
    rep["status"] = "FAIL" if errors else "PASS"
    rep["errors"] = errors
    rep["warnings"] = []
    if rep["negative_price_pct"] > 0:
        rep["warnings"].append(
            f"{rep['negative_price_pct']}% negative price hours "
            "(expected in DE-LU under high renewables; kept in dataset)")
    if longest >= 24:
        rep["warnings"].append(f"constant price run of {longest}h (possible stale feed)")
    return rep


def main() -> int:
    df = pd.read_csv(DATA / "dataset.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    rep = run_qa(df)
    (REPORTS / "qa_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nQA status: {rep['status']}")
    return 0 if rep["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
