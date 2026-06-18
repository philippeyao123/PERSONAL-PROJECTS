"""Global configuration for the DE-LU power fair-value pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
for p in (DATA, FIGS, REPORTS, LOGS):
    p.mkdir(exist_ok=True)

BZN = "DE-LU"                  # bidding zone
COUNTRY = "de"
START = "2024-09-01"           # ~21 months of history
END = "2026-06-11"

# Weather grid: 5 demand/generation-weighted German locations
CITIES = {
    "berlin":    (52.52, 13.40),
    "hamburg":   (53.55, 9.99),
    "munich":    (48.14, 11.58),
    "frankfurt": (50.11, 8.68),
    "cologne":   (50.94, 6.96),
}
WEATHER_VARS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation"]

TEST_DAYS = 180                # out-of-sample window (last N days)
RETRAIN_EVERY = 7              # retrain cadence (days)

SIGNAL_Z = 0.75                # z-score threshold for non-neutral view
PROMPT_WINDOW = 7              # trailing days defining the front-week prompt proxy
