# European Power Fair Value — Case Study Write-Up

**Candidate:** Philippe — *[your email]*
**Market & option:** DE-LU · Option A (next-day hourly prices) · June 2026

---

## 1. Problem framing

A day-ahead fair-value process answers one question every morning before the 12:00 CET auction: *where should tomorrow's 24 hourly prices clear given what is knowable now?* The answer is only tradable if (i) the information set used in the backtest is exactly the live one, and (ii) the forecast can be expressed against an instrument — here, the front-week prompt. The prototype was built around those two constraints.

## 2. Data & QA

**Sources (public, key-free, cached locally):** EPEX day-ahead auction prices for DE-LU from energy-charts.info (Fraunhofer ISE); TSO **day-ahead forecasts** of solar, onshore and offshore wind from the same API; and weather **forecasts as originally issued** (temperature 2 m, wind 100 m, shortwave radiation, five-city national proxy) from Open-Meteo's Historical Forecast API. Using forecast vintages rather than realised fundamentals removes look-ahead leakage by construction — realised wind on day D is not knowable at D-1 12:00, but the TSO's D-1 forecast is. Coverage: 2024-09-01 → 2026-06-11, 15,576 hourly observations on a UTC index (15-minute MTUs post-2025 averaged to hourly).

**QA** (`reports/qa_report.json`, status **PASS**) runs eight checks: grid completeness, duplicates, nulls, physical range bounds, negative-price share, MAD-z spike flags, stale-feed runs, and DST integrity (23/24/25 hours per local day). The pipeline fails hard on structural defects and only warns on genuine market features: 5.92% of hours are negative-priced — a defining property of DE-LU under renewables saturation — and are kept, since a fair-value model that cannot price negative hours is not fit for purpose.

## 3. Forecasting & validation

**Features (22)** respect the D-1 information set: price lags ≥ 24 h (D-1 prices publish at D-2 12:45), realised-day statistics, TSO renewables forecasts and their 24 h swing, weather forecasts with HDD/CDD transforms, and calendar terms including German holidays. **Models:** a naive same-hour-last-week benchmark, a standardised Ridge baseline, and LightGBM. **Validation** is walk-forward over the final 180 delivery days — expanding window, weekly refits, whole days predicted ahead — mirroring a production cadence.

| Model | MAE (€/MWh) | RMSE | R² | Skill vs naive |
|---|---|---|---|---|
| Naive (lag-168 h) | 34.11 | 53.00 | 0.02 | — |
| Ridge | 17.45 | 26.23 | 0.76 | +48.8% |
| **LightGBM** | **13.97** | **22.81** | **0.82** | **+59.0%** |

Two diagnostics matter for trading. First, errors peak at the **evening ramp (17–21 h)** where solar hands over to wind and the merit order is steepest — that is where forecast-driven positions deserve a haircut. Second, importance is dominated by recent price structure and the renewables forecast/swing: at the DA horizon, **feature quality beats model complexity**, and the linear baseline already captures half the available skill.

## 4. From forecast to prompt-curve view

The 24 hourly forecasts aggregate to a **fair-value baseload** for D. The reference mark is a **prompt proxy** — the trailing 7-day realised DA baseload (front-week EEX settlements are not freely redistributable; prompt forwards settle near expected DA outturn, and the code accepts licensed quotes directly). The signal is the 60-day rolling z-score of (fair value − prompt): |z| > 0.75 generates a LONG/SHORT front-week view, otherwise NEUTRAL.

Scoring is deliberately stricter than the forecast horizon: each view is judged against the **realised baseload of the delivered week (D…D+6)** versus the prompt mark, so being right requires the divergence to persist beyond the day the model actually forecasts. Out of 144 evaluable days, 63 views were active (29 long / 34 short) with a **76.2% hit rate** and **+17.2 €/MWh average captured spread**.

**Use & invalidation.** The view sizes with |z| and is recomputed every morning. It is flattened on: a TSO intraday revision of D+1 wind+solar exceeding 5 GW versus model inputs; a REMIT unplanned outage above 2 GW; a model residual on the last realised day above 2× its trailing 30-day std (regime break); or a >5% move in TTF/EUA — the model carries no fuel features, so fuel-led curve moves are explicitly out of scope rather than silently mispriced.

## 5. AI/LLM component

The one remaining manual task in the daily loop is writing the morning note. `src/ai_commentary.py` makes a single Anthropic Messages API call (`claude-sonnet-4-6`): the pipeline computes a JSON context (fair value, gap z-score, drivers in GW/°C, OOS MAE, QA status) and the model is instructed to **verbalise only — it may not produce any number itself**, which removes hallucination risk on quantitative content. Every call logs the full prompt and raw response to `logs/` with a timestamp, making the AI step auditable; a `--dry-run` mode builds and logs the prompt without credentials. Example prompt, response and the rendered `reports/morning_note.md` are included.

## 6. Honest limitations

The prompt proxy is not a tradable quote; with licensed EEX marks the same signal would be evaluated against settlements net of costs. The model is weather/renewables-only by design — a clean-spark/dark-spread feature is the natural extension for the curve leg. And the point forecast should graduate to quantile bands so the signal can size on dispersion, not just the gap.

**Reproduction:** `pip install -r requirements.txt && python run_all.py` (Python 3.12; first run pulls ~21 months with polite pacing, all API responses cached under `data/raw/`).
