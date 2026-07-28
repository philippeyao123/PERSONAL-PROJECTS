"""AI/LLM component: automated morning desk commentary.

Purpose
-------
The last manual step in a daily fair-value process is writing the morning
note: turning the forecast, drivers and QA status into two readable
paragraphs for the desk. This module automates it with one structured LLM
call (Anthropic Messages API, model claude-sonnet-4-6).

Design choices
--------------
- The LLM is instructed not to produce new numbers: every figure is computed
  by the pipeline and injected into the prompt. The model only verbalises,
  which reduces but does not eliminate generation risk.
- Full prompt and raw response are logged to logs/ with a timestamp, so the
  AI step is auditable and reproducible.
- `--dry-run` builds and logs the exact prompt without an API call (useful
  in CI or without credentials); otherwise set ANTHROPIC_API_KEY.

Usage
-----
    python src/ai_commentary.py            # live call (needs ANTHROPIC_API_KEY)
    python src/ai_commentary.py --dry-run  # build + log prompt only
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from config import DATA, LOGS, REPORTS

MODEL = "claude-sonnet-4-6"

SYSTEM = (
    "You are a power-market analyst writing a concise morning note for a "
    "prop trading desk. Use ONLY the numbers provided in the user message - "
    "never invent figures. Two short paragraphs: (1) tomorrow's DA fair "
    "value, the key drivers, and the shape of the day; (2) the prompt-curve "
    "view, its conviction, and what would invalidate it. Plain prose, no "
    "bullet points, no headers, max 160 words."
)


def gather_context() -> dict:
    """Collect everything the note needs from pipeline outputs."""
    daily = pd.read_csv(DATA / "daily_views.csv", index_col=0)
    daily.index = pd.to_datetime(daily.index, utc=True).tz_convert("Europe/Berlin")
    feats = pd.read_csv(DATA / "dataset.csv", index_col=0)
    feats.index = pd.to_datetime(feats.index, utc=True)
    qa = json.loads((REPORTS / "qa_report.json").read_text())
    metrics = json.loads((REPORTS / "model_metrics.json").read_text())

    last = daily.iloc[-1]
    day = daily.index[-1]
    local = feats.index.tz_convert("Europe/Berlin")
    mask = local.normalize() == day.normalize()
    fday = feats.loc[mask]

    return {
        "delivery_day": str(day.date()),
        "fair_value_baseload": round(float(last["fair_value"]), 2),
        "prompt_proxy": round(float(last["prompt_proxy"]), 2),
        "gap_eur_mwh": round(float(last["gap"]), 2),
        "gap_zscore": round(float(last["gap_z"]), 2),
        "view": str(last["view"]),
        "wind_fcst_avg_gw": round(float(fday["fcst_wind_total"].mean()) / 1000, 1),
        "solar_fcst_peak_gw": round(float(fday["fcst_solar"].max()) / 1000, 1),
        "temp_fcst_avg_c": round(float(fday["wx_temperature_2m"].mean()), 1),
        "model_oos_mae": metrics["metrics"]["lgbm"]["MAE"],
        "model_skill_vs_naive": metrics["metrics"]["lgbm"]["skill_vs_naive"],
        "qa_status": qa["status"],
        "qa_warnings": qa["warnings"],
    }


def build_prompt(ctx: dict) -> str:
    return ("Write today's morning note from this pipeline output:\n\n"
            + json.dumps(ctx, indent=2))


def call_llm(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=MODEL, max_tokens=400, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def main() -> None:
    dry = "--dry-run" in sys.argv
    ctx = gather_context()
    prompt = build_prompt(ctx)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = {"timestamp": stamp, "model": MODEL, "system": SYSTEM,
           "prompt": prompt, "dry_run": dry}

    if dry:
        log["response"] = None
        print("[dry-run] prompt built and logged; no API call made.")
    else:
        note = call_llm(prompt)
        log["response"] = note
        (REPORTS / "morning_note.md").write_text(
            f"# Morning note -- {ctx['delivery_day']}\n\n{note}\n")
        print(note)

    (LOGS / f"llm_call_{stamp}.json").write_text(json.dumps(log, indent=2))
    print(f"logged -> logs/llm_call_{stamp}.json")


if __name__ == "__main__":
    main()
