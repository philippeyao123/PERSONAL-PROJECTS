"""Convert frozen CSV/JSON evidence into LaTeX tables and result macros."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
OUT = ROOT / "paper" / "generated"
OUT.mkdir(parents=True, exist_ok=True)


def command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def sci(value: float) -> str:
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10**exponent
    return f"${mantissa:.2f}\\times 10^{{{exponent}}}$"


def write_macros() -> None:
    research = json.loads((REPORTS / "research_metrics.json").read_text())
    qa = json.loads((REPORTS / "qa_report.json").read_text())
    comparison = pd.read_csv(DATA / "model_comparison.csv").set_index("model")
    dm = pd.read_csv(DATA / "dm_tests.csv").set_index("comparator")
    conformal = research["conformal"]
    ablation = pd.read_csv(DATA / "ablation_metrics.csv").set_index("specification")
    regimes = pd.read_csv(DATA / "regime_metrics.csv")
    primary_regimes = regimes[regimes["model"] == "lgbm"].set_index(
        ["dimension", "level"]
    )
    sensitivity = pd.read_csv(DATA / "signal_sensitivity.csv")
    default = sensitivity[
        (sensitivity["prompt_window"] == 7)
        & (sensitivity["threshold"] == 0.75)
    ].iloc[0]
    tail = pd.read_csv(DATA / "tail_metrics.csv")

    def tail_value(regime: str, model: str, column: str) -> float:
        row = tail[(tail["regime"] == regime) & (tail["model"] == model)].iloc[0]
        return float(row[column])

    values = {
        "DataRows": f"{research['data']['rows']:,}",
        "DataStart": research["data"]["start_utc"][:10],
        "DataEnd": research["data"]["end_utc"][:10],
        "NegativeShare": f"{qa['negative_price_pct']:.2f}\\%",
        "OOSObservations": f"{int(comparison.loc['lgbm', 'observations']):,}",
        "OOSDays": f"{int(comparison.loc['lgbm', 'delivery_days'])}",
        "PrimaryMAE": f"{comparison.loc['lgbm', 'mae']:.2f}",
        "PrimaryMAELow": f"{comparison.loc['lgbm', 'mae_ci_low']:.2f}",
        "PrimaryMAEHigh": f"{comparison.loc['lgbm', 'mae_ci_high']:.2f}",
        "PrimaryRMSE": f"{comparison.loc['lgbm', 'rmse']:.2f}",
        "PrimaryBias": f"{comparison.loc['lgbm', 'bias']:.2f}",
        "PostWarmupMAE": f"{comparison.loc['lgbm', 'mae_after_first_90_days']:.2f}",
        "PostWarmupDays": f"{int(comparison.loc['lgbm', 'post_warmup_delivery_days'])}",
        "InitialTrainingDays": f"{research['data']['pre_oos_training_days']}",
        "NaiveMAE": f"{comparison.loc['naive_w', 'mae']:.2f}",
        "RidgeMAE": f"{comparison.loc['ridge', 'mae']:.2f}",
        "HourlyRidgeMAE": f"{comparison.loc['ridge_hourly', 'mae']:.2f}",
        "SkillNaive": (
            f"{100 * (1 - comparison.loc['lgbm', 'mae'] / comparison.loc['naive_w', 'mae']):.1f}\\%"
        ),
        "SkillRidge": (
            f"{100 * (1 - comparison.loc['lgbm', 'mae'] / comparison.loc['ridge', 'mae']):.1f}\\%"
        ),
        "SkillHourlyRidge": (
            f"{100 * (1 - comparison.loc['lgbm', 'mae'] / comparison.loc['ridge_hourly', 'mae']):.1f}\\%"
        ),
        "DMGainNaive": f"{dm.loc['naive_w', 'mean_loss_improvement']:.2f}",
        "DMGainRidge": f"{dm.loc['ridge', 'mean_loss_improvement']:.2f}",
        "DMGainHourlyRidge": f"{dm.loc['ridge_hourly', 'mean_loss_improvement']:.2f}",
        "ConformalCoverage": f"{100 * conformal['empirical_coverage']:.2f}\\%",
        "ConformalObservations": f"{conformal['observations']:,}",
        "ConformalWidth": f"{conformal['mean_width']:.2f}",
        "ConformalIntervalScore": f"{conformal['mean_interval_score']:.2f}",
        "ConditionalSignificantHours": f"{conformal['normal_approx_significant_hours']}",
        "ConditionalHolmHours": f"{conformal['holm_significant_hours']}",
        "ConditionalUnderHours": f"{conformal['undercover_significant_hours']}",
        "ConditionalOverHours": f"{conformal['overcover_significant_hours']}",
        "BiasCorrectedMAE": f"{research['bias_correction']['corrected_mae']:.2f}",
        "BiasCorrectedBias": f"{research['bias_correction']['corrected_bias']:.2f}",
        "BiasWindowMAE": f"{research['bias_correction']['uncorrected_mae']:.2f}",
        "BiasWindowBias": f"{research['bias_correction']['uncorrected_bias']:.2f}",
        "WeatherDelta": (
            f"{ablation.loc['weather_augmented', 'mae_change_vs_primary_pct']:.2f}\\%"
        ),
        "NoRenewablesDelta": (
            f"{ablation.loc['no_renewables', 'mae_change_vs_primary_pct']:.1f}\\%"
        ),
        "FundamentalsCalendarMAE": f"{ablation.loc['fundamentals_calendar', 'mae']:.2f}",
        "NoRenewablesMAE": f"{ablation.loc['no_renewables', 'mae']:.2f}",
        "SignalActive": f"{int(default['evaluable_active_days'])}",
        "SignalHit": f"{100 * default['hit_rate']:.1f}\\%",
        "SignalCaptured": f"{default['average_captured']:.2f}",
        "SignalHitLow": (
            f"{100 * research['default_signal_block_bootstrap']['hit_rate_ci'][0]:.1f}\\%"
        ),
        "SignalHitHigh": (
            f"{100 * research['default_signal_block_bootstrap']['hit_rate_ci'][1]:.1f}\\%"
        ),
        "SignalMeanLow": (
            f"{research['default_signal_block_bootstrap']['mean_ci'][0]:.2f}"
        ),
        "SignalMeanHigh": (
            f"{research['default_signal_block_bootstrap']['mean_ci'][1]:.2f}"
        ),
        "MAENight": f"{primary_regimes.loc[('hour_block', 'night'), 'mae']:.2f}",
        "MAEDaylight": f"{primary_regimes.loc[('hour_block', 'daylight'), 'mae']:.2f}",
        "MAEEveningRamp": f"{primary_regimes.loc[('hour_block', 'evening_ramp'), 'mae']:.2f}",
        "MAELateEvening": f"{primary_regimes.loc[('hour_block', 'late_evening'), 'mae']:.2f}",
        "MAESpring": f"{primary_regimes.loc[('season', 'spring'), 'mae']:.2f}",
        "MAEAutumn": f"{primary_regimes.loc[('season', 'autumn'), 'mae']:.2f}",
        "MAESummer": f"{primary_regimes.loc[('season', 'summer'), 'mae']:.2f}",
        "MAEWinter": f"{primary_regimes.loc[('season', 'winter'), 'mae']:.2f}",
        "NegativeMAE": f"{tail_value('negative', 'lgbm', 'mae'):.2f}",
        "RegularMAE": f"{tail_value('regular', 'lgbm', 'mae'):.2f}",
        "HighMAE": f"{tail_value('above_200', 'lgbm', 'mae'):.2f}",
        "HighRecall": (
            f"{100 * tail_value('above_200_detection', 'lgbm', 'mae'):.1f}\\%"
        ),
        "RenewableQOneMAE": f"{primary_regimes.loc[('renewable_quartile', 'Q1_low'), 'mae']:.2f}",
        "RenewableQTwoMAE": f"{primary_regimes.loc[('renewable_quartile', 'Q2'), 'mae']:.2f}",
        "RenewableQThreeMAE": f"{primary_regimes.loc[('renewable_quartile', 'Q3'), 'mae']:.2f}",
        "RenewableQFourMAE": f"{primary_regimes.loc[('renewable_quartile', 'Q4_high'), 'mae']:.2f}",
    }
    (OUT / "results_macros.tex").write_text(
        "".join(command(k, v) for k, v in values.items())
    )


def write_model_table() -> None:
    frame = pd.read_csv(DATA / "model_comparison.csv")
    labels = {
        "naive_w": "Weekly naive",
        "ridge": "Pooled Ridge",
        "ridge_hourly": "Hourly Ridge (24 models)",
        "lgbm": "LightGBM",
    }
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"{labels[row.model]} & {row.mae:.2f} "
            f"[{row.mae_ci_low:.2f}, {row.mae_ci_high:.2f}] & "
            f"{row.rmse:.2f} & {row.bias:+.2f} & "
            f"{row.median_ae:.2f} & {row.q95_ae:.2f} \\\\"
        )
    text = """\\begin{tabular}{lrrrrr}
\\toprule
Model & MAE [95\\% CI] & RMSE & Bias & Median AE & 95th pct. AE \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
"""
    (OUT / "model_comparison_table.tex").write_text(text)


def write_dm_table() -> None:
    frame = pd.read_csv(DATA / "dm_tests.csv")
    labels = {
        "naive_w": "Weekly naive",
        "ridge": "pooled Ridge",
        "ridge_hourly": "hourly Ridge",
    }
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"LightGBM vs. {labels[row.comparator]} & "
            f"{row.mean_loss_improvement:.2f} & {row.hac_standard_error:.2f} & "
            f"{row.hln_statistic:.2f} & {sci(row.p_value)} \\\\"
        )
    text = """\\begin{tabular}{lrrrr}
\\toprule
Comparison & Mean daily MAE gain & HAC SE & HLN statistic & Two-sided $p$ \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
"""
    (OUT / "dm_table.tex").write_text(text)


def write_ablation_table() -> None:
    frame = pd.read_csv(DATA / "ablation_metrics.csv").sort_values("mae")
    labels = {
        "weather_augmented": "Weather-augmented sensitivity",
        "primary": "Primary D-1 specification",
        "fundamentals_calendar": "Renewables + calendar",
        "no_renewables": "Price history + calendar",
        "calendar_only": "Calendar only",
    }
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"{labels[row.specification]} & {row.features} & {row.mae:.2f} & "
            f"{row.rmse:.2f} & {row.bias:+.2f} & "
            f"{row.mae_change_vs_primary_pct:+.1f}\\% \\\\"
        )
    text = """\\begin{tabular}{lrrrrr}
\\toprule
Specification & Features & MAE & RMSE & Bias & MAE vs. primary \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
"""
    (OUT / "ablation_table.tex").write_text(text)


def write_regime_tables() -> None:
    frame = pd.read_csv(DATA / "regime_metrics.csv")
    primary = frame[frame["model"] == "lgbm"]
    labels = {
        "winter": "Winter", "spring": "Spring", "summer": "Summer",
        "autumn": "Autumn", "night": "Night (00--06)",
        "daylight": "Daylight (07--15)", "evening_ramp": "Evening ramp (16--21)",
        "late_evening": "Late evening (22--23)",
    }
    rows = []
    for dimension in ("season", "hour_block"):
        for row in primary[primary["dimension"] == dimension].itertuples():
            rows.append(
                f"{dimension.replace('_', ' ').title()} & "
                f"{labels.get(row.level, row.level)} & {row.observations:,} & "
                f"{row.mae:.2f} & {row.rmse:.2f} & {row.bias:+.2f} \\\\"
            )
    text = """\\begin{tabular}{llrrrr}
\\toprule
Dimension & Regime & $n$ & MAE & RMSE & Bias \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
"""
    (OUT / "regime_table.tex").write_text(text)


def write_tail_table() -> None:
    frame = pd.read_csv(DATA / "tail_metrics.csv")
    frame = frame[frame["regime"].isin(["negative", "regular", "above_200"])]
    labels = {
        "naive_w": "Weekly naive", "ridge": "Pooled Ridge",
        "ridge_hourly": "Hourly Ridge", "lgbm": "LightGBM",
        "negative": "Price $<0$", "regular": "$0\\leq$ price $<200$",
        "above_200": "Price $\\geq 200$",
    }
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"{labels[row.regime]} & {labels[row.model]} & {row.observations:,} & "
            f"{row.mae:.2f} & {row.bias:+.2f} \\\\"
        )
    text = """\\begin{tabular}{llrrr}
\\toprule
Price regime (EUR/MWh) & Model & $n$ & MAE & Bias \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
"""
    (OUT / "tail_table.tex").write_text(text)


def write_signal_table() -> None:
    frame = pd.read_csv(DATA / "signal_sensitivity.csv")
    frame = frame[frame["prompt_window"] == 7]
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"{row.threshold:.2f} & {row.evaluable_active_days} & "
            f"{100 * row.hit_rate:.1f}\\% & {row.average_captured:.2f} & "
            f"{row.median_captured:.2f} & {row.long_days}/{row.short_days} \\\\"
        )
    text = """\\begin{tabular}{rrrrrr}
\\toprule
$|z|$ threshold & Active days & Hit rate & Mean spread & Median spread & Long/short \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
"""
    (OUT / "signal_table.tex").write_text(text)


def write_signal_benchmark_table() -> None:
    frame = pd.read_csv(DATA / "signal_benchmarks.csv")
    labels = {
        "always_long": "Always long (no model signal)",
        "weekly_naive": "Weekly naive fair value",
        "ridge": "Pooled Ridge fair value",
        "ridge_hourly": "Hourly Ridge fair value",
        "lightgbm": "LightGBM fair value",
        "perfect_day_d": "Perfect day-D baseload (infeasible)",
    }
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"{labels[row.input]} & {'yes' if row.feasible else 'no'} & "
            f"{row.evaluable_active_days} & {100 * row.hit_rate:.1f}\\% & "
            f"{row.average_captured:.2f} \\\\"
        )
    text = """\\begin{tabular}{lcrrr}
\\toprule
Signal input & Feasible & Active days & Hit rate & Mean spread \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
"""
    (OUT / "signal_benchmark_table.tex").write_text(text)


def main() -> None:
    write_macros()
    write_model_table()
    write_dm_table()
    write_ablation_table()
    write_regime_tables()
    write_tail_table()
    write_signal_table()
    write_signal_benchmark_table()
    print("paper artifacts:", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
