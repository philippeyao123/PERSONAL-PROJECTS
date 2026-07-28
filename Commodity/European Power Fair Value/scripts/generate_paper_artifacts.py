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
    trading = json.loads((REPORTS / "trading_stats.json").read_text())
    comparison = pd.read_csv(DATA / "model_comparison.csv").set_index("model")
    dm = pd.read_csv(DATA / "dm_tests.csv").set_index("comparator")
    conformal = research["conformal"]
    ablation = pd.read_csv(DATA / "ablation_metrics.csv").set_index("specification")
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
        "NaiveMAE": f"{comparison.loc['naive_w', 'mae']:.2f}",
        "RidgeMAE": f"{comparison.loc['ridge', 'mae']:.2f}",
        "SkillNaive": (
            f"{100 * (1 - comparison.loc['lgbm', 'mae'] / comparison.loc['naive_w', 'mae']):.1f}\\%"
        ),
        "SkillRidge": (
            f"{100 * (1 - comparison.loc['lgbm', 'mae'] / comparison.loc['ridge', 'mae']):.1f}\\%"
        ),
        "DMNaiveP": sci(float(dm.loc["naive_w", "p_value"])),
        "DMRidgeP": sci(float(dm.loc["ridge", "p_value"])),
        "ConformalCoverage": f"{100 * conformal['empirical_coverage']:.2f}\\%",
        "ConformalWidth": f"{conformal['mean_width']:.2f}",
        "WeatherDelta": (
            f"{ablation.loc['weather_augmented', 'mae_change_vs_primary_pct']:.2f}\\%"
        ),
        "NoRenewablesDelta": (
            f"{ablation.loc['no_renewables', 'mae_change_vs_primary_pct']:.1f}\\%"
        ),
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
        "RightCensored": f"{trading['right_censored_active_days']}",
        "NegativeMAE": f"{tail_value('negative', 'lgbm', 'mae'):.2f}",
        "HighMAE": f"{tail_value('above_200', 'lgbm', 'mae'):.2f}",
        "HighRecall": (
            f"{100 * tail_value('above_200_detection', 'lgbm', 'mae'):.1f}\\%"
        ),
    }
    (OUT / "results_macros.tex").write_text(
        "".join(command(k, v) for k, v in values.items())
    )


def write_model_table() -> None:
    frame = pd.read_csv(DATA / "model_comparison.csv")
    labels = {"naive_w": "Weekly naive", "ridge": "Ridge", "lgbm": "LightGBM"}
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
    labels = {"naive_w": "Weekly naive", "ridge": "Ridge"}
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"LightGBM vs. {labels[row.comparator]} & "
            f"{row.mean_loss_improvement:.2f} & {row.hac_standard_error:.2f} & "
            f"{row.statistic:.2f} & {sci(row.p_value)} \\\\"
        )
    text = """\\begin{tabular}{lrrrr}
\\toprule
Comparison & Mean daily MAE gain & HAC SE & Statistic & Two-sided $p$ \\\\
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
        "naive_w": "Weekly naive", "ridge": "Ridge", "lgbm": "LightGBM",
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


def main() -> None:
    write_macros()
    write_model_table()
    write_dm_table()
    write_ablation_table()
    write_regime_tables()
    write_tail_table()
    write_signal_table()
    print("paper artifacts:", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
