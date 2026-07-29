#!/usr/bin/env python3
"""Generate publication figures, LaTeX tables, and result macros from CSV files."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"
FIGURES = ROOT / "paper" / "figures"
GENERATED = ROOT / "paper" / "generated"


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def variance_figure() -> None:
    data = rows("variance_reduction.csv")
    labels = [row["method"] for row in data]
    ratios = [float(row["variance_ratio"]) for row in data]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    bars = ax.bar(labels, ratios, color=["#6B7280", "#2563EB", "#059669"])
    ax.set_ylabel("Variance ratio relative to plain Monte Carlo")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ratio + 0.025,
            f"{ratio:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure("variance_reduction")


def lsm_figure() -> None:
    data = rows("lsm_convergence.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    styles = {
        ("7", "Linear"): ("#2563EB", "o", "-"),
        ("7", "Quadratic"): ("#2563EB", "s", "--"),
        ("42", "Linear"): ("#DC2626", "o", "-"),
        ("42", "Quadratic"): ("#DC2626", "s", "--"),
    }
    for key, (color, marker, line) in styles.items():
        selected = [row for row in data if (row["seed"], row["basis"]) == key]
        x = [int(row["paths"]) for row in selected]
        y = [float(row["price"]) for row in selected]
        e = [1.96 * float(row["standard_error"]) for row in selected]
        ax.errorbar(
            x,
            y,
            yerr=e,
            color=color,
            marker=marker,
            linestyle=line,
            capsize=3,
            label=f"seed {key[0]}, {key[1].lower()}",
        )
    ax.set_xscale("log")
    ax.set_xticks([2000, 5000, 10000], labels=["2k", "5k", "10k"])
    ax.set_xlabel("Simulated paths")
    ax.set_ylabel("Bermudan swaption price")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_figure("lsm_convergence")


def monte_carlo_figure() -> None:
    data = rows("g2pp_monte_carlo_convergence.csv")
    deterministic = float(data[0]["deterministic"])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for seed, color, marker in [("7", "#2563EB", "o"), ("42", "#DC2626", "s")]:
        selected = [row for row in data if row["seed"] == seed]
        x = [int(row["paths"]) for row in selected]
        y = [float(row["price"]) for row in selected]
        e = [1.96 * float(row["standard_error"]) for row in selected]
        ax.errorbar(
            x,
            y,
            yerr=e,
            marker=marker,
            color=color,
            capsize=3,
            label=f"Monte Carlo, seed {seed}",
        )
    ax.axhline(
        deterministic,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label="deterministic quadrature",
    )
    ax.set_xscale("log")
    ax.set_xticks([2000, 5000, 10000, 30000], labels=["2k", "5k", "10k", "30k"])
    ax.set_xlabel("Simulated paths")
    ax.set_ylabel("European swaption price")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure("g2pp_mc_convergence")


def monte_carlo_standardized_error_figure() -> None:
    data = rows("g2pp_monte_carlo_convergence.csv")
    labels = [f"{int(row['paths']) // 1000}k/s{row['seed']}" for row in data]
    standardized = [
        (float(row["price"]) - float(row["deterministic"]))
        / float(row["standard_error"])
        for row in data
    ]
    colors = ["#2563EB" if row["seed"] == "7" else "#DC2626" for row in data]
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.bar(range(len(data)), standardized, color=colors)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.axhline(1.96, color="#6B7280", linestyle="--", linewidth=1.0)
    ax.axhline(-1.96, color="#6B7280", linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(data)), labels=labels, rotation=30, ha="right")
    ax.set_ylabel("Standardized pricing error")
    ax.set_xlabel("Path count / random seed")
    ax.set_ylim(-2.5, 2.5)
    ax.grid(axis="y", alpha=0.25)
    save_figure("g2pp_mc_standardized_error")


def validation_figure() -> None:
    data = rows("quantlib_validation.csv")
    selected = [
        row
        for row in data
        if row["quantity"] in {"Black-76 ATM", "G2++ European swaption"}
    ]
    labels = [row["quantity"] for row in selected]
    qf = [float(row["qf_rates"]) for row in selected]
    ql = [float(row["quantlib"]) for row in selected]
    # Normalize each pair to the QuantLib value so unlike units remain comparable.
    qf_rel = [a / b for a, b in zip(qf, ql)]
    x = list(range(len(labels)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.bar([v - width / 2 for v in x], qf_rel, width, label="qf-rates", color="#2563EB")
    ax.bar([v + width / 2 for v in x], [1.0] * len(x), width, label="QuantLib", color="#9CA3AF")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Price normalized by QuantLib")
    ax.set_ylim(0.985, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    save_figure("quantlib_comparison")


def quantlib_grid_validation_figure() -> None:
    data = rows("quantlib_g2pp_grid.csv")
    material = [row for row in data if float(row["quantlib"]) >= 1.0e-4]
    scenarios = [
        "low_volatility",
        "base",
        "high_volatility",
        "fast_mean_reversion",
        "weak_correlation",
    ]
    colors = {
        "low_volatility": "#2563EB",
        "base": "#111827",
        "high_volatility": "#DC2626",
        "fast_mean_reversion": "#059669",
        "weak_correlation": "#D97706",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for scenario in scenarios:
        selected = [row for row in material if row["scenario"] == scenario]
        axes[0].scatter(
            [1.0e4 * float(row["quantlib"]) for row in selected],
            [1.0e4 * float(row["qf_rates"]) for row in selected],
            s=20,
            alpha=0.8,
            color=colors[scenario],
            label=scenario.replace("_", " "),
        )
    all_prices = [
        1.0e4 * float(row[field])
        for row in material
        for field in ("quantlib", "qf_rates")
    ]
    lower = min(all_prices)
    upper = max(all_prices)
    axes[0].plot([lower, upper], [lower, upper], color="#6B7280", linestyle="--")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("QuantLib price (bp of notional)")
    axes[0].set_ylabel("qf-rates price (bp of notional)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    distributions = [
        [
            100.0 * float(row["relative_difference"])
            for row in material
            if row["scenario"] == scenario
        ]
        for scenario in scenarios
    ]
    boxes = axes[1].boxplot(
        distributions,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#111827"},
    )
    for patch, scenario in zip(boxes["boxes"], scenarios):
        patch.set_facecolor(colors[scenario])
        patch.set_alpha(0.7)
    axes[1].set_xticks(
        range(1, len(scenarios) + 1),
        [value.replace("_", "\n") for value in scenarios],
        fontsize=7,
    )
    axes[1].set_ylabel("Absolute relative price difference (%)")
    axes[1].grid(axis="y", alpha=0.25)
    save_figure("quantlib_g2pp_grid")


def risk_validation_figure() -> None:
    data = rows("g2pp_risk_validation.csv")
    labels = [row["case"].replace("_", "\n") for row in data]
    x = list(range(len(data)))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
    for axis, qf_field, ql_field, label in [
        (
            axes[0],
            "qf_curve_dv01",
            "quantlib_curve_dv01",
            r"Curve DV01 ($10^{-4}$ price units)",
        ),
        (
            axes[1],
            "qf_joint_volatility_vega",
            "quantlib_joint_volatility_vega",
            r"Joint model-volatility bump ($10^{-4}$ price units)",
        ),
    ]:
        qf_values = [1.0e4 * float(row[qf_field]) for row in data]
        ql_values = [1.0e4 * float(row[ql_field]) for row in data]
        axis.bar(
            [value - width / 2 for value in x],
            qf_values,
            width,
            label="qf-rates",
            color="#2563EB",
        )
        axis.bar(
            [value + width / 2 for value in x],
            ql_values,
            width,
            label="QuantLib",
            color="#9CA3AF",
        )
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    axes[1].set_xticks(x, labels, fontsize=8)
    save_figure("g2pp_risk_validation")


def time_convergence_figure() -> None:
    data = rows("g2pp_time_step_convergence.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for seed, color, marker in [("7", "#2563EB", "o"), ("42", "#DC2626", "s")]:
        selected = [row for row in data if row["seed"] == seed]
        x = [int(row["time_steps"]) for row in selected]
        y = [1.0e6 * float(row["paired_bias_vs_finest"]) for row in selected]
        e = [
            1.96 * 1.0e6 * float(row["paired_bias_standard_error"])
            for row in selected
        ]
        ax.errorbar(
            x,
            y,
            yerr=e,
            color=color,
            marker=marker,
            capsize=3,
            label=f"seed {seed}",
        )
    ax.axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([12, 24, 48, 96, 192, 384], labels=["12", "24", "48", "96", "192", "384"])
    ax.set_xlabel("Discount-integration time steps")
    ax.set_ylabel(r"Paired bias vs 384 steps ($10^{-6}$ price units)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure("g2pp_time_step_convergence")


def stress_grid_figure() -> None:
    data = rows("g2pp_stress_grid.csv")
    scenarios = [
        "low_volatility",
        "base",
        "high_volatility",
        "fast_mean_reversion",
        "weak_correlation",
    ]
    expiries = [1.0, 2.0, 5.0]
    tenors = [2.0, 5.0, 10.0]
    atm = [
        row
        for row in data
        if abs(float(row["moneyness_basis_points"])) < 1.0e-12
    ]
    values = [1.0e4 * float(row["price"]) for row in atm]
    lower, upper = min(values), max(values)
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.5), constrained_layout=True)
    image = None
    for axis, scenario in zip(axes.flat, scenarios):
        matrix = []
        for expiry in expiries:
            matrix.append(
                [
                    1.0e4
                    * float(
                        next(
                            row["price"]
                            for row in atm
                            if row["scenario"] == scenario
                            and float(row["expiry"]) == expiry
                            and float(row["tenor"]) == tenor
                        )
                    )
                    for tenor in tenors
                ]
            )
        image = axis.imshow(matrix, cmap="viridis", vmin=lower, vmax=upper, aspect="auto")
        axis.set_title(scenario.replace("_", " "), fontsize=9)
        axis.set_xticks(range(3), labels=["2Y", "5Y", "10Y"])
        axis.set_yticks(range(3), labels=["1Y", "2Y", "5Y"])
        axis.set_xlabel("Tenor")
        axis.set_ylabel("Expiry")
        for row_index, row_values in enumerate(matrix):
            for column_index, value in enumerate(row_values):
                color = "white" if value > 0.58 * upper else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
    axes.flat[-1].axis("off")
    if image is not None:
        fig.colorbar(image, ax=list(axes.flat), shrink=0.82, label="ATM price (bp of notional)")
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / "g2pp_stress_grid.pdf", bbox_inches="tight")
    plt.savefig(FIGURES / "g2pp_stress_grid.png", dpi=180, bbox_inches="tight")
    plt.close()


def stress_moneyness_figure() -> None:
    data = rows("g2pp_stress_grid.csv")
    scenarios = [
        ("low_volatility", "#64748B", "o"),
        ("base", "#2563EB", "s"),
        ("high_volatility", "#DC2626", "^"),
        ("fast_mean_reversion", "#059669", "D"),
        ("weak_correlation", "#7C3AED", "v"),
    ]
    moneyness = [-100.0, 0.0, 100.0]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for scenario, color, marker in scenarios:
        means = []
        for strike_shift in moneyness:
            selected = [
                1.0e4 * float(row["price"])
                for row in data
                if row["scenario"] == scenario
                and float(row["moneyness_basis_points"]) == strike_shift
            ]
            means.append(sum(selected) / len(selected))
        ax.plot(
            moneyness,
            means,
            color=color,
            marker=marker,
            linewidth=1.6,
            label=scenario.replace("_", " "),
        )
    ax.set_xticks(moneyness, labels=["-100 bp", "ATM", "+100 bp"])
    ax.set_xlabel("Strike relative to par")
    ax.set_ylabel("Mean price across expiry-tenor grid (bp of notional)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_figure("g2pp_stress_moneyness")


def multistart_figure() -> None:
    data = rows("g2pp_multistart_calibration.csv")
    runs = [int(row["run"]) for row in data]
    rmse_basis_points = [1.0e4 * float(row["rmse"]) for row in data]
    colors = [
        "#059669" if row["selected"] == "1" else "#6B7280" if row["run"] == "0" else "#2563EB"
        for row in data
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bars = ax.bar(runs, rmse_basis_points, color=colors)
    ax.axhline(rmse_basis_points[0], color="#6B7280", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Deterministic start index")
    ax.set_ylabel("Normal-volatility RMSE (bp)")
    ax.set_xticks(runs)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, rmse_basis_points):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    save_figure("g2pp_multistart_calibration")


def multistart_parameter_figure() -> None:
    data = rows("g2pp_multistart_calibration.csv")
    parameters = ["a", "b", "sigma", "eta", "rho"]
    lower = {"a": 0.005, "b": 0.01, "sigma": 0.0001, "eta": 0.0001, "rho": -0.95}
    upper = {"a": 1.0, "b": 1.5, "sigma": 0.1, "eta": 0.1, "rho": 0.95}
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for row in data:
        normalized = [
            (float(row[f"calibrated_{parameter}"]) - lower[parameter])
            / (upper[parameter] - lower[parameter])
            for parameter in parameters
        ]
        selected = row["selected"] == "1"
        default = row["run"] == "0"
        ax.plot(
            range(len(parameters)),
            normalized,
            color="#059669" if selected else "#6B7280" if default else "#2563EB",
            linewidth=2.5 if selected else 1.0,
            alpha=1.0 if selected else 0.55,
            marker="o" if selected else None,
            label=(
                f"selected run {row['run']}"
                if selected
                else "default run"
                if default
                else None
            ),
        )
    ax.set_xticks(range(len(parameters)), labels=[r"$a$", r"$b$", r"$\sigma$", r"$\eta$", r"$\rho$"])
    ax.set_ylabel("Final parameter normalized to declared bounds")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8)
    save_figure("g2pp_multistart_parameters")


def calibration_residual_figure() -> None:
    data = rows("g2pp_calibration_residuals.csv")
    labels = [f"{row['expiry']}Yx{row['tenor']}Y" for row in data]
    market = [1.0e4 * float(row["market_volatility"]) for row in data]
    model = [1.0e4 * float(row["model_volatility"]) for row in data]
    errors = [float(row["error_basis_points"]) for row in data]
    x = list(range(len(data)))
    width = 0.34
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    axes[0].bar([value - width / 2 for value in x], market, width, label="market", color="#9CA3AF")
    axes[0].bar([value + width / 2 for value in x], model, width, label="model", color="#2563EB")
    axes[0].set_ylabel("Normal volatility (bp)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].bar(x, errors, color=["#059669" if value >= 0.0 else "#DC2626" for value in errors])
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_ylabel("Model minus market (bp)")
    axes[1].set_xticks(x, labels=labels, rotation=25, ha="right")
    axes[1].set_xlabel("Expiry x underlying tenor")
    axes[1].grid(axis="y", alpha=0.25)
    save_figure("g2pp_calibration_residuals")


def lsm_out_of_sample_figure() -> None:
    data = rows("lsm_out_of_sample.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    styles = {
        ("7", "Linear"): ("#2563EB", "o", "-"),
        ("7", "Quadratic"): ("#2563EB", "s", "--"),
        ("42", "Linear"): ("#DC2626", "o", "-"),
        ("42", "Quadratic"): ("#DC2626", "s", "--"),
    }
    for key, (color, marker, line) in styles.items():
        selected = [
            row
            for row in data
            if (row["training_seed"], row["basis"]) == key
        ]
        x = [int(row["training_paths"]) for row in selected]
        y = [float(row["optimism"]) for row in selected]
        e = [1.96 * float(row["standard_error"]) for row in selected]
        ax.errorbar(
            x,
            y,
            yerr=e,
            color=color,
            marker=marker,
            linestyle=line,
            capsize=3,
            label=f"seed {key[0]}, {key[1].lower()}",
        )
    ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xticks([2000, 5000, 10000], labels=["2k", "5k", "10k"])
    ax.set_xlabel("Policy-training paths")
    ax.set_ylabel("Training price minus independent valuation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_figure("lsm_out_of_sample")


def lsm_exercise_distribution_figure() -> None:
    data = [
        row
        for row in rows("lsm_out_of_sample.csv")
        if int(row["training_paths"]) == 10000
    ]
    labels = [f"s{row['training_seed']} {row['basis'].lower()}" for row in data]
    components = [
        ("exercise_probability_1y", "exercise 1Y", "#2563EB"),
        ("exercise_probability_2y", "exercise 2Y", "#60A5FA"),
        ("exercise_probability_3y", "exercise 3Y", "#A78BFA"),
        ("non_exercise_probability", "no exercise", "#9CA3AF"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bottoms = [0.0] * len(data)
    for field, label, color in components:
        values = [float(row[field]) for row in data]
        ax.bar(range(len(data)), values, bottom=bottoms, label=label, color=color)
        bottoms = [left + value for left, value in zip(bottoms, values)]
    ax.set_xticks(range(len(data)), labels=labels)
    ax.set_ylabel("Independent-path probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_figure("lsm_exercise_distribution")


def wrong_way_risk_figure() -> None:
    data = rows("wrong_way_risk_grid.csv")
    betas = [float(row["beta"]) for row in data]
    cvas = [float(row["cva"]) for row in data]
    exposure_fields = [field for field in data[0] if field.startswith("epe_")]
    times = [float(field.removeprefix("epe_").removesuffix("y")) for field in exposure_fields]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].plot(betas, cvas, color="#DC2626", marker="o")
    axes[0].set_xlabel(r"Proxy sensitivity $\beta$")
    axes[0].set_ylabel("CVA")
    axes[0].grid(alpha=0.25)
    for beta in [0.0, 10.0, 20.0, 30.0]:
        row = next(candidate for candidate in data if float(candidate["beta"]) == beta)
        exposures = [float(row[field]) for field in exposure_fields]
        axes[1].plot(times, exposures, marker="o", label=rf"$\beta={beta:.0f}$")
    axes[1].set_xlabel("Exposure time (years)")
    axes[1].set_ylabel("Expected positive exposure")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    save_figure("wrong_way_risk_sensitivity")


def latex_escape(value: str) -> str:
    return value.replace("++", r"\texttt{++}").replace("%", r"\%")


def generated_tex() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    variance = rows("variance_reduction.csv")
    mc = rows("g2pp_monte_carlo_convergence.csv")
    wwr = rows("wrong_way_risk.csv")
    ql = rows("quantlib_validation.csv")
    ql_grid = rows("quantlib_g2pp_grid.csv")
    risk_validation = rows("g2pp_risk_validation.csv")
    time_convergence = rows("g2pp_time_step_convergence.csv")
    stress = rows("g2pp_stress_grid.csv")
    multistart = rows("g2pp_multistart_calibration.csv")
    calibration_residuals = rows("g2pp_calibration_residuals.csv")
    out_of_sample = rows("lsm_out_of_sample.csv")
    wrong_way_grid = rows("wrong_way_risk_grid.csv")

    control_ratio = float(
        next(row["variance_ratio"] for row in variance if row["method"] == "Control variate")
    )
    anti_ratio = float(
        next(row["variance_ratio"] for row in variance if row["method"] == "Antithetic")
    )
    largest_mc = max(mc, key=lambda row: int(row["paths"]))
    g2_row = next(row for row in ql if row["quantity"] == "G2++ European swaption")
    independent = float(wwr[0]["cva"])
    wrong_way = float(wwr[1]["cva"])
    non_finest_time_rows = [
        row
        for row in time_convergence
        if int(row["time_steps"])
        < max(int(candidate["time_steps"]) for candidate in time_convergence)
    ]
    max_paired_bias = max(
        abs(float(row["paired_bias_vs_finest"])) for row in non_finest_time_rows
    )
    default_rmse = float(multistart[0]["rmse"])
    selected_run = next(row for row in multistart if row["selected"] == "1")
    best_rmse = float(selected_run["rmse"])
    valuation_prices = [float(row["valuation_price"]) for row in out_of_sample]
    max_oos_gap = max(abs(float(row["optimism"])) for row in out_of_sample)
    standardized_mc_errors = [
        (float(row["price"]) - float(row["deterministic"]))
        / float(row["standard_error"])
        for row in mc
    ]
    covered_mc = sum(abs(value) <= 1.96 for value in standardized_mc_errors)
    stress_prices = [float(row["price"]) for row in stress]
    max_calibration_error = max(
        abs(float(row["error_basis_points"])) for row in calibration_residuals
    )
    beta_thirty = next(row for row in wrong_way_grid if float(row["beta"]) == 30.0)
    beta_thirty_cva = float(beta_thirty["cva"])
    exercise_one_year = [
        float(row["exercise_probability_1y"]) for row in out_of_sample
    ]
    non_exercise = [float(row["non_exercise_probability"]) for row in out_of_sample]
    material_grid = [row for row in ql_grid if float(row["quantlib"]) >= 1.0e-4]
    material_relative_errors = [
        float(row["relative_difference"]) for row in material_grid
    ]
    max_grid_difference_bp = max(
        float(row["difference_basis_points_notional"]) for row in ql_grid
    )
    max_dv01_difference = max(
        float(row["curve_dv01_relative_difference"]) for row in risk_validation
    )
    max_volatility_difference = max(
        float(row["volatility_vega_relative_difference"]) for row in risk_validation
    )
    macros = rf"""\newcommand{{\ControlVarianceRatio}}{{{control_ratio:.3f}}}
\newcommand{{\ControlVarianceReduction}}{{{(1.0 - control_ratio) * 100.0:.1f}\%}}
\newcommand{{\AntitheticVarianceRatio}}{{{anti_ratio:.3f}}}
\newcommand{{\DeterministicPrice}}{{{float(largest_mc["deterministic"]):.8f}}}
\newcommand{{\LargestMCPrice}}{{{float(largest_mc["price"]):.8f}}}
\newcommand{{\LargestMCSE}}{{{float(largest_mc["standard_error"]):.8f}}}
\newcommand{{\QuantLibGTwoRelativeError}}{{{float(g2_row["relative_difference"]) * 100.0:.3f}\%}}
\newcommand{{\IndependentCVA}}{{{independent:.2f}}}
\newcommand{{\WrongWayCVA}}{{{wrong_way:.2f}}}
\newcommand{{\WrongWayIncrease}}{{{(wrong_way / independent - 1.0) * 100.0:.1f}\%}}
\newcommand{{\TimeMaxPairedBias}}{{{max_paired_bias:.2e}}}
\newcommand{{\TimeMaxPairedBiasBp}}{{{max_paired_bias * 1.0e4:.4f}}}
\newcommand{{\StressGridCells}}{{{len(stress)}}}
\newcommand{{\MultiStartDefaultRMSE}}{{{default_rmse:.8f}}}
\newcommand{{\MultiStartBestRMSE}}{{{best_rmse:.8f}}}
\newcommand{{\MultiStartImprovement}}{{{(1.0 - best_rmse / default_rmse) * 100.0:.1f}\%}}
\newcommand{{\MultiStartBestRun}}{{{selected_run["run"]}}}
\newcommand{{\OOSPriceMinimum}}{{{min(valuation_prices):.8f}}}
\newcommand{{\OOSPriceMaximum}}{{{max(valuation_prices):.8f}}}
\newcommand{{\OOSMaxGap}}{{{max_oos_gap:.8f}}}
\newcommand{{\MCCoveredRuns}}{{{covered_mc}}}
\newcommand{{\MCTotalRuns}}{{{len(standardized_mc_errors)}}}
\newcommand{{\MCMaxAbsZ}}{{{max(abs(value) for value in standardized_mc_errors):.2f}}}
\newcommand{{\StressMinimumPriceBp}}{{{min(stress_prices) * 1.0e4:.6f}}}
\newcommand{{\StressMaximumPriceBp}}{{{max(stress_prices) * 1.0e4:.1f}}}
\newcommand{{\CalibrationMaxResidualBp}}{{{max_calibration_error:.3f}}}
\newcommand{{\OOSOneYearExerciseMinimum}}{{{min(exercise_one_year) * 100.0:.1f}\%}}
\newcommand{{\OOSOneYearExerciseMaximum}}{{{max(exercise_one_year) * 100.0:.1f}\%}}
\newcommand{{\OOSNonExerciseMinimum}}{{{min(non_exercise) * 100.0:.1f}\%}}
\newcommand{{\OOSNonExerciseMaximum}}{{{max(non_exercise) * 100.0:.1f}\%}}
\newcommand{{\WrongWayThirtyCVA}}{{{beta_thirty_cva:.2f}}}
\newcommand{{\WrongWayThirtyIncrease}}{{{(beta_thirty_cva / independent - 1.0) * 100.0:.1f}\%}}
\newcommand{{\QuantLibGridCells}}{{{len(ql_grid)}}}
\newcommand{{\QuantLibGridMaterialCells}}{{{len(material_grid)}}}
\newcommand{{\QuantLibGridMedianRelativeError}}{{{100.0 * statistics.median(material_relative_errors):.3f}\%}}
\newcommand{{\QuantLibGridNinetyFifthRelativeError}}{{{100.0 * percentile(material_relative_errors, 0.95):.3f}\%}}
\newcommand{{\QuantLibGridMaxRelativeError}}{{{100.0 * max(material_relative_errors):.3f}\%}}
\newcommand{{\QuantLibGridMaxDifferenceBp}}{{{max_grid_difference_bp:.3f}}}
\newcommand{{\RiskMaxDVOneDifference}}{{{100.0 * max_dv01_difference:.2f}\%}}
\newcommand{{\RiskMaxVolatilityDifference}}{{{100.0 * max_volatility_difference:.2f}\%}}
"""
    (GENERATED / "results_macros.tex").write_text(macros, encoding="utf-8")

    ql_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Quantity & \texttt{qf-rates} & QuantLib & Relative difference \\",
        r"\midrule",
    ]
    for row in ql:
        rel = float(row["relative_difference"])
        ql_lines.append(
            f"{latex_escape(row['quantity'])} & {float(row['qf_rates']):.8g} & "
            f"{float(row['quantlib']):.8g} & {rel * 100.0:.4f}\\% \\\\"
        )
    ql_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "quantlib_table.tex").write_text(
        "\n".join(ql_lines) + "\n", encoding="utf-8"
    )

    grid_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Regime & Cells & Material & Median rel. & 95th pct. rel. & Max abs. (bp) \\",
        r"\midrule",
    ]
    for scenario in [
        "low_volatility",
        "base",
        "high_volatility",
        "fast_mean_reversion",
        "weak_correlation",
    ]:
        selected = [row for row in ql_grid if row["scenario"] == scenario]
        selected_material = [
            row for row in selected if float(row["quantlib"]) >= 1.0e-4
        ]
        relative = [
            100.0 * float(row["relative_difference"]) for row in selected_material
        ]
        grid_lines.append(
            f"{scenario.replace('_', ' ')} & {len(selected)} & {len(selected_material)} & "
            f"{statistics.median(relative):.3f}\% & {percentile(relative, 0.95):.3f}\% & "
            f"{max(float(row['difference_basis_points_notional']) for row in selected):.3f} \\\\"
        )
    grid_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "quantlib_grid_summary_table.tex").write_text(
        "\n".join(grid_lines) + "\n", encoding="utf-8"
    )

    risk_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Case & QF DV01 & QL DV01 & Rel. & QF vol. bump & QL vol. bump & Rel. \\",
        r"\midrule",
    ]
    for row in risk_validation:
        risk_lines.append(
            f"{row['case'].replace('_', ' ')} & "
            f"{1.0e4 * float(row['qf_curve_dv01']):+.3f} & "
            f"{1.0e4 * float(row['quantlib_curve_dv01']):+.3f} & "
            f"{100.0 * float(row['curve_dv01_relative_difference']):.2f}\% & "
            f"{1.0e4 * float(row['qf_joint_volatility_vega']):.3f} & "
            f"{1.0e4 * float(row['quantlib_joint_volatility_vega']):.3f} & "
            f"{100.0 * float(row['volatility_vega_relative_difference']):.2f}\% \\\\"
        )
    risk_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "risk_validation_table.tex").write_text(
        "\n".join(risk_lines) + "\n", encoding="utf-8"
    )

    lsm = rows("lsm_convergence.csv")
    lsm_lines = [
        r"\begin{tabular}{rrlrr}",
        r"\toprule",
        r"Paths & Seed & Basis & Price & Standard error \\",
        r"\midrule",
    ]
    for row in lsm:
        lsm_lines.append(
            f"{int(row['paths']):,} & {row['seed']} & {row['basis']} & "
            f"{float(row['price']):.8f} & {float(row['standard_error']):.8f} \\\\"
        )
    lsm_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "lsm_table.tex").write_text(
        "\n".join(lsm_lines) + "\n", encoding="utf-8"
    )

    mc_lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Paths & Seed & Price & Standard error & Standardized error \\",
        r"\midrule",
    ]
    for row, standardized in zip(mc, standardized_mc_errors):
        mc_lines.append(
            f"{int(row['paths']):,} & {row['seed']} & {float(row['price']):.8f} & "
            f"{float(row['standard_error']):.8f} & {standardized:.3f} \\\\"
        )
    mc_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "mc_diagnostics_table.tex").write_text(
        "\n".join(mc_lines) + "\n", encoding="utf-8"
    )

    residual_lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Expiry & Tenor & Market vol. & Model vol. & Error (bp) \\",
        r"\midrule",
    ]
    for row in calibration_residuals:
        residual_lines.append(
            f"{float(row['expiry']):.0f}Y & {float(row['tenor']):.0f}Y & "
            f"{1.0e4 * float(row['market_volatility']):.3f} & "
            f"{1.0e4 * float(row['model_volatility']):.3f} & "
            f"{float(row['error_basis_points']):+.3f} \\\\"
        )
    residual_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "calibration_residuals_table.tex").write_text(
        "\n".join(residual_lines) + "\n", encoding="utf-8"
    )

    multistart_lines = [
        r"\begin{tabular}{rcrrrrrrr}",
        r"\toprule",
        r"Run & Selected & $a$ & $b$ & $\sigma$ & $\eta$ & $\rho$ & RMSE (bp) & Iter. \\",
        r"\midrule",
    ]
    for row in multistart:
        multistart_lines.append(
            f"{row['run']} & {'yes' if row['selected'] == '1' else 'no'} & "
            f"{float(row['calibrated_a']):.4f} & {float(row['calibrated_b']):.4f} & "
            f"{float(row['calibrated_sigma']):.4f} & {float(row['calibrated_eta']):.4f} & "
            f"{float(row['calibrated_rho']):.4f} & {1.0e4 * float(row['rmse']):.3f} & "
            f"{row['iterations']} \\\\"
        )
    multistart_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "multistart_table.tex").write_text(
        "\n".join(multistart_lines) + "\n", encoding="utf-8"
    )

    oos_lines = [
        r"\begin{tabular}{rrlrrrrr}",
        r"\toprule",
        r"Train paths & Seed & Basis & Train price & Test price & Gap & SE & No exercise \\",
        r"\midrule",
    ]
    for row in out_of_sample:
        oos_lines.append(
            f"{int(row['training_paths']):,} & {row['training_seed']} & {row['basis']} & "
            f"{float(row['training_price']):.6f} & {float(row['valuation_price']):.6f} & "
            f"{float(row['optimism']):+.6f} & {float(row['standard_error']):.6f} & "
            f"{100.0 * float(row['non_exercise_probability']):.1f}\\% \\\\"
        )
    oos_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "lsm_oos_table.tex").write_text(
        "\n".join(oos_lines) + "\n", encoding="utf-8"
    )

    wwr_lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$\beta$ & CVA & Change vs. $\beta=0$ & Peak EPE \\",
        r"\midrule",
    ]
    for row in wrong_way_grid:
        epe_values = [
            float(value) for field, value in row.items() if field.startswith("epe_")
        ]
        wwr_lines.append(
            f"{float(row['beta']):.0f} & {float(row['cva']):.2f} & "
            f"{(float(row['cva']) / independent - 1.0) * 100.0:+.1f}\\% & "
            f"{max(epe_values):.2f} \\\\"
        )
    wwr_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "wrong_way_grid_table.tex").write_text(
        "\n".join(wwr_lines) + "\n", encoding="utf-8"
    )

    stress_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Regime & Minimum & Median & Maximum \\",
        r"\midrule",
    ]
    for scenario in [
        "low_volatility",
        "base",
        "high_volatility",
        "fast_mean_reversion",
        "weak_correlation",
    ]:
        values = sorted(
            1.0e4 * float(row["price"])
            for row in stress
            if row["scenario"] == scenario
        )
        stress_lines.append(
            f"{scenario.replace('_', ' ')} & {min(values):.6f} & "
            f"{statistics.median(values):.3f} & {max(values):.3f} \\\\"
        )
    stress_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "stress_summary_table.tex").write_text(
        "\n".join(stress_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    variance_figure()
    lsm_figure()
    monte_carlo_figure()
    monte_carlo_standardized_error_figure()
    validation_figure()
    quantlib_grid_validation_figure()
    risk_validation_figure()
    time_convergence_figure()
    stress_grid_figure()
    stress_moneyness_figure()
    multistart_figure()
    multistart_parameter_figure()
    calibration_residual_figure()
    lsm_out_of_sample_figure()
    lsm_exercise_distribution_figure()
    wrong_way_risk_figure()
    generated_tex()
    print(f"Generated paper artifacts in {FIGURES} and {GENERATED}")


if __name__ == "__main__":
    main()
