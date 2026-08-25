"""Plot the battery-size and SoC-cycling NPV comparison."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "outputs"
COMPARISON_PATH = OUTPUT_ROOT / "soc_cycling_npv_comparison.csv"
DAILY_NPV_PATH = OUTPUT_ROOT / "daily_cycling_battery_npv_results.csv"
FIGURE_PATH = OUTPUT_ROOT / "battery_npv_by_soc_cycling_period.png"
DAILY_FIGURE_PATH = OUTPUT_ROOT / "daily_cycling_battery_npv_sensitivity.png"

POST_PROCESSING_ROOT = HERE.parent
sys.path.insert(0, str(POST_PROCESSING_ROOT))
sys.path.insert(0, str(HERE))

from compare_results import build_comparison  # noqa: E402
from common.plotting_style import get_palette  # noqa: E402
from config import C_PPA, LAMBDA_PPA, PALETTE_NAME  # noqa: E402


def label_bars(axis, bars, values, y_span):
    offset = 0.018 * y_span
    for bar, value in zip(bars, values):
        vertical_alignment = "bottom" if value >= 0.0 else "top"
        y = value + offset if value >= 0.0 else value - offset
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{value:.2f}",
            ha="center",
            va=vertical_alignment,
            rotation=55,
            fontsize=7.6,
            color="black",
        )


def plot_comparison() -> Path:
    if COMPARISON_PATH.exists():
        data = pd.read_csv(COMPARISON_PATH)
    else:
        data = build_comparison()

    if "No battery, 3-day" not in set(data["soc_cycle_label"]):
        no_battery = data.loc[
            data["soc_cycle_label"].eq("No battery, daily")
        ].copy()
        if len(no_battery) != 1:
            raise ValueError("Expected one daily no-battery comparison row.")
        no_battery["soc_cycle_hours"] = 72
        no_battery["soc_cycle_label"] = "No battery, 3-day"
        data = pd.concat([data, no_battery], ignore_index=True)

    expected_labels = {
        "No battery, daily",
        "No battery, 3-day",
        "10 MW/30 MWh, daily",
        "10 MW/30 MWh, 3-day",
        "20 MW/60 MWh, daily",
        "20 MW/60 MWh, 3-day",
    }
    if set(data["soc_cycle_label"]) != expected_labels:
        raise ValueError("The SoC-cycling comparison contains unexpected cases.")

    def npv_for(label: str) -> float:
        rows = data.loc[data["soc_cycle_label"].eq(label), "npv_meur"]
        if len(rows) != 1:
            raise ValueError(f"Expected one NPV result for {label}.")
        return float(rows.iloc[0])

    battery_labels = ["No battery", "10 MW/30 MWh", "20 MW/60 MWh"]
    daily_values = [
        npv_for("No battery, daily"),
        npv_for("10 MW/30 MWh, daily"),
        npv_for("20 MW/60 MWh, daily"),
    ]
    three_day_values = [
        npv_for("No battery, 3-day"),
        npv_for("10 MW/30 MWh, 3-day"),
        npv_for("20 MW/60 MWh, 3-day"),
    ]

    palette = get_palette(PALETTE_NAME)
    x_positions = np.arange(len(battery_labels), dtype=float)
    group_width = 0.54
    bar_width = group_width / 2.0
    daily_positions = x_positions - bar_width / 2.0
    three_day_positions = x_positions + bar_width / 2.0
    all_values = np.asarray(daily_values + three_day_values, dtype=float)
    y_min = min(float(all_values.min()), 0.0)
    y_max = max(float(all_values.max()), 0.1)
    y_span = y_max - y_min

    fig, axis = plt.subplots(figsize=(10.2, 5.8))
    daily_bars = axis.bar(
        daily_positions,
        daily_values,
        width=bar_width * 0.78,
        color=palette[0],
        edgecolor="white",
        linewidth=0.4,
        label="Daily SoC cycling",
    )
    three_day_bars = axis.bar(
        three_day_positions,
        three_day_values,
        width=bar_width * 0.78,
        color=palette[2],
        edgecolor="white",
        linewidth=0.4,
        label="Three-day SoC cycling",
    )

    label_bars(axis, daily_bars, daily_values, y_span)
    label_bars(axis, three_day_bars, three_day_values, y_span)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_positions, battery_labels)
    axis.set_ylim(y_min - 0.13 * y_span, y_max + 0.08 * y_span)
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=8.5)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    fig.suptitle(
        "NPV sensitivity to the battery SoC cycling period\n"
        f"Generic Industrial, hourly PPA deviation settlement, "
        f"$\\lambda^{{PPA}}={LAMBDA_PPA:g}$ EUR/MWh, "
        f"$c^{{PPA}}={C_PPA:g}$ EUR/MWh",
        y=0.995,
        fontsize=12,
    )
    fig.supylabel("NPV [MEUR]", x=0.025, fontsize=11.5)
    fig.subplots_adjust(top=0.76, bottom=0.12, left=0.10, right=0.98)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {FIGURE_PATH}")
    return FIGURE_PATH


def plot_daily_battery_npv() -> Path:
    if not DAILY_NPV_PATH.exists():
        build_comparison()
    data = pd.read_csv(DAILY_NPV_PATH)

    penalties = sorted(data["c_ppa"].unique())
    prices = sorted(data["lambda_PPA"].unique())
    batteries = [
        ("No battery", "No battery"),
        ("10 MW/30 MWh", "10 MW/30 MWh"),
        ("20 MW/60 MWh", "20 MW/60 MWh"),
    ]
    palette = get_palette(PALETTE_NAME)
    values = data["npv_meur"].to_numpy(dtype=float)
    y_min = min(float(values.min()), 0.0)
    y_max = max(float(values.max()), 0.1)
    y_span = y_max - y_min

    fig = plt.figure(figsize=(12.8, 8.6))
    grid = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.42)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 1:3]),
    ]
    x_positions = np.arange(len(prices), dtype=float)
    group_width = 0.76
    bar_width = group_width / len(batteries)

    for panel, (axis, penalty) in enumerate(zip(axes, penalties)):
        panel_data = data.loc[data["c_ppa"].eq(penalty)]
        for battery_index, (source_label, display_label) in enumerate(batteries):
            values = []
            for price in prices:
                rows = panel_data.loc[
                    panel_data["battery_label"].eq(source_label)
                    & panel_data["lambda_PPA"].eq(price),
                    "npv_meur",
                ]
                if len(rows) != 1:
                    raise ValueError(
                        f"Expected one daily NPV row for {source_label}, "
                        f"lambda={price:g}, c={penalty:g}."
                    )
                values.append(float(rows.iloc[0]))
            offset = (
                battery_index - (len(batteries) - 1) / 2
            ) * bar_width
            bars = axis.bar(
                x_positions + offset,
                values,
                width=bar_width * 0.92,
                color=palette[battery_index],
                edgecolor="white",
                linewidth=0.4,
                label=display_label,
            )
            label_bars(axis, bars, values, y_span)

        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(
            f"({chr(97 + panel)}) $c^{{PPA}}={penalty:g}$ EUR/MWh",
            fontsize=10.5,
        )
        axis.set_xticks(x_positions, [f"{price:g}" for price in prices])
        axis.set_ylim(y_min - 0.19 * y_span, y_max + 0.17 * y_span)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", labelsize=8.5)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.925),
        fontsize=10.5,
    )
    fig.suptitle(
        "Project NPV by battery configuration\n"
        "Generic Industrial, hourly PPA deviation settlement, daily SoC cycling",
        y=0.995,
        fontsize=12,
    )
    fig.supxlabel(
        r"PPA price, $\lambda^{PPA}$ [EUR/MWh]",
        y=0.03,
        fontsize=11.5,
    )
    fig.supylabel("NPV [MEUR]", x=0.025, fontsize=11.5)
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.085, right=0.98)
    fig.savefig(DAILY_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {DAILY_FIGURE_PATH}")
    return DAILY_FIGURE_PATH


def main() -> None:
    plot_daily_battery_npv()
    plot_comparison()


if __name__ == "__main__":
    main()
