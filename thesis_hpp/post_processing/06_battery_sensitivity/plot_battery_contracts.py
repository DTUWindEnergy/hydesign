"""Plot optimal contracted PPA energy across battery configurations."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
OUTPUT_ROOT = HERE / "outputs"

sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.plotting_style import get_palette  # noqa: E402
from analysis import battery_label, load_catalog, profile_slug, select_contract_cases
from config import (  # noqa: E402
    BATTERY_CASES,
    CONTRACT_FIGURE_METRIC,
    PALETTE_NAME,
    PROFILE,
    SETTLEMENT_PERIOD,
)


def metric_settings() -> tuple[str, str, str]:
    if CONTRACT_FIGURE_METRIC == "daily_energy":
        return (
            "contracted_energy_mwh_per_day",
            r"Optimal daily contracted PPA energy, "
            r"$E^{PPA,day,*}$ [MWh/day]",
            "daily_contracted_energy",
        )
    if CONTRACT_FIGURE_METRIC == "capacity":
        return (
            "S_PPA",
            r"Optimal contracted PPA capacity, $S^{PPA,*}$ [MW]",
            "contracted_capacity",
        )
    raise ValueError(
        "CONTRACT_FIGURE_METRIC must be 'daily_energy' or 'capacity'."
    )


def plot_contracts(data: pd.DataFrame) -> list[Path]:
    value_column, y_label, output_name = metric_settings()
    c_ppa_values = sorted(data["c_ppa"].unique())
    ppa_prices = sorted(data["lambda_PPA"].unique())
    if len(c_ppa_values) != 3:
        raise ValueError("The centered 2+1 layout requires three c_ppa values.")

    palette = get_palette(PALETTE_NAME)
    if len(palette) < len(BATTERY_CASES):
        raise ValueError(
            f"Palette '{PALETTE_NAME}' does not contain enough colors."
        )

    y_max = max(float(data[value_column].max()), 1.0)
    fig = plt.figure(figsize=(12.8, 8.6))
    grid = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.42)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 1:3]),
    ]

    x_positions = np.arange(len(ppa_prices), dtype=float)
    group_width = 0.76
    bar_width = group_width / len(BATTERY_CASES)

    for panel_index, (ax, c_ppa) in enumerate(zip(axes, c_ppa_values)):
        panel = data.loc[np.isclose(data["c_ppa"], c_ppa)]
        for battery_index, (power, energy) in enumerate(BATTERY_CASES):
            battery_data = (
                panel.loc[
                    np.isclose(panel["P_b_max"], power)
                    & np.isclose(panel["E_max"], energy)
                ]
                .set_index("lambda_PPA")
                .reindex(ppa_prices)
            )
            if battery_data[value_column].isna().any():
                raise ValueError(
                    f"Incomplete data for c_ppa={c_ppa:g}, "
                    f"battery={power:g} MW/{energy:g} MWh."
                )

            offset = (
                battery_index - (len(BATTERY_CASES) - 1) / 2
            ) * bar_width
            values = battery_data[value_column].to_numpy(dtype=float)
            bars = ax.bar(
                x_positions + offset,
                values,
                width=bar_width * 0.92,
                color=palette[battery_index],
                edgecolor="white",
                linewidth=0.4,
                label=battery_label(power, energy),
            )

            for bar, value in zip(bars, values):
                decimals = 1 if CONTRACT_FIGURE_METRIC == "daily_energy" else 2
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.018 * y_max,
                    f"{value:.{decimals}f}",
                    ha="center",
                    va="bottom",
                    rotation=55,
                    fontsize=7.6,
                )

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            rf"({panel_letter}) $c^{{PPA}}={c_ppa:g}$ EUR/MWh",
            fontsize=10.5,
        )
        ax.set_xticks(x_positions, [f"{value:g}" for value in ppa_prices])
        ax.set_ylim(0.0, y_max * 1.17)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )
    fig.suptitle(
        "Battery sensitivity of optimal PPA contracting\n"
        f"{PROFILE}, {SETTLEMENT_PERIOD.capitalize()} PPA deviation settlement",
        y=0.995,
        fontsize=12,
    )
    fig.supxlabel(
        r"PPA price, $\lambda^{PPA}$ [EUR/MWh]",
        y=0.03,
        fontsize=11.5,
    )
    fig.supylabel(y_label, x=0.025, fontsize=11.5)
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.085, right=0.98)

    output_dir = OUTPUT_ROOT / "contracting"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / (
        f"{output_name}_{profile_slug(PROFILE)}_"
        f"settlement_{SETTLEMENT_PERIOD.lower()}"
    )
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    catalog = load_catalog()
    data = select_contract_cases(catalog, PROFILE)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data_path = OUTPUT_ROOT / "battery_contracting_cases.csv"
    data.to_csv(data_path, index=False, float_format="%.15g")
    outputs = plot_contracts(data)
    print(f"Saved selected contract cases: {data_path}")
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()

