"""Plot expected annual profit by delivery profile and PPA price."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
DECOMPOSITION_DATA = (
    POST_PROCESSING_ROOT
    / "04_profit_decomposition"
    / "outputs"
    / "expected_profit_decomposition_split_imbalance.csv"
)
OUTPUT_ROOT = HERE / "outputs"

sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.plotting_style import get_palette
from common.results_catalog import PROJECT_FOLDERS
from config import (
    BATTERY_ENERGY_MWH,
    BATTERY_POWER_MW,
    C_PPA_VALUES,
    LAMBDA_PPA_VALUES,
    PALETTE_NAME,
    SETTLEMENT_PERIOD,
)


PROFILE_ORDER = list(PROJECT_FOLDERS)
PROFILE_TICK_LABELS = {
    "Generic Industrial": "Generic\nIndustrial",
    "Generic Private": "Generic\nPrivate",
    "Inverse Solar": "Inverse\nSolar",
    "Continuous Baseload": "Continuous\nBaseload",
}


def number_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def battery_slug(power: float, energy: float) -> str:
    return f"battery_p_{number_slug(power)}_e_{number_slug(energy)}"


def battery_label(power: float, energy: float) -> str:
    return f"{float(power):g} MW/{float(energy):g} MWh battery"


def compact_eur_label(value: float) -> str:
    absolute = abs(value)
    sign = "-" if value < 0 else ""
    if absolute >= 1_000_000:
        scaled = absolute / 1_000_000
        decimals = 2 if scaled < 10 else 1
        return f"{sign}{scaled:.{decimals}f}M"
    if absolute >= 1_000:
        scaled = absolute / 1_000
        decimals = 1 if scaled < 100 else 0
        return f"{sign}{scaled:.{decimals}f}k"
    return f"{value:.0f}"


def load_expected_profit() -> pd.DataFrame:
    if not DECOMPOSITION_DATA.exists():
        raise FileNotFoundError(
            f"Missing audited expected-profit data: {DECOMPOSITION_DATA}\n"
            "Run post_processing/04_profit_decomposition/"
            "plot_profit_decomposition.py first."
        )

    data = pd.read_csv(DECOMPOSITION_DATA)
    required = {
        "settlement_period",
        "P_b_max",
        "E_max",
        "c_ppa",
        "lambda_PPA",
        "sdp_label",
        "expected_profit",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"Missing columns from {DECOMPOSITION_DATA}: {missing}")

    selected = data.loc[
        data["settlement_period"].str.lower().eq(SETTLEMENT_PERIOD.lower())
        & np.isclose(data["P_b_max"], BATTERY_POWER_MW)
        & np.isclose(data["E_max"], BATTERY_ENERGY_MWH)
        & data["c_ppa"].isin(C_PPA_VALUES)
        & data["lambda_PPA"].isin(LAMBDA_PPA_VALUES)
    ].copy()

    expected_rows = (
        len(PROFILE_ORDER) * len(C_PPA_VALUES) * len(LAMBDA_PPA_VALUES)
    )
    if len(selected) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} cases, found {len(selected)} in "
            f"{DECOMPOSITION_DATA}."
        )
    return selected


def plot_expected_profit(data: pd.DataFrame) -> list[Path]:
    c_ppa_values = sorted(data["c_ppa"].unique())
    ppa_prices = sorted(data["lambda_PPA"].unique())
    if len(c_ppa_values) != 3:
        raise ValueError("The centered 2+1 layout requires three c_ppa values.")

    palette = get_palette(PALETTE_NAME)
    if len(palette) < len(ppa_prices):
        raise ValueError(
            f"Palette '{PALETTE_NAME}' has {len(palette)} colors, "
            f"but {len(ppa_prices)} PPA prices were requested."
        )
    price_colors = {
        price: palette[index] for index, price in enumerate(ppa_prices)
    }

    values_meur = data["expected_profit"] / 1e6
    y_min = min(float(values_meur.min()), 0.0)
    y_max = max(float(values_meur.max()), 0.1)
    y_span = y_max - y_min

    fig = plt.figure(figsize=(13.2, 8.8))
    grid = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.45)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 1:3]),
    ]

    x_positions = np.arange(len(PROFILE_ORDER), dtype=float)
    group_width = 0.78
    bar_width = group_width / len(ppa_prices)

    for panel_index, (ax, c_ppa) in enumerate(zip(axes, c_ppa_values)):
        panel = data.loc[np.isclose(data["c_ppa"], c_ppa)]
        pivot = panel.pivot(
            index="sdp_label",
            columns="lambda_PPA",
            values="expected_profit",
        ).reindex(index=PROFILE_ORDER, columns=ppa_prices)
        if pivot.isna().any().any():
            raise ValueError(
                f"Incomplete expected-profit data for c_ppa={c_ppa:g}."
            )

        for price_index, price in enumerate(ppa_prices):
            offset = (
                price_index - (len(ppa_prices) - 1) / 2
            ) * bar_width
            values_eur = pivot[price].to_numpy(dtype=float)
            panel_values_meur = values_eur / 1e6
            bars = ax.bar(
                x_positions + offset,
                panel_values_meur,
                width=bar_width * 0.92,
                color=price_colors[price],
                edgecolor="white",
                linewidth=0.4,
                label=rf"$\lambda^{{PPA}}={price:g}$",
            )

            for bar, value_eur, value_meur in zip(
                bars, values_eur, panel_values_meur
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value_meur + 0.018 * y_span,
                    compact_eur_label(value_eur),
                    ha="center",
                    va="bottom",
                    rotation=55,
                    fontsize=8.0,
                )

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            rf"({panel_letter}) $c^{{PPA}}={c_ppa:g}$ EUR/MWh",
            fontsize=10.5,
        )
        ax.set_xticks(
            x_positions,
            [PROFILE_TICK_LABELS[profile] for profile in PROFILE_ORDER],
        )
        ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.16 * y_span)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        "Expected annual profit by shaped delivery profile and PPA price\n"
        f"{SETTLEMENT_PERIOD.capitalize()} PPA deviation settlement, "
        f"{battery_label(BATTERY_POWER_MW, BATTERY_ENERGY_MWH)}",
        y=0.995,
        fontsize=12,
    )
    fig.supylabel(
        "Expected annual profit [MEUR/year]",
        x=0.03,
        fontsize=12,
    )
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.98)

    output_dir = (
        OUTPUT_ROOT
        / f"settlement_{SETTLEMENT_PERIOD.lower()}"
        / battery_slug(BATTERY_POWER_MW, BATTERY_ENERGY_MWH)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "expected_profit_by_sdp_2plus1"

    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    data = load_expected_profit()
    outputs = plot_expected_profit(data)
    print("Created expected-profit figure:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
