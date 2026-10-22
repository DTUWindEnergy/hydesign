"""Plot expected annual profit against the deviation settlement period."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
OUTPUT_DIR = HERE / "outputs" / "expected_profit"

sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.plotting_style import get_palette  # noqa: E402
from settings import (  # noqa: E402
    C_PPA_VALUES,
    E_MAX,
    LAMBDA_PPA_VALUES,
    P_B_MAX,
    PALETTE_NAME,
    PROFILE,
    SETTLEMENT_PERIODS,
)
from settlement_analysis import build_audited_settlement_data  # noqa: E402


PERIOD_LABELS = {
    "hourly": "Hourly",
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
}


def plot_profit() -> list[Path]:
    data = build_audited_settlement_data()
    data = data.loc[data["sdp_label"].eq(PROFILE)].copy()
    palette = get_palette(PALETTE_NAME)
    if len(palette) < len(LAMBDA_PPA_VALUES):
        raise ValueError(f"Palette '{PALETTE_NAME}' needs at least four colors.")

    fig = plt.figure(figsize=(12.8, 8.6))
    grid = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.42)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 1:3]),
    ]
    x_positions = np.arange(len(SETTLEMENT_PERIODS), dtype=float)
    bar_width = 0.18
    bar_spacing = 0.20
    number_of_prices = len(LAMBDA_PPA_VALUES)
    profit_meur = data["expected_profit"] / 1e6
    y_min = min(float(profit_meur.min()), 0.0)
    y_max = max(float(profit_meur.max()), 0.1)
    y_span = y_max - y_min

    for panel_index, (ax, c_ppa) in enumerate(zip(axes, C_PPA_VALUES)):
        panel = data.loc[np.isclose(data["c_ppa"], c_ppa)]
        for price_index, lambda_ppa in enumerate(LAMBDA_PPA_VALUES):
            curve = (
                panel.loc[np.isclose(panel["lambda_PPA"], lambda_ppa)]
                .set_index("settlement_period")
                .reindex(SETTLEMENT_PERIODS)
            )
            if curve["expected_profit"].isna().any():
                raise ValueError(
                    f"Incomplete profit data for c_ppa={c_ppa:g}, "
                    f"lambda_PPA={lambda_ppa:g}."
                )
            offset = (
                price_index - (number_of_prices - 1) / 2
            ) * bar_spacing
            values = curve["expected_profit"].to_numpy(dtype=float) / 1e6
            bars = ax.bar(
                x_positions + offset,
                values,
                width=bar_width,
                color=palette[price_index],
                label=rf"$\lambda^{{PPA}}={lambda_ppa:g}$",
            )
            ax.bar_label(
                bars,
                labels=[f"{value:.2f}" for value in values],
                padding=2,
                rotation=55,
                fontsize=7,
            )

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            rf"({panel_letter}) $c^{{PPA}}={c_ppa:g}$ EUR/MWh",
            fontsize=10.5,
        )
        ax.set_xticks(
            x_positions,
            [PERIOD_LABELS[period] for period in SETTLEMENT_PERIODS],
        )
        ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.16 * y_span)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Expected annual profit by deviation settlement period\n"
        f"{PROFILE}, {P_B_MAX:g} MW/{E_MAX:g} MWh battery",
        y=0.995,
        fontsize=12,
    )
    fig.supxlabel("PPA deviation settlement period", y=0.03, fontsize=11.5)
    fig.supylabel(
        "Expected annual profit [MEUR/year]",
        x=0.025,
        fontsize=11.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.085, right=0.98)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "expected_profit_by_settlement_period"
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    outputs = plot_profit()
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
