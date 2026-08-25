"""Plot optimal PPA capacity against the PPA deviation settlement period."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
OUTPUT_DIR = HERE / "outputs" / "optimal_contract"

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
from settlement_analysis import load_contract_data  # noqa: E402


PERIOD_LABELS = {
    "hourly": "Hourly",
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
}


def plot_capacity() -> list[Path]:
    data = load_contract_data()
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
    y_max = max(float(data["S_PPA"].max()), 0.1)

    for panel_index, (ax, c_ppa) in enumerate(zip(axes, C_PPA_VALUES)):
        panel = data.loc[np.isclose(data["c_ppa"], c_ppa)]
        for price_index, lambda_ppa in enumerate(LAMBDA_PPA_VALUES):
            curve = (
                panel.loc[np.isclose(panel["lambda_PPA"], lambda_ppa)]
                .set_index("settlement_period")
                .reindex(SETTLEMENT_PERIODS)
            )
            if curve["S_PPA"].isna().any():
                raise ValueError(
                    f"Incomplete capacity data for c_ppa={c_ppa:g}, "
                    f"lambda_PPA={lambda_ppa:g}."
                )
            ax.plot(
                x_positions,
                curve["S_PPA"],
                color=palette[price_index],
                marker="o",
                markersize=5,
                linewidth=1.8,
                label=rf"$\lambda^{{PPA}}={lambda_ppa:g}$",
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
        ax.set_ylim(0.0, y_max * 1.08)
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
        "Optimal PPA contracted capacity by deviation settlement period\n"
        f"{PROFILE}, {P_B_MAX:g} MW/{E_MAX:g} MWh battery",
        y=0.995,
        fontsize=12,
    )
    fig.supxlabel("PPA deviation settlement period", y=0.03, fontsize=11.5)
    fig.supylabel(
        r"Optimal PPA contracted capacity, $S^{PPA,*}$ [MW]",
        x=0.025,
        fontsize=11.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.085, right=0.98)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "optimal_ppa_capacity_by_settlement_period"
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    outputs = plot_capacity()
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
