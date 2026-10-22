"""Create one three-panel NPV figure for each settlement period."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
OUTPUT_DIR = HERE / "outputs" / "npv_by_settlement_period"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.plotting_style import get_palette  # noqa: E402
from npv_analysis import calculate_npv_results  # noqa: E402
from settings import (  # noqa: E402
    C_PPA_VALUES,
    E_MAX,
    LAMBDA_PPA_VALUES,
    P_B_MAX,
    PALETTE_NAME,
    PROFILE,
    SETTLEMENT_PERIODS,
)


PERIOD_LABELS = {
    "hourly": "Hourly",
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
}


def _plot_period(data, settlement_period: str) -> list[Path]:
    palette = get_palette(PALETTE_NAME)
    if len(palette) < len(LAMBDA_PPA_VALUES):
        raise ValueError(f"Palette '{PALETTE_NAME}' needs at least four colors.")

    period_data = data.loc[
        data["sdp_label"].eq(PROFILE)
        & data["settlement_period"].eq(settlement_period)
    ].copy()
    fig = plt.figure(figsize=(12.8, 8.6))
    grid = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.42)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 1:3]),
    ]

    y_min = min(float(period_data["npv_meur"].min()), 0.0)
    y_max = max(float(period_data["npv_meur"].max()), 0.0)
    y_span = max(y_max - y_min, 1.0)
    x = np.arange(len(LAMBDA_PPA_VALUES))

    for panel_index, (ax, c_ppa) in enumerate(zip(axes, C_PPA_VALUES)):
        panel = (
            period_data.loc[np.isclose(period_data["c_ppa"], c_ppa)]
            .set_index("lambda_PPA")
            .reindex(LAMBDA_PPA_VALUES)
        )
        if panel["npv_meur"].isna().any():
            raise ValueError(
                f"Incomplete NPV data for {settlement_period}, c_ppa={c_ppa:g}."
            )

        values = panel["npv_meur"].to_numpy(dtype=float)
        bars = ax.bar(x, values, width=0.62, color=palette[: len(x)])
        for bar, value in zip(bars, values):
            offset = 0.025 * y_span
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset if value >= 0 else value - offset,
                f"{value:.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8.5,
                rotation=45,
            )

        letter = chr(ord("a") + panel_index)
        ax.set_title(
            rf"({letter}) $c^{{PPA}}={c_ppa:g}$ EUR/MWh",
            fontsize=10.5,
        )
        ax.set_xticks(x, [f"{value:g}" for value in LAMBDA_PPA_VALUES])
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_ylim(y_min - 0.12 * y_span, y_max + 0.15 * y_span)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8.5)

    fig.suptitle(
        f"Project NPV under {PERIOD_LABELS[settlement_period].lower()} "
        "PPA deviation settlement\n"
        f"{PROFILE}, {P_B_MAX:g} MW/{E_MAX:g} MWh battery",
        y=0.99,
        fontsize=12,
    )
    fig.supxlabel(r"PPA price, $\lambda^{PPA}$ [EUR/MWh]", y=0.03, fontsize=11.5)
    fig.supylabel("Net present value [MEUR2016]", x=0.025, fontsize=11.5)
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.085, right=0.98)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"npv_settlement_{settlement_period}"
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_all_periods(data=None) -> list[Path]:
    if data is None:
        data = calculate_npv_results()
    outputs = []
    for settlement_period in SETTLEMENT_PERIODS:
        outputs.extend(_plot_period(data, settlement_period))
    return outputs


def main() -> None:
    outputs = plot_all_periods()
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
