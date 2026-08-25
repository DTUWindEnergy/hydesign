"""Create a 2x2 NPV heatmap for the four PPA settlement periods."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
OUTPUT_DIR = HERE / "outputs" / "npv_heatmap_by_settlement_period"
NPV_RESULTS_PATH = HERE / "outputs" / "npv_results.csv"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(POST_PROCESSING_ROOT))

from npv_analysis import calculate_npv_results  # noqa: E402
from settings import (  # noqa: E402
    C_PPA_VALUES,
    DISCOUNT_RATE,
    E_MAX,
    LAMBDA_PPA_VALUES,
    P_B_MAX,
    PROFILE,
    PROJECT_LIFETIME_YEARS,
    SETTLEMENT_PERIODS,
)


PERIOD_LABELS = {
    "hourly": "Hourly",
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
}

# Negative NPV, zero NPV and positive NPV, respectively.
NPV_CMAP = LinearSegmentedColormap.from_list(
    "npv_feasibility",
    ["#C05503", "#F7F7F7", "#0072B2"],
)


def _load_npv_results() -> pd.DataFrame:
    if NPV_RESULTS_PATH.exists():
        return pd.read_csv(NPV_RESULTS_PATH)
    return calculate_npv_results()


def _period_matrix(data: pd.DataFrame, settlement_period: str) -> np.ndarray:
    period_data = data.loc[
        data["sdp_label"].eq(PROFILE)
        & data["settlement_period"].eq(settlement_period)
    ].copy()

    matrix = np.full(
        (len(C_PPA_VALUES), len(LAMBDA_PPA_VALUES)),
        np.nan,
        dtype=float,
    )
    for row_index, c_ppa in enumerate(C_PPA_VALUES):
        for column_index, lambda_ppa in enumerate(LAMBDA_PPA_VALUES):
            match = period_data.loc[
                np.isclose(period_data["c_ppa"], c_ppa)
                & np.isclose(period_data["lambda_PPA"], lambda_ppa)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one NPV result for {settlement_period}, "
                    f"lambda_PPA={lambda_ppa:g}, c_ppa={c_ppa:g}; "
                    f"found {len(match)}."
                )
            matrix[row_index, column_index] = float(match.iloc[0]["npv_meur"])
    return matrix


def plot_npv_heatmap(data: pd.DataFrame | None = None) -> list[Path]:
    if data is None:
        data = _load_npv_results()

    matrices = {
        period: _period_matrix(data, period) for period in SETTLEMENT_PERIODS
    }
    all_values = np.concatenate([matrix.ravel() for matrix in matrices.values()])
    absolute_limit = max(abs(float(all_values.min())), abs(float(all_values.max())))
    norm = TwoSlopeNorm(vmin=-absolute_limit, vcenter=0.0, vmax=absolute_limit)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.6, 8.2),
        sharex=True,
        sharey=True,
    )

    image = None
    panel_letters = ("a", "b", "c", "d")
    for ax, panel_letter, settlement_period in zip(
        axes.flat,
        panel_letters,
        SETTLEMENT_PERIODS,
    ):
        matrix = matrices[settlement_period]
        image = ax.imshow(matrix, cmap=NPV_CMAP, norm=norm, aspect="auto")

        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                rgba = NPV_CMAP(norm(value))
                luminance = (
                    0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                )
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="white" if luminance < 0.55 else "black",
                    fontweight="semibold",
                )
                ax.add_patch(
                    Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="#202020",
                        linewidth=1.2,
                    )
                )

        ax.set_title(
            f"({panel_letter}) {PERIOD_LABELS[settlement_period]} settlement",
            fontsize=12,
            pad=4 if settlement_period in ("hourly", "daily") else 6,
        )
        ax.set_xticks(
            np.arange(len(LAMBDA_PPA_VALUES)),
            [f"{value:g}" for value in LAMBDA_PPA_VALUES],
        )
        ax.set_yticks(
            np.arange(len(C_PPA_VALUES)),
            [f"{value:g}" for value in C_PPA_VALUES],
        )
        ax.tick_params(axis="both", labelsize=10.5)

    fig.subplots_adjust(
        left=0.09,
        right=0.86,
        bottom=0.09,
        top=0.88,
        wspace=0.06,
        hspace=0.18,
    )
    fig.supxlabel(
        r"PPA price, $\lambda^{PPA}$ [EUR/MWh]",
        fontsize=12,
        y=0.025,
    )
    fig.supylabel(
        r"PPA deviation penalty, $c^{PPA}$ [EUR/MWh]",
        fontsize=12,
        x=0.025,
    )
    fig.suptitle(
        "Project NPV by PPA deviation settlement period\n"
        f"{PROFILE}, {P_B_MAX:g} MW/{E_MAX:g} MWh battery, "
        f"{PROJECT_LIFETIME_YEARS}-year project, "
        f"{DISCOUNT_RATE:.0%} discount rate",
        fontsize=14,
        y=0.965,
    )

    if image is None:
        raise RuntimeError("No heatmap panels were created.")
    colorbar_axis = fig.add_axes([0.89, 0.14, 0.018, 0.58])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"Net present value [MEUR$_{2016}$]", fontsize=11)
    colorbar.ax.tick_params(labelsize=10)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "npv_heatmap_all_settlement_periods"
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    outputs = plot_npv_heatmap()
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
