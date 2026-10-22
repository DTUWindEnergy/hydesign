"""Create one equally spaced 12-bar NPV figure per settlement period."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
OUTPUT_DIR = HERE / "outputs" / "npv_12_bars_by_settlement_period"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(POST_PROCESSING_ROOT))

from npv_analysis import calculate_npv_results  # noqa: E402
from settings import (  # noqa: E402
    C_PPA_VALUES,
    E_MAX,
    LAMBDA_PPA_VALUES,
    NPV_12_BAR_COLOR,
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


def _plot_period(data, settlement_period: str) -> list[Path]:
    period_data = data.loc[
        data["sdp_label"].eq(PROFILE)
        & data["settlement_period"].eq(settlement_period)
    ].copy()
    if period_data.empty:
        raise ValueError(f"No NPV data for {settlement_period} settlement.")

    y_min = min(float(period_data["npv_meur"].min()), 0.0)
    y_max = max(float(period_data["npv_meur"].max()), 0.0)
    y_span = max(y_max - y_min, 1.0)

    values = []
    tick_labels = []

    for lambda_ppa in LAMBDA_PPA_VALUES:
        for c_ppa in reversed(C_PPA_VALUES):
            match = period_data.loc[
                np.isclose(period_data["lambda_PPA"], lambda_ppa)
                & np.isclose(period_data["c_ppa"], c_ppa)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one NPV result for {settlement_period}, "
                    f"lambda_PPA={lambda_ppa:g}, c_ppa={c_ppa:g}; "
                    f"found {len(match)}."
                )

            values.append(float(match.iloc[0]["npv_meur"]))
            tick_labels.append(
                rf"$\lambda^{{PPA}}={lambda_ppa:g}$"
                "\n"
                rf"$c^{{PPA}}={c_ppa:g}$"
            )

    positions = np.arange(len(values), dtype=float)
    fig, ax = plt.subplots(figsize=(12.4, 6.5))
    bars = ax.bar(positions, values, width=0.58, color=NPV_12_BAR_COLOR)

    for bar, value in zip(bars, values):
        label_offset = 0.025 * y_span
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + label_offset if value >= 0 else value - label_offset,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
            rotation=45,
        )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(positions, tick_labels)
    ax.set_xlabel(
        r"PPA price $\lambda^{PPA}$ and deviation penalty $c^{PPA}$ "
        "[EUR/MWh]",
        fontsize=11.5,
    )
    ax.set_ylabel(r"Net present value [MEUR$_{2016}$]", fontsize=11.5)
    ax.set_ylim(y_min - 0.13 * y_span, y_max + 0.17 * y_span)
    ax.set_title(
        f"Project NPV under {PERIOD_LABELS[settlement_period].lower()} "
        "PPA deviation settlement\n"
        f"{PROFILE}, {P_B_MAX:g} MW/{E_MAX:g} MWh battery, "
        f"{PROJECT_LIFETIME_YEARS}-year project",
        fontsize=12.5,
    )
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"npv_12_bars_{settlement_period}"
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_all_periods_12_bars(data=None) -> list[Path]:
    if data is None:
        data = calculate_npv_results()
    outputs = []
    for settlement_period in SETTLEMENT_PERIODS:
        outputs.extend(_plot_period(data, settlement_period))
    return outputs


def main() -> None:
    outputs = plot_all_periods_12_bars()
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
