"""Calculate and plot PPA indifference curves by settlement period."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
PROJECT_ROOT = ANALYSIS_DIR.parent
SOURCE_PATH = (
    PROJECT_ROOT
    / "post_processing"
    / "07_settlement_period_effect"
    / "outputs"
    / "settlement_period_analysis.csv"
)
MERCHANT_PATH = ANALYSIS_DIR / "results" / "merchant" / "summary.json"
OUTPUT_DIR = ANALYSIS_DIR / "results" / "settlement_period_comparison"
FIGURE_DIR = OUTPUT_DIR / "figures"
CSV_PATH = OUTPUT_DIR / "indifference_prices_by_settlement_period.csv"

sys.path.insert(0, str(PROJECT_ROOT / "post_processing"))
from common.plotting_style import get_palette  # noqa: E402

sys.path.insert(0, str(HERE))
from settings import (  # noqa: E402
    C_PPA_VALUES,
    E_MAX,
    PALETTE_NAME,
    PROFILE,
    P_B_MAX,
    REFERENCE_PPA_PRICES,
    SETTLEMENT_PERIODS,
)


def calculate_indifference_prices() -> pd.DataFrame:
    """Read completed solves and apply the exact indifference-price shortcut."""
    for path in (SOURCE_PATH, MERCHANT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}.")

    merchant = pd.read_json(MERCHANT_PATH, typ="series")
    merchant_objective = float(merchant["objective_value"])
    source = pd.read_csv(SOURCE_PATH)

    selected = source.loc[
        (source["sdp_label"] == PROFILE)
        & source["settlement_period"].str.lower().isin(SETTLEMENT_PERIODS)
        & np.isclose(source["P_b_max"], P_B_MAX)
        & np.isclose(source["E_max"], E_MAX)
        & source["c_ppa"].isin(C_PPA_VALUES)
        & source["lambda_PPA"].isin(REFERENCE_PPA_PRICES)
    ].copy()

    required = {
        "settlement_period",
        "c_ppa",
        "lambda_PPA",
        "S_PPA",
        "annual_contracted_energy_mwh",
        "reported_objective",
    }
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"Settlement results are missing columns: {missing}")

    expected_rows = (
        len(C_PPA_VALUES)
        * len(SETTLEMENT_PERIODS)
        * len(REFERENCE_PPA_PRICES)
    )
    if len(selected) != expected_rows:
        counts = selected.groupby(
            ["c_ppa", "settlement_period"], dropna=False
        ).size()
        raise ValueError(
            f"Expected {expected_rows} cases, found {len(selected)}.\n{counts}"
        )
    if (selected["annual_contracted_energy_mwh"] <= 0.0).any():
        invalid = selected.loc[
            selected["annual_contracted_energy_mwh"] <= 0.0,
            ["settlement_period", "c_ppa", "lambda_PPA", "S_PPA"],
        ]
        raise ValueError(
            "Indifference price is undefined for zero contracted energy:\n"
            f"{invalid.to_string(index=False)}"
        )

    selected["merchant_objective"] = merchant_objective
    selected["indifference_price_eur_per_mwh"] = (
        selected["lambda_PPA"]
        + (merchant_objective - selected["reported_objective"])
        / selected["annual_contracted_energy_mwh"]
    )
    period_order = {period: index for index, period in enumerate(SETTLEMENT_PERIODS)}
    selected["period_order"] = selected["settlement_period"].map(period_order)
    selected = selected.sort_values(["c_ppa", "period_order", "S_PPA"])

    output_columns = [
        "settlement_period",
        "c_ppa",
        "lambda_PPA",
        "S_PPA",
        "annual_contracted_energy_mwh",
        "reported_objective",
        "merchant_objective",
        "indifference_price_eur_per_mwh",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected[output_columns].to_csv(CSV_PATH, index=False)
    return selected


def plot_curves(data: pd.DataFrame) -> list[Path]:
    """Overlay the four settlement-period curves for each deviation penalty."""
    colors = get_palette(PALETTE_NAME)
    markers = ("o", "s", "^", "D")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []

    for c_ppa in C_PPA_VALUES:
        panel = data.loc[np.isclose(data["c_ppa"], c_ppa)]
        x_values = panel["S_PPA"].to_numpy(dtype=float)
        y_values = panel["indifference_price_eur_per_mwh"].to_numpy(dtype=float)
        x_margin = max(0.5, 0.06 * (x_values.max() - x_values.min()))
        y_margin = max(2.0, 0.06 * (y_values.max() - y_values.min()))
        x_limits = (max(0.0, x_values.min() - x_margin), x_values.max() + x_margin)
        y_limits = (0.0, y_values.max() + y_margin)

        fig, ax = plt.subplots(figsize=(10.8, 6.6))

        for index, period in enumerate(SETTLEMENT_PERIODS):
            curve = panel.loc[
                panel["settlement_period"].str.lower() == period
            ].sort_values("S_PPA")
            ax.plot(
                curve["S_PPA"],
                curve["indifference_price_eur_per_mwh"],
                color=colors[index],
                marker=markers[index],
                linewidth=2.2,
                markersize=6.5,
                label=period.capitalize(),
            )
        ax.set_title(
            "Merchant-equivalent PPA price curves by deviation settlement period\n"
            rf"{PROFILE}, $c^{{PPA}}={c_ppa:.0f}$ EUR/MWh, "
            "20 MW/60 MWh battery",
            fontsize=15,
        )
        ax.set_xlabel(
            r"Optimal contracted PPA capacity, $S^{PPA,*}$ [MW]",
            fontsize=13,
        )
        ax.set_ylabel("PPA indifference price [EUR/MWh]", fontsize=13)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="both", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(
            title="Deviation settlement period",
            fontsize=10.5,
            title_fontsize=10.5,
        )
        fig.tight_layout()

        output = FIGURE_DIR / (
            "indifference_curves_by_settlement_period_"
            f"c_ppa_{c_ppa:.0f}.png"
        )
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

    return outputs


def main() -> None:
    data = calculate_indifference_prices()
    figures = plot_curves(data)
    print(f"CSV: {CSV_PATH}")
    for figure in figures:
        print(f"Figure: {figure}")


if __name__ == "__main__":
    main()
