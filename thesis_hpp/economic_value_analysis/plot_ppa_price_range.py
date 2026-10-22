"""Plot PPA reference, indifference and project break-even price curves."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
THESIS_ROOT = HERE.parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MERCHANT_PATH = RESULTS_DIR / "merchant" / "summary.json"
VALIDATION_PATH = RESULTS_DIR / "indifference_validations_summary.csv"
SETTLEMENT_RESULTS_PATH = (
    THESIS_ROOT
    / "post_processing"
    / "07_settlement_period_effect"
    / "outputs"
    / "settlement_period_analysis.csv"
)
NPV_RESULTS_PATH = (
    THESIS_ROOT
    / "post_processing"
    / "08_npv_analysis"
    / "outputs"
    / "npv_results.csv"
)
PLOT_DATA_PATH = RESULTS_DIR / "ppa_price_range_all_cppas.csv"

sys.path.insert(0, str(THESIS_ROOT / "post_processing"))
from common.plotting_style import get_palette  # noqa: E402

sys.path.insert(0, str(HERE))
from plot_settings import (  # noqa: E402
    C_PPA_VALUES,
    E_MAX,
    PALETTE_NAME,
    PROFILE,
    P_B_MAX,
    SETTLEMENT_PERIOD,
)


def _load_price_range_data() -> pd.DataFrame:
    for path in (MERCHANT_PATH, SETTLEMENT_RESULTS_PATH, NPV_RESULTS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}.")

    merchant = pd.read_json(MERCHANT_PATH, typ="series")
    merchant_objective = float(merchant["objective_value"])

    settlement = pd.read_csv(SETTLEMENT_RESULTS_PATH)
    selected = settlement.loc[
        (settlement["sdp_label"] == PROFILE)
        & (settlement["settlement_period"].str.lower() == SETTLEMENT_PERIOD)
        & np.isclose(settlement["P_b_max"], P_B_MAX)
        & np.isclose(settlement["E_max"], E_MAX)
        & settlement["c_ppa"].isin(C_PPA_VALUES)
    ].copy()

    required_settlement = {
        "c_ppa",
        "lambda_PPA",
        "S_PPA",
        "annual_contracted_energy_mwh",
        "reported_objective",
    }
    missing = sorted(required_settlement.difference(selected.columns))
    if missing:
        raise ValueError(f"Settlement results are missing columns: {missing}")

    npv = pd.read_csv(NPV_RESULTS_PATH)
    npv_selected = npv.loc[
        (npv["sdp_label"] == PROFILE)
        & (npv["settlement_period"].str.lower() == SETTLEMENT_PERIOD)
        & np.isclose(npv["P_b_max"], P_B_MAX)
        & np.isclose(npv["E_max"], E_MAX)
        & npv["c_ppa"].isin(C_PPA_VALUES)
    ][
        [
            "c_ppa",
            "lambda_PPA",
            "S_PPA",
            "annual_cash_flow_eur",
            "required_annual_cash_flow_eur",
        ]
    ].copy()

    merged = selected.merge(
        npv_selected,
        on=["c_ppa", "lambda_PPA", "S_PPA"],
        how="inner",
        validate="one_to_one",
    )
    expected_rows = len(C_PPA_VALUES) * 4
    if len(merged) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} price-range points, found {len(merged)}."
        )
    if (merged["annual_contracted_energy_mwh"] <= 0.0).any():
        raise ValueError("The price-range calculation requires positive PPA energy.")

    energy = merged["annual_contracted_energy_mwh"]
    merged["risk_adjusted_indifference_price_eur_per_mwh"] = (
        merged["lambda_PPA"]
        + (merchant_objective - merged["reported_objective"]) / energy
    )
    merged["npv_zero_price_eur_per_mwh"] = (
        merged["lambda_PPA"]
        + (
            merged["required_annual_cash_flow_eur"]
            - merged["annual_cash_flow_eur"]
        )
        / energy
    )
    merged["merchant_objective"] = merchant_objective
    merged = merged.sort_values(["c_ppa", "S_PPA"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    merged[
        [
            "c_ppa",
            "lambda_PPA",
            "S_PPA",
            "annual_contracted_energy_mwh",
            "reported_objective",
            "merchant_objective",
            "risk_adjusted_indifference_price_eur_per_mwh",
            "annual_cash_flow_eur",
            "required_annual_cash_flow_eur",
            "npv_zero_price_eur_per_mwh",
        ]
    ].to_csv(PLOT_DATA_PATH, index=False)
    return merged


def _price_axis_limits(panel: pd.DataFrame) -> tuple[float, float]:
    """Keep the scale determined by the reference and indifference curves."""
    visible_values = np.concatenate(
        [
            panel["lambda_PPA"].to_numpy(dtype=float),
            panel[
                "risk_adjusted_indifference_price_eur_per_mwh"
            ].to_numpy(dtype=float),
        ]
    )
    value_range = float(visible_values.max() - visible_values.min())
    margin = max(2.5, 0.05 * value_range)
    return (
        max(0.0, float(visible_values.min()) - margin),
        float(visible_values.max()) + margin,
    )


def _plot_npv_zero_curve(
    ax: plt.Axes,
    capacity: np.ndarray,
    npv_zero_price: np.ndarray,
    reference_price: np.ndarray,
    y_min: float,
    y_max: float,
    color: str,
) -> None:
    visible = (npv_zero_price >= y_min) & (npv_zero_price <= y_max)
    ax.plot(
        capacity,
        npv_zero_price,
        color=color,
        marker="^",
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.6,
        linestyle="--",
        linewidth=2.0,
        markersize=7.0,
        label=r"Project break-even curve ($NPV=0$)",
        clip_on=True,
        zorder=4,
    )

    if visible.any():
        visible_indices = np.flatnonzero(visible)
        for point_index in visible_indices:
            x_value = capacity[point_index]
            price = npv_zero_price[point_index]
            close_to_reference = (
                abs(price - reference_price[point_index]) < 1.5
            )
            text_offset = (30, 18) if close_to_reference else (0, 10)
            horizontal_alignment = "left" if close_to_reference else "center"
            ax.annotate(
                f"{price:.2f}",
                xy=(x_value, price),
                xytext=text_offset,
                textcoords="offset points",
                ha=horizontal_alignment,
                va="bottom",
                fontsize=9.5,
                color=color,
                arrowprops=(
                    {"arrowstyle": "-", "color": color, "lw": 0.9}
                    if close_to_reference
                    else None
                ),
            )

    for x_value, price in zip(capacity[~visible], npv_zero_price[~visible]):
        boundary = y_max if price > y_max else y_min
        vertical_offset = -25 if price > y_max else 25
        vertical_alignment = "top" if price > y_max else "bottom"
        ax.annotate(
            f"{price:,.2f}",
            xy=(x_value, boundary),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=9.5,
            color=color,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.2},
            annotation_clip=False,
        )


def _annotate_curve(
    ax: plt.Axes,
    capacity: np.ndarray,
    values: np.ndarray,
    color: str,
    offset: float,
    comparison_values: np.ndarray | None = None,
) -> None:
    for point_index, (x_value, value) in enumerate(zip(capacity, values)):
        close_to_comparison = (
            comparison_values is not None
            and abs(value - comparison_values[point_index]) < 1.5
        )
        text_offset = (-14, -18) if close_to_comparison else (0, offset)
        horizontal_alignment = "right" if close_to_comparison else "center"
        ax.annotate(
            f"{value:.2f}" if not float(value).is_integer() else f"{value:.0f}",
            xy=(x_value, value),
            xytext=text_offset,
            textcoords="offset points",
            ha=horizontal_alignment,
            va=(
                "top"
                if close_to_comparison
                else ("bottom" if offset >= 0 else "top")
            ),
            fontsize=9.5,
            color=color,
            arrowprops=(
                {"arrowstyle": "-", "color": color, "lw": 0.9}
                if close_to_comparison
                else None
            ),
        )


def plot_price_ranges(data: pd.DataFrame) -> list[Path]:
    colors = get_palette(PALETTE_NAME)
    outputs: list[Path] = []
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for c_ppa in C_PPA_VALUES:
        panel = data.loc[np.isclose(data["c_ppa"], c_ppa)].sort_values("S_PPA")
        capacity = panel["S_PPA"].to_numpy(dtype=float)
        reference = panel["lambda_PPA"].to_numpy(dtype=float)
        indifference = panel[
            "risk_adjusted_indifference_price_eur_per_mwh"
        ].to_numpy(dtype=float)
        npv_zero = panel["npv_zero_price_eur_per_mwh"].to_numpy(dtype=float)
        y_min, y_max = _price_axis_limits(panel)

        fig, ax = plt.subplots(figsize=(10.8, 7.2))
        ax.plot(
            capacity,
            reference,
            color=colors[0],
            marker="o",
            linewidth=2.2,
            markersize=6.5,
            label="Optimal PPA curve",
            zorder=3,
        )
        ax.plot(
            capacity,
            indifference,
            color=colors[1],
            marker="s",
            linewidth=2.2,
            markersize=6.2,
            label="Merchant-equivalent PPA curve",
            zorder=3,
        )
        _plot_npv_zero_curve(
            ax,
            capacity,
            npv_zero,
            reference,
            y_min,
            y_max,
            colors[3],
        )
        _annotate_curve(
            ax,
            capacity,
            reference,
            colors[0],
            8,
            comparison_values=npv_zero,
        )
        _annotate_curve(ax, capacity, indifference, colors[1], -16)
        ax.set_ylim(y_min, y_max)
        ax.set_title(
            f"PPA price curves for {SETTLEMENT_PERIOD.capitalize()} "
            "PPA deviation settlement\n"
            rf"Generic Industrial, $c^{{PPA}}={c_ppa:.0f}$ EUR/MWh, "
            "20 MW/60 MWh battery",
            fontsize=15,
        )
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_ylabel("PPA price [EUR/MWh]", fontsize=13)
        ax.set_xlabel(
            r"Fixed optimal contracted PPA capacity, $S^{PPA,*}$ [MW]",
            fontsize=13,
        )
        ax.set_xticks(
            capacity,
            [f"{value:.2f}" for value in capacity],
        )
        ax.legend(frameon=True, fontsize=10.5, loc="best")
        fig.tight_layout()

        output = FIGURES_DIR / (
            f"ppa_price_range_curve_c_ppa_{c_ppa:.0f}.png"
        )
        fig.savefig(output, dpi=300, bbox_inches="tight")
        if np.isclose(c_ppa, 160.0):
            # Keep the original filename, now without the expected-profit line.
            fig.savefig(
                FIGURES_DIR / "ppa_price_range_curve.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)
        outputs.append(output)

    return outputs


def plot_weighted_objective_components() -> Path:
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing {VALIDATION_PATH}. Run the four validations first."
        )
    data = pd.read_csv(VALIDATION_PATH).sort_values("s_ppa")
    required = {
        "s_ppa",
        "expected_profit_difference_from_merchant",
        "cvar_difference_from_merchant",
        "beta",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Validation summary is missing columns: {missing}")

    colors = get_palette(PALETTE_NAME)
    beta_values = data["beta"].to_numpy(dtype=float)
    if not np.allclose(beta_values, beta_values[0]):
        raise ValueError("The validation points do not use a common beta.")
    beta = float(beta_values[0])
    expected_contribution = (
        (1.0 - beta)
        * data["expected_profit_difference_from_merchant"].to_numpy(dtype=float)
        / 1_000.0
    )
    cvar_contribution = (
        beta
        * data["cvar_difference_from_merchant"].to_numpy(dtype=float)
        / 1_000.0
    )
    net_difference = expected_contribution + cvar_contribution
    labels = [
        rf"$S^{{PPA,*}}={value:.2f}$ MW"
        for value in data["s_ppa"].to_numpy(dtype=float)
    ]
    x_pos = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars_expected = ax.bar(
        x_pos - width / 2,
        expected_contribution,
        width,
        color=colors[0],
        label=r"$(1-\beta)\,\Delta$ expected profit",
    )
    bars_cvar = ax.bar(
        x_pos + width / 2,
        cvar_contribution,
        width,
        color=colors[1],
        label=r"$\beta\,\Delta$ CVaR",
    )
    ax.scatter(
        x_pos,
        net_difference,
        color=colors[2],
        marker="D",
        s=42,
        label=r"Net $\Delta J$",
        zorder=4,
    )
    ax.bar_label(bars_expected, fmt="%.2f", padding=3, fontsize=9.5)
    ax.bar_label(bars_cvar, fmt="%.2f", padding=3, fontsize=9.5)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x_pos, labels)
    ax.set_ylabel("Contribution relative to merchant operation [kEUR/year]", fontsize=13)
    ax.set_title(
        "Expected-profit and CVaR contributions at the indifference prices",
        fontsize=15,
    )
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=10.5, loc="best")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "indifference_objective_components.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    data = _load_price_range_data()
    outputs = plot_price_ranges(data)
    outputs.append(plot_weighted_objective_components())
    print(PLOT_DATA_PATH)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
