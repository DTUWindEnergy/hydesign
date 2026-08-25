"""Calculate and plot expected annual profit components for selected cases."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
CATALOG_PATH = POST_PROCESSING_ROOT / "common" / "results_catalog.csv"
OUTPUT_ROOT = HERE / "outputs"

sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.results_catalog import PROJECT_FOLDERS, project_roots
from common.plotting_style import get_palette
from config import (
    BATTERY_CASES,
    CASE_WORKERS,
    COMPONENTS,
    C_PPA_VALUES,
    CSV_CHUNK_SIZE,
    GENERATE_GROUPED_2X2_FIGURES,
    GENERATE_INDIVIDUAL_GROUPED_FIGURES,
    GENERATE_STACKED_FIGURES,
    GROUPED_MIN_LABEL_EUR,
    GROUPED_PLOT_PALETTE,
    LAMBDA_PPA_VALUES,
    MIN_SEGMENT_LABEL_MEUR,
    OBJECTIVE_AUDIT_ABS_TOL_EUR,
    OBJECTIVE_AUDIT_REL_TOL,
    REBUILD_DECOMPOSITION_DATA,
    SETTLEMENT_PERIODS,
)


PROFILE_ORDER = list(PROJECT_FOLDERS)
COMPONENT_KEYS = [key for key, _, _ in COMPONENTS]
COMPONENT_LABELS = {key: label for key, label, _ in COMPONENTS}
COMPONENT_COLORS = {key: color for key, _, color in COMPONENTS}


def number_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def battery_slug(power: float, energy: float) -> str:
    return f"battery_p_{number_slug(power)}_e_{number_slug(energy)}"


def battery_label(power: float, energy: float) -> str:
    return f"{float(power):g} MW/{float(energy):g} MWh battery"


def load_catalog() -> pd.DataFrame:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CATALOG_PATH}. Build the shared results catalog first."
        )

    catalog = pd.read_csv(CATALOG_PATH)
    numeric_columns = [
        "lambda_PPA",
        "c_ppa",
        "P_b_max",
        "E_max",
        "objective_value",
        "S_PPA",
    ]
    for column in numeric_columns:
        catalog[column] = pd.to_numeric(catalog[column], errors="raise")
    return catalog


def selected_cases(catalog: pd.DataFrame) -> pd.DataFrame:
    settlement_periods = {str(period).lower() for period in SETTLEMENT_PERIODS}
    battery_mask = pd.Series(False, index=catalog.index)
    for power, energy in BATTERY_CASES:
        battery_mask |= np.isclose(catalog["P_b_max"], power) & np.isclose(
            catalog["E_max"], energy
        )

    selected = catalog.loc[
        catalog["settlement_period"].str.lower().isin(settlement_periods)
        & battery_mask
        & catalog["c_ppa"].isin(C_PPA_VALUES)
        & catalog["lambda_PPA"].isin(LAMBDA_PPA_VALUES)
    ].copy()

    expected = (
        len(settlement_periods)
        * len(BATTERY_CASES)
        * len(C_PPA_VALUES)
        * len(LAMBDA_PPA_VALUES)
        * len(PROFILE_ORDER)
    )
    if len(selected) != expected:
        keys = [
            "settlement_period",
            "P_b_max",
            "E_max",
            "c_ppa",
            "lambda_PPA",
            "sdp_label",
        ]
        raise ValueError(
            f"Expected {expected} selected cases, found {len(selected)}.\n"
            + selected[keys].to_string(index=False)
        )

    return selected.sort_values(
        [
            "settlement_period",
            "P_b_max",
            "E_max",
            "c_ppa",
            "lambda_PPA",
            "sdp_label",
        ]
    ).reset_index(drop=True)


def local_case_directory(case: pd.Series) -> Path:
    """Translate the catalog's Sophia path to the current project root."""
    case_dir = str(case["case_dir"]).replace("\\", "/")
    marker = "/results/"
    if marker not in case_dir:
        raise ValueError(f"Cannot interpret catalog case path: {case_dir}")

    relative_results_path = case_dir.split(marker, maxsplit=1)[1]
    project_root = project_roots()[str(case["sdp_label"])]
    return project_root / "results" / Path(*relative_results_path.split("/"))


def load_project_parameters(project_root: Path) -> dict[str, float]:
    config_path = project_root / "inputs" / "Input_files_created" / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = ["c_imb", "degradation_cost_per_mw", "alpha", "beta"]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing {missing} from {config_path}")
    return {key: float(config[key]) for key in required}


def load_scenario_probabilities(project_root: Path) -> pd.DataFrame:
    inputs_dir = project_root / "inputs" / "Input_files_created"
    omega = pd.read_csv(
        inputs_dir / "omega_probabilities.csv",
        usecols=["omega", "prob"],
    ).rename(columns={"prob": "pi_omega"})
    xi = pd.read_csv(
        inputs_dir / "xi_probabilities.csv",
        usecols=["omega", "xi", "prob"],
    ).rename(columns={"prob": "pi_xi_given_omega"})

    probabilities = xi.merge(omega, on="omega", validate="many_to_one")
    probabilities["scenario_probability"] = (
        probabilities["pi_omega"] * probabilities["pi_xi_given_omega"]
    )

    if not math.isclose(
        float(omega["pi_omega"].sum()), 1.0, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError(f"Omega probabilities do not sum to one in {inputs_dir}.")
    xi_sums = xi.groupby("omega")["pi_xi_given_omega"].sum()
    if not np.allclose(xi_sums.to_numpy(), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(
            f"Conditional xi probabilities do not sum to one in {inputs_dir}."
        )

    return probabilities[["omega", "xi", "scenario_probability"]]


def load_stage_two(case_dir: Path) -> pd.DataFrame:
    stage_two = pd.read_csv(case_dir / "stage1_2_results.csv")
    da_columns = [
        column for column in stage_two.columns if column.startswith("P_DA_omega_")
    ]
    if not da_columns:
        raise ValueError(f"No day-ahead columns found in {case_dir}.")

    stage_two_long = stage_two.melt(
        id_vars=["t", "P_PPA"],
        value_vars=da_columns,
        var_name="omega_column",
        value_name="P_DA",
    )
    stage_two_long["omega"] = (
        stage_two_long["omega_column"].str.rsplit("_", n=1).str[-1].astype(int)
    )
    return stage_two_long[["t", "omega", "P_PPA", "P_DA"]]


def lower_tail_cvar(
    scenario_profit: np.ndarray,
    probability: np.ndarray,
    alpha: float,
) -> float:
    """Return the lower-tail CVaR of profit used by the Pyomo formulation."""
    losses = -np.asarray(scenario_profit, dtype=float)
    weights = np.asarray(probability, dtype=float)
    candidates = np.unique(losses)
    loss_cvar = min(
        eta + np.sum(weights * np.maximum(losses - eta, 0.0)) / (1.0 - alpha)
        for eta in candidates
    )
    return -float(loss_cvar)


def calculate_case(case: pd.Series) -> dict[str, float | str]:
    case_dir = local_case_directory(case)
    required_files = [
        case_dir / "stage1_2_results.csv",
        case_dir / "stage3_results.csv",
        case_dir / "settlement_period_results.csv",
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing detailed result files:\n" + "\n".join(map(str, missing))
        )

    project_root = project_roots()[str(case["sdp_label"])]
    parameters = load_project_parameters(project_root)
    probabilities = load_scenario_probabilities(project_root)
    stage_two = load_stage_two(case_dir)

    scenario_index = pd.MultiIndex.from_frame(probabilities[["omega", "xi"]])
    scenario_totals = pd.DataFrame(
        0.0,
        index=scenario_index,
        columns=[
            "ppa_revenue",
            "spot_revenue",
            "mfrr_up",
            "mfrr_down",
            "imbalance_settlement",
            "imbalance_penalty",
            "battery_degradation",
        ],
    )

    stage_three_columns = [
        "t",
        "omega",
        "xi",
        "P_mfrr_up",
        "P_mfrr_down",
        "P_discharge",
        "P_charge",
        "imbalance",
        "lambda_mfrr_down",
        "lambda_mfrr_up",
        "lambda_imb",
        "lambda_spot",
    ]
    stage_three_path = case_dir / "stage3_results.csv"

    for chunk in pd.read_csv(
        stage_three_path,
        usecols=stage_three_columns,
        chunksize=CSV_CHUNK_SIZE,
    ):
        chunk = chunk.merge(
            stage_two,
            on=["t", "omega"],
            how="left",
            validate="many_to_one",
        )
        if chunk[["P_PPA", "P_DA"]].isna().any().any():
            raise ValueError(f"Stage-two merge failed for {case_dir}.")

        chunk["ppa_revenue"] = float(case["lambda_PPA"]) * chunk["P_PPA"]
        chunk["spot_revenue"] = chunk["lambda_spot"] * chunk["P_DA"]
        chunk["mfrr_up"] = chunk["lambda_mfrr_up"] * chunk["P_mfrr_up"]
        chunk["mfrr_down"] = -chunk["lambda_mfrr_down"] * chunk["P_mfrr_down"]
        chunk["imbalance_settlement"] = (
            chunk["lambda_imb"] * chunk["imbalance"]
        )
        chunk["imbalance_penalty"] = (
            -parameters["c_imb"] * chunk["imbalance"].abs()
        )
        chunk["battery_degradation"] = -parameters[
            "degradation_cost_per_mw"
        ] * (chunk["P_discharge"] + chunk["P_charge"])

        grouped = chunk.groupby(["omega", "xi"], sort=False)[
            scenario_totals.columns
        ].sum()
        scenario_totals.loc[grouped.index, grouped.columns] += grouped

    shortfall = pd.read_csv(
        case_dir / "settlement_period_results.csv",
        usecols=["omega", "xi", "settlement_shortfall"],
    )
    shortfall_by_scenario = shortfall.groupby(["omega", "xi"])[
        "settlement_shortfall"
    ].sum()
    scenario_totals["ppa_shortfall_cost"] = (
        -float(case["c_ppa"])
        * shortfall_by_scenario.reindex(scenario_totals.index, fill_value=0.0)
    )

    scenario_totals = scenario_totals.reset_index().merge(
        probabilities,
        on=["omega", "xi"],
        validate="one_to_one",
    )
    scenario_totals["scenario_profit"] = scenario_totals[COMPONENT_KEYS].sum(
        axis=1
    )

    expected_components = {
        key: float(
            (scenario_totals[key] * scenario_totals["scenario_probability"]).sum()
        )
        for key in COMPONENT_KEYS
    }
    expected_profit = float(sum(expected_components.values()))
    cvar_profit = lower_tail_cvar(
        scenario_totals["scenario_profit"].to_numpy(),
        scenario_totals["scenario_probability"].to_numpy(),
        parameters["alpha"],
    )
    reconstructed_objective = (
        (1.0 - parameters["beta"]) * expected_profit
        + parameters["beta"] * cvar_profit
    )
    reported_objective = float(case["objective_value"])
    objective_difference = reconstructed_objective - reported_objective
    tolerance = max(
        OBJECTIVE_AUDIT_ABS_TOL_EUR,
        OBJECTIVE_AUDIT_REL_TOL * abs(reported_objective),
    )
    if abs(objective_difference) > tolerance:
        raise ValueError(
            "Profit decomposition does not reproduce the reported objective.\n"
            f"Case: {case_dir}\n"
            f"Reconstructed: {reconstructed_objective:.6f}\n"
            f"Reported: {reported_objective:.6f}\n"
            f"Difference: {objective_difference:.6f}"
        )

    result: dict[str, float | str] = {
        "settlement_period": str(case["settlement_period"]),
        "P_b_max": float(case["P_b_max"]),
        "E_max": float(case["E_max"]),
        "c_ppa": float(case["c_ppa"]),
        "lambda_PPA": float(case["lambda_PPA"]),
        "sdp_label": str(case["sdp_label"]),
        "S_PPA": float(case["S_PPA"]),
        **expected_components,
        "expected_profit": expected_profit,
        "cvar_profit": cvar_profit,
        "reconstructed_objective": reconstructed_objective,
        "reported_objective": reported_objective,
        "objective_audit_difference": objective_difference,
        "case_dir": str(case_dir),
    }
    return result


def contrasting_text_color(hex_color: str) -> str:
    red, green, blue = (
        int(hex_color[index : index + 2], 16) / 255.0
        for index in (1, 3, 5)
    )
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.57 else "white"


def cache_matches_cases(
    decomposition: pd.DataFrame,
    cases: pd.DataFrame,
) -> bool:
    key_columns = [
        "settlement_period",
        "P_b_max",
        "E_max",
        "c_ppa",
        "lambda_PPA",
        "sdp_label",
    ]
    if not set(key_columns + COMPONENT_KEYS).issubset(decomposition.columns):
        return False

    cached_keys = {
        tuple(row)
        for row in decomposition[key_columns].itertuples(index=False, name=None)
    }
    requested_keys = {
        tuple(row)
        for row in cases[key_columns].itertuples(index=False, name=None)
    }
    return cached_keys == requested_keys


def plot_decomposition_group(group: pd.DataFrame) -> list[Path]:
    period = str(group["settlement_period"].iloc[0])
    power = float(group["P_b_max"].iloc[0])
    energy = float(group["E_max"].iloc[0])
    c_ppa = float(group["c_ppa"].iloc[0])
    ppa_prices = sorted(group["lambda_PPA"].unique())

    positive_totals = group[COMPONENT_KEYS].clip(lower=0).sum(axis=1) / 1e6
    negative_totals = group[COMPONENT_KEYS].clip(upper=0).sum(axis=1) / 1e6
    y_max = max(float(positive_totals.max()), 0.1)
    y_min = min(float(negative_totals.min()), 0.0)
    y_span = y_max - y_min

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 7.8),
        sharex=True,
        sharey=True,
    )
    legend_handles = {}

    for panel_index, (ax, profile) in enumerate(zip(axes.flat, PROFILE_ORDER)):
        panel = (
            group.loc[group["sdp_label"] == profile]
            .set_index("lambda_PPA")
            .reindex(ppa_prices)
        )
        if panel[COMPONENT_KEYS].isna().any().any():
            raise ValueError(f"Incomplete decomposition data for {profile}.")

        x_positions = np.arange(len(ppa_prices), dtype=float)
        positive_bottom = np.zeros(len(ppa_prices))
        negative_bottom = np.zeros(len(ppa_prices))

        for key in COMPONENT_KEYS:
            values = panel[key].to_numpy(dtype=float) / 1e6
            bottom = np.where(values >= 0.0, positive_bottom, negative_bottom)
            bars = ax.bar(
                x_positions,
                values,
                width=0.64,
                bottom=bottom,
                color=COMPONENT_COLORS[key],
                edgecolor="white",
                linewidth=0.35,
                label=COMPONENT_LABELS[key],
            )
            legend_handles.setdefault(key, bars[0])

            for bar, value, base in zip(bars, values, bottom):
                if abs(value) < MIN_SEGMENT_LABEL_MEUR:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=contrasting_text_color(COMPONENT_COLORS[key]),
                )

            positive_bottom += np.where(values >= 0.0, values, 0.0)
            negative_bottom += np.where(values < 0.0, values, 0.0)

        expected_profit = panel["expected_profit"].to_numpy(dtype=float) / 1e6
        label_offset = 0.025 * y_span
        for x_position, positive_top, net_profit in zip(
            x_positions, positive_bottom, expected_profit
        ):
            ax.text(
                x_position,
                positive_top + label_offset,
                f"Net {net_profit:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
            )

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(f"({panel_letter}) {profile}")
        ax.set_xticks(x_positions, [f"{value:g}" for value in ppa_prices])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)

    for ax in axes.flat:
        ax.set_ylim(y_min - 0.06 * y_span, y_max + 0.17 * y_span)

    fig.legend(
        [legend_handles[key] for key in COMPONENT_KEYS],
        [COMPONENT_LABELS[key] for key in COMPONENT_KEYS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(
        "Expected annual profit decomposition\n"
        rf"$c^{{PPA}}={c_ppa:g}$ EUR/MWh, "
        f"{period.capitalize()} PPA deviation settlement, "
        f"{battery_label(power, energy)}",
        y=0.91,
        fontsize=11,
    )
    fig.supxlabel(r"PPA price, $\lambda^{PPA}$ [EUR/MWh]", y=0.025)
    fig.supylabel("Expected annual contribution [MEUR/year]", x=0.018)
    fig.tight_layout(rect=(0.045, 0.065, 1.0, 0.84))

    output_dir = (
        OUTPUT_ROOT
        / f"settlement_{period}"
        / battery_slug(power, energy)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"profit_decomposition_c_ppa_{number_slug(c_ppa)}"
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


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


def plot_grouped_components(group: pd.DataFrame) -> list[Path]:
    """Plot one midterm-style 2x2 grouped chart for a fixed c_ppa case."""
    period = str(group["settlement_period"].iloc[0])
    power = float(group["P_b_max"].iloc[0])
    energy = float(group["E_max"].iloc[0])
    c_ppa = float(group["c_ppa"].iloc[0])
    ppa_prices = sorted(group["lambda_PPA"].unique())

    palette = get_palette(GROUPED_PLOT_PALETTE)
    if len(palette) < len(ppa_prices):
        raise ValueError(
            f"Palette '{GROUPED_PLOT_PALETTE}' has {len(palette)} colors, "
            f"but {len(ppa_prices)} PPA prices were requested."
        )
    price_colors = {
        price: palette[index] for index, price in enumerate(ppa_prices)
    }

    all_values_meur = group[COMPONENT_KEYS].to_numpy(dtype=float) / 1e6
    y_min = min(float(np.nanmin(all_values_meur)), 0.0)
    y_max = max(float(np.nanmax(all_values_meur)), 0.1)
    y_span = y_max - y_min
    upper_margin = 0.16 * y_span
    lower_margin = 0.12 * y_span

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.2, 9.2),
        sharex=True,
        sharey=True,
    )
    x_positions = np.arange(len(COMPONENT_KEYS), dtype=float)
    group_width = 0.78
    bar_width = group_width / len(ppa_prices)

    for panel_index, (ax, profile) in enumerate(zip(axes.flat, PROFILE_ORDER)):
        panel = (
            group.loc[group["sdp_label"] == profile]
            .set_index("lambda_PPA")
            .reindex(ppa_prices)
        )
        if panel[COMPONENT_KEYS].isna().any().any():
            raise ValueError(f"Incomplete decomposition data for {profile}.")

        for price_index, price in enumerate(ppa_prices):
            offset = (
                price_index - (len(ppa_prices) - 1) / 2
            ) * bar_width
            values_eur = panel.loc[price, COMPONENT_KEYS].to_numpy(dtype=float)
            values_meur = values_eur / 1e6
            bars = ax.bar(
                x_positions + offset,
                values_meur,
                width=bar_width * 0.92,
                color=price_colors[price],
                edgecolor="white",
                linewidth=0.35,
                label=rf"$\lambda^{{PPA}}={price:g}$",
            )

            for bar, value_eur, value_meur in zip(
                bars, values_eur, values_meur
            ):
                if abs(value_eur) < GROUPED_MIN_LABEL_EUR:
                    continue
                vertical_offset = 0.012 * y_span
                y_position = (
                    value_meur + vertical_offset
                    if value_meur >= 0
                    else value_meur - vertical_offset
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_position,
                    compact_eur_label(value_eur),
                    ha="center",
                    va="bottom" if value_meur >= 0 else "top",
                    rotation=58,
                    fontsize=6.2,
                    color="black",
                )

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(f"({panel_letter}) {profile}", fontsize=10)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.set_xticks(
            x_positions,
            [COMPONENT_LABELS[key] for key in COMPONENT_KEYS],
            rotation=27,
            ha="right",
        )
        ax.tick_params(axis="x", labelsize=7.2, labelbottom=True)
        ax.tick_params(axis="y", labelsize=8)

    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        frameon=False,
        fontsize=8.5,
        columnspacing=1.4,
        handlelength=1.6,
    )
    for ax in axes.flat:
        ax.set_ylim(y_min - lower_margin, y_max + upper_margin)

    fig.suptitle(
        "Expected annual revenues and costs by PPA price\n"
        rf"$c^{{PPA}}={c_ppa:g}$ EUR/MWh, "
        f"{period.capitalize()} PPA deviation settlement, "
        f"{battery_label(power, energy)}",
        y=0.995,
        fontsize=12,
    )
    fig.supylabel("Expected annual value [MEUR/year]", x=0.018)
    fig.tight_layout(rect=(0.035, 0.025, 1.0, 0.875), h_pad=2.8, w_pad=1.1)

    output_dir = (
        OUTPUT_ROOT
        / "grouped_components"
        / f"settlement_{period}"
        / battery_slug(power, energy)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / (
        f"grouped_profit_components_c_ppa_{number_slug(c_ppa)}"
    )
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def profile_slug(profile: str) -> str:
    return "_".join(profile.lower().replace("-", " ").split())


def plot_individual_grouped_components(group: pd.DataFrame) -> list[Path]:
    """Plot one full-width midterm-style figure per delivery profile."""
    period = str(group["settlement_period"].iloc[0])
    power = float(group["P_b_max"].iloc[0])
    energy = float(group["E_max"].iloc[0])
    c_ppa = float(group["c_ppa"].iloc[0])
    ppa_prices = sorted(group["lambda_PPA"].unique())

    palette = get_palette(GROUPED_PLOT_PALETTE)
    if len(palette) < len(ppa_prices):
        raise ValueError(
            f"Palette '{GROUPED_PLOT_PALETTE}' has {len(palette)} colors, "
            f"but {len(ppa_prices)} PPA prices were requested."
        )
    price_colors = {
        price: palette[index] for index, price in enumerate(ppa_prices)
    }

    # Keep the same vertical scale across the four profile-specific figures.
    all_values_meur = group[COMPONENT_KEYS].to_numpy(dtype=float) / 1e6
    y_min = min(float(np.nanmin(all_values_meur)), 0.0)
    y_max = max(float(np.nanmax(all_values_meur)), 0.1)
    y_span = y_max - y_min
    upper_margin = 0.14 * y_span
    lower_margin = 0.12 * y_span

    x_positions = np.arange(len(COMPONENT_KEYS), dtype=float)
    group_width = 0.78
    bar_width = group_width / len(ppa_prices)
    outputs = []

    output_dir = (
        OUTPUT_ROOT
        / "grouped_components_individual"
        / f"settlement_{period}"
        / battery_slug(power, energy)
        / f"c_ppa_{number_slug(c_ppa)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for profile in PROFILE_ORDER:
        panel = (
            group.loc[group["sdp_label"] == profile]
            .set_index("lambda_PPA")
            .reindex(ppa_prices)
        )
        if panel[COMPONENT_KEYS].isna().any().any():
            raise ValueError(f"Incomplete decomposition data for {profile}.")

        fig, ax = plt.subplots(figsize=(11.8, 6.6))

        for price_index, price in enumerate(ppa_prices):
            offset = (
                price_index - (len(ppa_prices) - 1) / 2
            ) * bar_width
            values_eur = panel.loc[price, COMPONENT_KEYS].to_numpy(dtype=float)
            values_meur = values_eur / 1e6
            bars = ax.bar(
                x_positions + offset,
                values_meur,
                width=bar_width * 0.92,
                color=price_colors[price],
                edgecolor="white",
                linewidth=0.4,
                label=rf"$\lambda^{{PPA}}={price:g}$",
            )

            for bar, value_eur, value_meur in zip(
                bars, values_eur, values_meur
            ):
                if abs(value_eur) < GROUPED_MIN_LABEL_EUR:
                    continue
                vertical_offset = 0.012 * y_span
                y_position = (
                    value_meur + vertical_offset
                    if value_meur >= 0
                    else value_meur - vertical_offset
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_position,
                    compact_eur_label(value_eur),
                    ha="center",
                    va="bottom" if value_meur >= 0 else "top",
                    rotation=52,
                    fontsize=8.5,
                    color="black",
                )

        ax.set_title(
            "Expected annual revenues and costs by PPA price\n"
            f"{profile}, "
            rf"$c^{{PPA}}={c_ppa:g}$ EUR/MWh, "
            f"{period.capitalize()} PPA deviation settlement, "
            f"{battery_label(power, energy)}",
            fontsize=12,
        )
        ax.set_ylabel("Expected annual value [MEUR/year]", fontsize=12)
        ax.set_xticks(
            x_positions,
            [COMPONENT_LABELS[key] for key in COMPONENT_KEYS],
            rotation=27,
            ha="right",
        )
        ax.set_ylim(y_min - lower_margin, y_max + upper_margin)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper right",
            ncol=2,
            frameon=True,
            fontsize=11,
        )
        fig.tight_layout()

        stem = output_dir / (
            f"grouped_profit_components_{profile_slug(profile)}"
            f"_c_ppa_{number_slug(c_ppa)}"
        )
        for suffix in ("pdf", "png"):
            path = stem.with_suffix(f".{suffix}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)

    return outputs


def main() -> None:
    catalog = load_catalog()
    cases = selected_cases(catalog)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data_path = OUTPUT_ROOT / "expected_profit_decomposition_split_imbalance.csv"

    decomposition = None
    if data_path.exists() and not REBUILD_DECOMPOSITION_DATA:
        cached = pd.read_csv(data_path)
        if cache_matches_cases(cached, cases):
            decomposition = cached
            print(f"Loaded audited decomposition data: {data_path}")
        else:
            print("Cached decomposition does not match the selected cases; rebuilding.")

    if decomposition is None:
        rows: list[dict[str, float | str] | None] = [None] * len(cases)

        def run_case(index: int) -> tuple[int, dict[str, float | str]]:
            return index, calculate_case(cases.iloc[index])

        with ThreadPoolExecutor(max_workers=CASE_WORKERS) as executor:
            futures = {
                executor.submit(run_case, index): index
                for index in range(len(cases))
            }
            completed = 0
            for future in as_completed(futures):
                index, result = future.result()
                rows[index] = result
                completed += 1
                case = cases.iloc[index]
                print(
                    f"[{completed}/{len(cases)} completed] "
                    f"{case['sdp_label']}, "
                    f"settlement={case['settlement_period']}, "
                    f"c_ppa={case['c_ppa']:g}, "
                    f"lambda_PPA={case['lambda_PPA']:g}"
                )

        if any(row is None for row in rows):
            raise RuntimeError("One or more decomposition cases did not finish.")

        decomposition = pd.DataFrame(rows)
        decomposition.to_csv(data_path, index=False, float_format="%.15g")
        print(f"Saved decomposition data: {data_path}")

    figure_outputs = []
    for _, group in decomposition.groupby(
        ["settlement_period", "P_b_max", "E_max", "c_ppa"],
        sort=True,
    ):
        if GENERATE_INDIVIDUAL_GROUPED_FIGURES:
            figure_outputs.extend(plot_individual_grouped_components(group))
        if GENERATE_GROUPED_2X2_FIGURES:
            figure_outputs.extend(plot_grouped_components(group))
        if GENERATE_STACKED_FIGURES:
            figure_outputs.extend(plot_decomposition_group(group))

    maximum_audit_difference = decomposition[
        "objective_audit_difference"
    ].abs().max()
    print(
        "Maximum reconstructed-objective difference: "
        f"{maximum_audit_difference:.6g} EUR"
    )
    print(f"Created {len(figure_outputs) // 2} figures in PDF and PNG:")
    for path in figure_outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
