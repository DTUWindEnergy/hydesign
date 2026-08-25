"""Generate contracted PPA delivery datasets, figures, and appendix tables."""

from __future__ import annotations

from pathlib import Path
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd


POST_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = POST_PROCESSING_ROOT.parent
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
CATALOG_PATH = POST_PROCESSING_ROOT / "common" / "results_catalog.csv"

sys.path.insert(0, str(POST_PROCESSING_ROOT))
sys.path.insert(0, str(THESIS_ROOT / "inputs"))

from common.plotting_style import get_palette
from common.results_catalog import PROJECT_FOLDERS, build_results_catalog
from config import PALETTE_NAME
from sdp import get_delivery_profile


PROFILE_ORDER = list(PROJECT_FOLDERS)
PROFILE_COLORS = dict(zip(PROFILE_ORDER, get_palette(PALETTE_NAME)))
PROFILE_MARKERS = {
    "Generic Industrial": "o",
    "Generic Private": "s",
    "Inverse Solar": "^",
    "Continuous Baseload": "D",
}


def load_catalog() -> pd.DataFrame:
    """Use the cached catalog unless an explicit refresh is requested."""
    refresh = os.environ.get("REFRESH_RESULTS_CATALOG", "0") == "1"
    if refresh or not CATALOG_PATH.exists():
        return build_results_catalog(save=True)

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
    print(f"Loaded cached results catalog with {len(catalog)} cases: {CATALOG_PATH}")
    return catalog


def number_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def battery_slug(power: float, energy: float) -> str:
    return f"battery_p_{number_slug(power)}_e_{number_slug(energy)}"


def battery_label(power: float, energy: float) -> str:
    if math.isclose(float(power), 0.0) and math.isclose(float(energy), 0.0):
        return "No battery"
    return f"{float(power):g} MW/{float(energy):g} MWh battery"


def latex_battery_label(power: float, energy: float) -> str:
    if math.isclose(float(power), 0.0) and math.isclose(float(energy), 0.0):
        return "no battery"
    return (
        f"a ${float(power):g}~\\mathrm{{MW}}/"
        f"{float(energy):g}~\\mathrm{{MWh}}$ battery"
    )


def settlement_label(period: str) -> str:
    return f"{str(period).capitalize()} PPA deviation settlement"


def profile_definitions(catalog: pd.DataFrame) -> dict[str, dict[str, object]]:
    definitions = {}
    for sdp_label in PROFILE_ORDER:
        names = catalog.loc[
            catalog["sdp_label"] == sdp_label, "delivery_profile"
        ].drop_duplicates()
        if len(names) != 1:
            raise ValueError(
                f"Expected one delivery-profile definition for {sdp_label}, "
                f"found {len(names)}."
            )
        profile_name = str(names.iloc[0])
        shape = [float(value) for value in get_delivery_profile(profile_name).hourly_shape]
        if len(shape) != 24:
            raise ValueError(f"{profile_name} must contain exactly 24 hourly values.")
        if min(shape) < 0 or not math.isclose(max(shape), 1.0, abs_tol=1e-9):
            raise ValueError(
                f"{profile_name} must be nonnegative and normalized to a peak of 1."
            )
        definitions[sdp_label] = {
            "delivery_profile": profile_name,
            "shape": shape,
            "equivalent_full_load_hours": sum(shape),
            "average_profile_factor": sum(shape) / 24.0,
        }
    return definitions


def validate_catalog(catalog: pd.DataFrame) -> None:
    key_columns = [
        "settlement_period",
        "P_b_max",
        "E_max",
        "lambda_PPA",
        "c_ppa",
        "sdp_label",
    ]
    duplicates = catalog.duplicated(key_columns, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate contracted-delivery cases found:\n"
            + catalog.loc[duplicates, key_columns].to_string(index=False)
        )

    for group_key, group in catalog.groupby(
        ["settlement_period", "P_b_max", "E_max"], sort=True
    ):
        lambdas = sorted(group["lambda_PPA"].unique())
        penalties = sorted(group["c_ppa"].unique())
        expected = len(PROFILE_ORDER) * len(lambdas) * len(penalties)
        if len(group) != expected:
            raise ValueError(
                f"Incomplete case group {group_key}: expected {expected} rows, "
                f"found {len(group)}."
            )
        expected_profiles = set(PROFILE_ORDER)
        for case_key, case in group.groupby(["lambda_PPA", "c_ppa"], sort=True):
            actual_profiles = set(case["sdp_label"])
            if actual_profiles != expected_profiles:
                raise ValueError(
                    f"Case {group_key + case_key} has profiles {actual_profiles}; "
                    f"expected {expected_profiles}."
                )


def build_case_dataset(
    catalog: pd.DataFrame,
    definitions: dict[str, dict[str, object]],
) -> pd.DataFrame:
    case_data = catalog.copy()
    case_data["profile_full_load_hours"] = case_data["sdp_label"].map(
        {
            label: definition["equivalent_full_load_hours"]
            for label, definition in definitions.items()
        }
    )
    case_data["average_profile_factor"] = case_data["sdp_label"].map(
        {
            label: definition["average_profile_factor"]
            for label, definition in definitions.items()
        }
    )
    case_data["contracted_energy_mwh_per_day"] = (
        case_data["S_PPA"] * case_data["profile_full_load_hours"]
    )
    case_data["battery_duration_h"] = case_data.apply(
        lambda row: (
            float(row["E_max"]) / float(row["P_b_max"])
            if float(row["P_b_max"]) > 0
            else 0.0
        ),
        axis=1,
    )
    case_data["_profile_order"] = case_data["sdp_label"].map(
        {label: index for index, label in enumerate(PROFILE_ORDER)}
    )
    case_data = case_data.sort_values(
        [
            "settlement_period",
            "P_b_max",
            "E_max",
            "lambda_PPA",
            "c_ppa",
            "_profile_order",
        ]
    ).drop(columns="_profile_order")
    keep_columns = [
        "settlement_period",
        "P_b_max",
        "E_max",
        "battery_duration_h",
        "lambda_PPA",
        "c_ppa",
        "sdp_label",
        "delivery_profile",
        "S_PPA",
        "profile_full_load_hours",
        "average_profile_factor",
        "contracted_energy_mwh_per_day",
        "objective_value",
        "case_dir",
    ]
    return case_data[keep_columns].reset_index(drop=True)


def build_hourly_dataset(
    case_data: pd.DataFrame,
    definitions: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for case in case_data.to_dict(orient="records"):
        shape = definitions[str(case["sdp_label"])]["shape"]
        for hour, profile_factor in enumerate(shape):
            row = dict(case)
            row["hour"] = hour
            row["profile_factor"] = float(profile_factor)
            row["P_PPA"] = float(case["S_PPA"]) * float(profile_factor)
            rows.append(row)
    hourly_data = pd.DataFrame(rows)

    check_columns = [
        "settlement_period",
        "P_b_max",
        "E_max",
        "lambda_PPA",
        "c_ppa",
        "sdp_label",
    ]
    hourly_totals = (
        hourly_data.groupby(check_columns, as_index=False)["P_PPA"]
        .sum()
        .rename(columns={"P_PPA": "hourly_sum"})
    )
    expected = case_data[check_columns + ["contracted_energy_mwh_per_day"]]
    check = hourly_totals.merge(expected, on=check_columns, validate="one_to_one")
    if not (
        (check["hourly_sum"] - check["contracted_energy_mwh_per_day"]).abs()
        < 1e-8
    ).all():
        raise ValueError("Hourly contracted delivery does not reproduce daily energy.")
    return hourly_data


def write_datasets(
    case_data: pd.DataFrame,
    hourly_data: pd.DataFrame,
    definitions: dict[str, dict[str, object]],
) -> list[Path]:
    data_dir = OUTPUT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    case_path = data_dir / "contracted_ppa_delivery_cases.csv"
    hourly_path = data_dir / "contracted_ppa_delivery_hourly.csv"
    metrics_path = data_dir / "delivery_profile_metrics.csv"
    case_data.to_csv(case_path, index=False, float_format="%.15g")
    hourly_data.to_csv(hourly_path, index=False, float_format="%.15g")
    metrics = pd.DataFrame(
        [
            {
                "sdp_label": label,
                "delivery_profile": definitions[label]["delivery_profile"],
                "profile_full_load_hours": definitions[label][
                    "equivalent_full_load_hours"
                ],
                "average_profile_factor": definitions[label][
                    "average_profile_factor"
                ],
            }
            for label in PROFILE_ORDER
        ]
    )
    metrics.to_csv(metrics_path, index=False, float_format="%.15g")
    return [case_path, hourly_path, metrics_path]


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_diurnal_cases(hourly_data: pd.DataFrame) -> list[Path]:
    outputs = []
    group_columns = [
        "settlement_period",
        "P_b_max",
        "E_max",
        "lambda_PPA",
        "c_ppa",
    ]
    for group_key, group in hourly_data.groupby(group_columns, sort=True):
        period, power, energy, lambda_ppa, c_ppa = group_key
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        for profile in PROFILE_ORDER:
            profile_data = group.loc[group["sdp_label"] == profile].sort_values("hour")
            closed = pd.concat(
                [
                    profile_data[["hour", "P_PPA"]],
                    pd.DataFrame(
                        {
                            "hour": [24],
                            "P_PPA": [float(profile_data.iloc[0]["P_PPA"])],
                        }
                    ),
                ],
                ignore_index=True,
            )
            ax.plot(
                closed["hour"],
                closed["P_PPA"],
                color=PROFILE_COLORS[profile],
                marker=PROFILE_MARKERS[profile],
                markevery=3,
                linewidth=1.8,
                markersize=4.5,
                label=profile,
            )
        ax.set_xlabel("Hour of the day")
        ax.set_ylabel(r"Hourly contracted PPA delivery, $P_t^{PPA}$ [MW]")
        ax.set_title(
            rf"$\lambda^{{PPA}}={float(lambda_ppa):g}$ EUR/MWh, "
            rf"$c^{{PPA}}={float(c_ppa):g}$ EUR/MWh"
            "\n"
            + f"{settlement_label(str(period))}, {battery_label(power, energy)}"
        )
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlim(0, 24)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.14),
            ncol=2,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        stem = (
            OUTPUT_ROOT
            / "diurnal_profiles"
            / f"settlement_{period}"
            / battery_slug(power, energy)
            / f"lambda_ppa_{number_slug(lambda_ppa)}"
            / f"c_ppa_{number_slug(c_ppa)}"
        )
        outputs.extend(save_figure(fig, stem))
    return outputs


def plot_daily_energy(case_data: pd.DataFrame) -> list[Path]:
    outputs = []
    for group_key, group in case_data.groupby(
        ["settlement_period", "P_b_max", "E_max"], sort=True
    ):
        period, power, energy = group_key
        penalties = sorted(group["c_ppa"].unique())
        if len(penalties) != 3:
            raise ValueError(
                "The daily-energy figure layout expects exactly three PPA penalties."
            )
        fig = plt.figure(figsize=(9.5, 7.8))
        grid = fig.add_gridspec(2, 4)
        first_axis = fig.add_subplot(grid[0, 0:2])
        axes = [
            first_axis,
            fig.add_subplot(grid[0, 2:4], sharex=first_axis, sharey=first_axis),
            fig.add_subplot(grid[1, 1:3], sharex=first_axis, sharey=first_axis),
        ]
        maximum_daily_energy = float(group["contracted_energy_mwh_per_day"].max())
        for ax, c_ppa in zip(axes, penalties):
            panel = group.loc[group["c_ppa"] == c_ppa]
            ppa_prices = sorted(panel["lambda_PPA"].unique())
            x_positions = list(range(len(ppa_prices)))
            bar_width = 0.17
            for profile_index, profile in enumerate(PROFILE_ORDER):
                profile_data = (
                    panel.loc[panel["sdp_label"] == profile]
                    .set_index("lambda_PPA")
                    .reindex(ppa_prices)
                )
                if profile_data["contracted_energy_mwh_per_day"].isna().any():
                    raise ValueError(
                        f"Missing daily-energy result for {profile}, "
                        f"settlement={period}, battery=({power}, {energy}), "
                        f"c_ppa={c_ppa}."
                    )
                offset = (profile_index - (len(PROFILE_ORDER) - 1) / 2) * bar_width
                bar_positions = [position + offset for position in x_positions]
                bars = ax.bar(
                    bar_positions,
                    profile_data["contracted_energy_mwh_per_day"],
                    width=bar_width,
                    color=PROFILE_COLORS[profile],
                    label=profile,
                )
                ax.bar_label(
                    bars,
                    fmt="%.1f",
                    padding=2,
                    rotation=60,
                    fontsize=6.5,
                )
            ax.set_title(rf"$c^{{PPA}}={float(c_ppa):g}$ EUR/MWh")
            ax.set_xticks(x_positions, [f"{float(value):g}" for value in ppa_prices])
            ax.set_ylim(0, maximum_daily_energy * 1.18 if maximum_daily_energy > 0 else 1)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=4,
            frameon=False,
        )
        fig.supxlabel(r"PPA price, $\lambda^{PPA}$ [EUR/MWh]", y=0.03)
        fig.supylabel(r"Daily contracted PPA energy, $E^{PPA}$ [MWh/day]", x=0.02)
        fig.suptitle(
            f"{settlement_label(str(period))}, {battery_label(power, energy)}",
            y=0.90,
        )
        fig.tight_layout(rect=(0.05, 0.07, 1, 0.86))
        stem = (
            OUTPUT_ROOT
            / "daily_energy"
            / f"settlement_{period}"
            / battery_slug(power, energy)
            / "daily_contracted_energy_by_ppa_price"
        )
        outputs.extend(save_figure(fig, stem))
    return outputs


def plot_daily_energy_supply_curves(case_data: pd.DataFrame) -> list[Path]:
    """Plot PPA price against optimal daily contracted energy in a 2x2 layout."""
    outputs = []
    penalty_colors = get_palette(PALETTE_NAME)[:3]
    penalty_markers = ["o", "s", "^"]

    for group_key, group in case_data.groupby(
        ["settlement_period", "P_b_max", "E_max"], sort=True
    ):
        period, power, energy = group_key
        penalties = sorted(group["c_ppa"].unique())
        if len(penalties) != 3:
            raise ValueError(
                "The daily-energy supply-curve figure expects exactly three "
                f"PPA deviation penalties, but found {penalties}."
            )

        maximum_daily_energy = float(group["contracted_energy_mwh_per_day"].max())
        minimum_ppa_price = float(group["lambda_PPA"].min())
        maximum_ppa_price = float(group["lambda_PPA"].max())

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10.5, 7.5),
            sharex=True,
            sharey=True,
        )

        handles = None
        labels = None
        for panel_index, (ax, profile) in enumerate(zip(axes.flat, PROFILE_ORDER)):
            profile_data = group.loc[group["sdp_label"] == profile]

            for c_ppa, color, marker in zip(
                penalties, penalty_colors, penalty_markers
            ):
                curve = profile_data.loc[
                    profile_data["c_ppa"] == c_ppa
                ].sort_values("lambda_PPA")
                if curve.empty:
                    raise ValueError(
                        f"Missing daily-energy supply-curve results for {profile}, "
                        f"settlement={period}, battery=({power}, {energy}), "
                        f"c_ppa={c_ppa}."
                    )

                ax.plot(
                    curve["contracted_energy_mwh_per_day"],
                    curve["lambda_PPA"],
                    color=color,
                    marker=marker,
                    linewidth=1.8,
                    markersize=5.5,
                    label=rf"$c^{{PPA}}={float(c_ppa):g}$ EUR/MWh",
                )

            panel_letter = chr(ord("a") + panel_index)
            ax.set_title(f"({panel_letter}) {profile}")
            ax.set_xlim(
                0.0,
                maximum_daily_energy * 1.03
                if maximum_daily_energy > 0
                else 1.0,
            )
            ax.set_ylim(minimum_ppa_price - 5.0, maximum_ppa_price + 5.0)
            ax.grid(True, alpha=0.25)

            if handles is None:
                handles, labels = ax.get_legend_handles_labels()

        fig.supxlabel(
            r"Optimal daily contracted PPA energy, "
            r"$E^{PPA,day,*}$ [MWh/day]",
            y=0.03,
        )
        fig.supylabel(r"PPA price, $\lambda^{PPA}$ [EUR/MWh]", x=0.02)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
        )
        fig.suptitle(
            f"{settlement_label(str(period))}, {battery_label(power, energy)}",
            y=0.93,
            fontsize=11,
        )
        fig.tight_layout(rect=(0.04, 0.07, 1.0, 0.88))

        stem = (
            OUTPUT_ROOT
            / "daily_energy_supply_curves"
            / f"settlement_{period}"
            / battery_slug(power, energy)
            / "daily_energy_supply_curves_all_sdps_2x2"
        )
        outputs.extend(save_figure(fig, stem))

    return outputs


def latex_escape(value: str) -> str:
    return str(value).replace("&", r"\&").replace("_", r"\_")


def profile_metrics_table(definitions: dict[str, dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        (
            r"\caption{Properties of the normalized shaped delivery profiles. "
            r"The equivalent full-load hours equal the daily contracted energy "
            r"associated with 1 MW of contracted PPA capacity.}"
        ),
        r"\label{tab:ppa_delivery_profile_metrics}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrr@{}}",
        r"\toprule",
        (
            r"Delivery profile & Equivalent full-load hours & "
            r"Average profile factor \\"
        ),
        r"\midrule",
    ]
    for profile in PROFILE_ORDER:
        definition = definitions[profile]
        lines.append(
            f"{latex_escape(profile)} & "
            f"{float(definition['equivalent_full_load_hours']):.6f} & "
            f"{float(definition['average_profile_factor']):.6f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def contracted_energy_table(
    group: pd.DataFrame,
    period: str,
    power: float,
    energy: float,
) -> str:
    pivot = group.pivot(
        index=["lambda_PPA", "c_ppa"],
        columns="sdp_label",
        values="contracted_energy_mwh_per_day",
    ).sort_index()
    label_suffix = re.sub(
        r"[^a-z0-9]+",
        "-",
        f"{period}-p-{number_slug(power)}-e-{number_slug(energy)}".lower(),
    ).strip("-")
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        (
            r"\caption{Daily contracted PPA energy under "
            + latex_escape(str(period).lower())
            + " PPA deviation settlement with "
            + latex_battery_label(power, energy)
            + r". PPA prices and deviation penalties are given in EUR/MWh, "
            + r"and contracted energy is given in MWh/day.}"
        ),
        rf"\label{{tab:contracted-energy-{label_suffix}}}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}rrrrrr@{}}",
        r"\toprule",
        (
            r"$\lambda^{PPA}$ & $c^{PPA}$ & Industrial & Private & "
            r"Inverse solar & Baseload \\"
        ),
        r"\midrule",
    ]
    previous_lambda = None
    for (lambda_ppa, c_ppa), row in pivot.iterrows():
        if previous_lambda is not None and not math.isclose(lambda_ppa, previous_lambda):
            lines.append(r"\addlinespace")
        values = " & ".join(f"{float(row[profile]):.6f}" for profile in PROFILE_ORDER)
        lines.append(
            f"{float(lambda_ppa):g} & {float(c_ppa):g} & {values} \\\\"
        )
        previous_lambda = lambda_ppa
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_tables(
    case_data: pd.DataFrame,
    definitions: dict[str, dict[str, object]],
) -> list[Path]:
    table_dir = OUTPUT_ROOT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    metrics_text = profile_metrics_table(definitions)
    metrics_path = table_dir / "delivery_profile_metrics.tex"
    metrics_path.write_text(metrics_text, encoding="utf-8")
    outputs.append(metrics_path)
    combined_parts = [metrics_text]
    for group_key, group in case_data.groupby(
        ["settlement_period", "P_b_max", "E_max"], sort=True
    ):
        period, power, energy = group_key
        table_text = contracted_energy_table(group, str(period), power, energy)
        table_path = table_dir / (
            f"contracted_energy_settlement_{period}_"
            f"{battery_slug(power, energy)}.tex"
        )
        table_path.write_text(table_text, encoding="utf-8")
        outputs.append(table_path)
        combined_parts.append(table_text)
    combined_path = table_dir / "appendix_contracted_delivery_tables.tex"
    combined_path.write_text("\n\n".join(combined_parts), encoding="utf-8")
    outputs.append(combined_path)
    return outputs


def write_readme(case_data: pd.DataFrame, output_counts: dict[str, int]) -> Path:
    groups = (
        case_data[["settlement_period", "P_b_max", "E_max"]]
        .drop_duplicates()
        .sort_values(["settlement_period", "P_b_max", "E_max"])
    )
    group_lines = [
        f"- {row.settlement_period}: {battery_label(row.P_b_max, row.E_max)}"
        for row in groups.itertuples(index=False)
    ]
    text = "\n".join(
        [
            "# Contracted PPA delivery outputs",
            "",
            "The scripts use the shared results catalog and `inputs/sdp.py`.",
            "",
            "## Commands",
            "",
            "- `python build_datasets.py`: rebuild only the CSV datasets.",
            "- `python plot_diurnal_profiles.py`: rebuild only the diurnal figures.",
            "- `python plot_daily_energy.py`: rebuild only the daily-energy figures.",
            "- `python plot_daily_energy_supply_curves.py`: rebuild only the daily-energy supply curves.",
            "- `python create_tables.py`: rebuild only the LaTeX tables.",
            "- `python run_all.py`: rebuild every output.",
            "",
            "Plot palette selection is in `config.py`.",
            "",
            "## Available case groups",
            "",
            *group_lines,
            "",
            "## Output structure",
            "",
            "- `data/contracted_ppa_delivery_cases.csv`: one row per optimization case.",
            "- `data/contracted_ppa_delivery_hourly.csv`: 24 rows per case with profile factors and hourly contracted delivery.",
            "- `data/delivery_profile_metrics.csv`: profile full-load hours and average-to-peak factors.",
            "- `diurnal_profiles/`: one four-profile figure per settlement, battery, PPA price, and deviation penalty.",
            "- `daily_energy/`: one three-panel figure per settlement and battery configuration.",
            "- `daily_energy_supply_curves/`: one 2x2 supply-curve figure per settlement and battery configuration.",
            "- `tables/`: individual appendix tables and one combined LaTeX file.",
            "",
            "## Generated counts",
            "",
            *[f"- {name}: {count}" for name, count in output_counts.items()],
            "",
        ]
    )
    path = Path(__file__).resolve().parent / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def prepare_analysis_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, object]],
]:
    """Load, validate, and transform the optimization results in memory."""
    catalog = load_catalog()
    validate_catalog(catalog)
    definitions = profile_definitions(catalog)
    case_data = build_case_dataset(catalog, definitions)
    hourly_data = build_hourly_dataset(case_data, definitions)
    return case_data, hourly_data, definitions


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    case_data, hourly_data, definitions = prepare_analysis_data()
    dataset_outputs = write_datasets(case_data, hourly_data, definitions)
    diurnal_outputs = plot_diurnal_cases(hourly_data)
    daily_outputs = plot_daily_energy(case_data)
    daily_supply_outputs = plot_daily_energy_supply_curves(case_data)
    table_outputs = write_tables(case_data, definitions)
    readme = write_readme(
        case_data,
        {
            "optimization cases": len(case_data),
            "hourly dataset rows": len(hourly_data),
            "diurnal figure files (PDF and PNG)": len(diurnal_outputs),
            "daily-energy figure files (PDF and PNG)": len(daily_outputs),
            "daily-energy supply-curve files (PDF and PNG)": len(
                daily_supply_outputs
            ),
            "LaTeX table files": len(table_outputs),
        },
    )
    print(f"Validated {len(case_data)} optimization cases.")
    print(f"Created {len(hourly_data)} hourly contracted-delivery rows.")
    print(f"Created {len(diurnal_outputs) // 2} diurnal figures in PDF and PNG.")
    print(f"Created {len(daily_outputs) // 2} daily-energy figures in PDF and PNG.")
    print(
        f"Created {len(daily_supply_outputs) // 2} daily-energy supply curves "
        "in PDF and PNG."
    )
    print("Created datasets:")
    for path in dataset_outputs:
        print(path)
    print("Created tables:")
    for path in table_outputs:
        print(path)
    print(f"Documentation: {readme}")


if __name__ == "__main__":
    main()
