"""Export combined battery-sensitivity profit and contracting tables."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
OUTPUT_ROOT = HERE / "outputs" / "appendix_tables"

sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.results_catalog import PROJECT_FOLDERS  # noqa: E402
from analysis import battery_label, load_catalog, select_contract_cases
from config import BATTERY_CASES, SETTLEMENT_PERIOD  # noqa: E402
from plot_battery_expected_profit import build_expected_profit_data  # noqa: E402


PROFILE_ORDER = list(PROJECT_FOLDERS)


def build_wide_table(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["battery_case"] = [
        battery_label(power, energy)
        for power, energy in zip(data["P_b_max"], data["E_max"])
    ]
    value_columns = ["S_PPA", "contracted_energy_mwh_per_day"]
    wide = data.pivot(
        index=["sdp_label", "lambda_PPA", "c_ppa"],
        columns="battery_case",
        values=value_columns,
    )

    ordered_batteries = [
        battery_label(power, energy) for power, energy in BATTERY_CASES
    ]
    ordered_columns = [
        (metric, battery)
        for battery in ordered_batteries
        for metric in value_columns
    ]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered_columns))
    if wide.isna().any().any():
        raise ValueError("The battery appendix table contains missing cases.")

    wide.columns = [
        f"{battery}__{metric}" for metric, battery in wide.columns
    ]
    wide = wide.reset_index()
    profile_order = {
        profile: index for index, profile in enumerate(PROFILE_ORDER)
    }
    wide["_profile_order"] = wide["sdp_label"].map(profile_order)
    return wide.sort_values(
        ["_profile_order", "lambda_PPA", "c_ppa"]
    ).drop(columns="_profile_order")


def format_triplet(
    expected_profit: float,
    capacity: float,
    daily_energy: float,
) -> str:
    return (
        f"{float(expected_profit) / 1e6:.3f} / "
        f"{float(capacity):.3f} / "
        f"{float(daily_energy):.3f}"
    )


def profile_slug(profile: str) -> str:
    return "_".join(str(profile).lower().replace("-", " ").split())


def latex_profile_table(wide: pd.DataFrame, profile: str) -> str:
    battery_labels = [
        battery_label(power, energy) for power, energy in BATTERY_CASES
    ]
    table = wide.loc[wide["sdp_label"] == profile]
    expected_rows = len(table["lambda_PPA"].unique()) * len(
        table["c_ppa"].unique()
    )
    if len(table) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for {profile}, found {len(table)}."
        )

    label = f"tab:battery_sensitivity_{profile_slug(profile)}"
    caption = (
        "Expected annual profit, optimal contracted PPA capacity and optimal "
        "daily contracted energy for the "
        f"{profile} profile under {SETTLEMENT_PERIOD} PPA deviation "
        r"settlement. Each battery column reports "
        r"expected profit [MEUR/year] / $S^{PPA,*}$ [MW] / "
        r"$E^{PPA,day,*}$ [MWh/day]. Profits exclude the CVaR valuation term."
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}rrrrr@{}}",
        r"\toprule",
        (
            r"$\lambda^{PPA}$ & $c^{PPA}$ & "
            + " & ".join(battery_labels)
            + r" \\"
        ),
        (
            r"\multicolumn{2}{c}{[EUR/MWh]} & "
            + r"\multicolumn{3}{c}"
            + r"{Expected profit [MEUR/year] / $S^{PPA,*}$ [MW] / "
            + r"$E^{PPA,day,*}$ [MWh/day]} \\"
        ),
        r"\midrule",
    ]

    previous_lambda = None
    for _, row in table.iterrows():
        if (
            previous_lambda is not None
            and row["lambda_PPA"] != previous_lambda
        ):
            lines.append(r"\addlinespace")
        previous_lambda = row["lambda_PPA"]

        values = []
        for battery in battery_labels:
            values.append(
                format_triplet(
                    row[f"{battery}__expected_profit"],
                    row[f"{battery}__S_PPA"],
                    row[f"{battery}__contracted_energy_mwh_per_day"],
                )
            )
        lines.append(
            f"{row['lambda_PPA']:g} & {row['c_ppa']:g} & "
            + " & ".join(values)
            + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])
    return "\n".join(lines)


def build_expected_profit_wide(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["battery_case"] = [
        battery_label(power, energy)
        for power, energy in zip(data["P_b_max"], data["E_max"])
    ]
    battery_labels = [
        battery_label(power, energy) for power, energy in BATTERY_CASES
    ]
    wide = data.pivot(
        index=["sdp_label", "lambda_PPA", "c_ppa"],
        columns="battery_case",
        values="expected_profit",
    ).reindex(columns=battery_labels)
    if wide.isna().any().any():
        raise ValueError("The expected-profit appendix table has missing cases.")

    wide = wide.reset_index()
    profile_order = {
        profile: index for index, profile in enumerate(PROFILE_ORDER)
    }
    wide["_profile_order"] = wide["sdp_label"].map(profile_order)
    return wide.sort_values(
        ["_profile_order", "lambda_PPA", "c_ppa"]
    ).drop(columns="_profile_order")


def main() -> None:
    catalog = load_catalog()
    data = select_contract_cases(catalog, profile=None)
    wide = build_wide_table(data)
    expected_profit = build_expected_profit_data(profile=None)
    expected_profit_wide = build_expected_profit_wide(expected_profit)
    battery_labels = [
        battery_label(power, energy) for power, energy in BATTERY_CASES
    ]
    expected_profit_for_merge = expected_profit_wide.rename(
        columns={
            battery: f"{battery}__expected_profit"
            for battery in battery_labels
        }
    )
    combined_wide = wide.merge(
        expected_profit_for_merge,
        on=["sdp_label", "lambda_PPA", "c_ppa"],
        how="inner",
        validate="one_to_one",
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    long_csv = OUTPUT_ROOT / "battery_sensitivity_contracting_long.csv"
    wide_csv = OUTPUT_ROOT / "battery_sensitivity_contracting_wide.csv"
    expected_profit_csv = OUTPUT_ROOT / "battery_expected_profit_wide.csv"
    combined_csv = OUTPUT_ROOT / "battery_sensitivity_combined_wide.csv"
    data.to_csv(long_csv, index=False, float_format="%.15g")
    wide.to_csv(wide_csv, index=False, float_format="%.15g")
    expected_profit_wide.to_csv(
        expected_profit_csv,
        index=False,
        float_format="%.15g",
    )
    combined_wide.to_csv(combined_csv, index=False, float_format="%.15g")

    table_paths = []
    combined_tables = []
    for profile in PROFILE_ORDER:
        table_text = latex_profile_table(combined_wide, profile)
        combined_tables.append(table_text)
        table_path = OUTPUT_ROOT / (
            f"battery_sensitivity_{profile_slug(profile)}.tex"
        )
        table_path.write_text(table_text, encoding="utf-8")
        table_paths.append(table_path)

    combined_path = OUTPUT_ROOT / "battery_sensitivity_contracting_tables.tex"
    combined_path.write_text("\n".join(combined_tables), encoding="utf-8")
    print(f"Exported {len(data)} complete battery-sensitivity cases.")
    print("Created:")
    for path in (
        long_csv,
        wide_csv,
        expected_profit_csv,
        combined_csv,
        *table_paths,
        combined_path,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
