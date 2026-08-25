"""Export appendix-ready revenue decomposition tables from audited data."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
INPUT_PATH = HERE / "outputs" / "expected_profit_decomposition_split_imbalance.csv"
OUTPUT_ROOT = HERE / "outputs" / "appendix_tables"

sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.results_catalog import PROJECT_FOLDERS
from config import (
    BATTERY_CASES,
    COMPONENTS,
    C_PPA_VALUES,
    LAMBDA_PPA_VALUES,
    SETTLEMENT_PERIODS,
)


PROFILE_ORDER = list(PROJECT_FOLDERS)
COMPONENT_KEYS = [key for key, _, _ in COMPONENTS]
EXPECTED_COLUMNS = [
    "settlement_period",
    "P_b_max",
    "E_max",
    "c_ppa",
    "lambda_PPA",
    "sdp_label",
    *COMPONENT_KEYS,
    "expected_profit",
]


def number_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def battery_slug(power: float, energy: float) -> str:
    return f"battery_p_{number_slug(power)}_e_{number_slug(energy)}"


def load_selected_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing audited decomposition data: {INPUT_PATH}\n"
            "Run plot_profit_decomposition.py first."
        )

    data = pd.read_csv(INPUT_PATH)
    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{INPUT_PATH} is missing required columns: {missing_columns}"
        )

    settlement_periods = {str(period).lower() for period in SETTLEMENT_PERIODS}
    battery_mask = pd.Series(False, index=data.index)
    for power, energy in BATTERY_CASES:
        battery_mask |= np.isclose(data["P_b_max"], power) & np.isclose(
            data["E_max"], energy
        )

    selected = data.loc[
        data["settlement_period"].str.lower().isin(settlement_periods)
        & battery_mask
        & data["c_ppa"].isin(C_PPA_VALUES)
        & data["lambda_PPA"].isin(LAMBDA_PPA_VALUES)
        & data["sdp_label"].isin(PROFILE_ORDER)
    ].copy()

    key_columns = [
        "settlement_period",
        "P_b_max",
        "E_max",
        "c_ppa",
        "lambda_PPA",
        "sdp_label",
    ]
    duplicates = selected.duplicated(key_columns, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate decomposition rows found:\n"
            + selected.loc[duplicates, key_columns].to_string(index=False)
        )

    expected_rows = (
        len(settlement_periods)
        * len(BATTERY_CASES)
        * len(C_PPA_VALUES)
        * len(LAMBDA_PPA_VALUES)
        * len(PROFILE_ORDER)
    )
    if len(selected) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} decomposition rows, found {len(selected)}."
        )

    component_sum = selected[COMPONENT_KEYS].sum(axis=1)
    selected["component_sum"] = component_sum
    selected["profit_reconciliation_difference"] = (
        selected["expected_profit"] - component_sum
    )
    tolerance = np.maximum(
        1.0,
        1e-7 * selected["expected_profit"].abs().to_numpy(),
    )
    failed = (
        selected["profit_reconciliation_difference"].abs().to_numpy()
        > tolerance
    )
    if failed.any():
        audit_columns = [
            *key_columns,
            "expected_profit",
            "component_sum",
            "profit_reconciliation_difference",
        ]
        raise ValueError(
            "Revenue components do not reconcile to expected profit:\n"
            + selected.loc[failed, audit_columns].to_string(index=False)
        )

    profile_order = {
        profile: index for index, profile in enumerate(PROFILE_ORDER)
    }
    selected["_profile_order"] = selected["sdp_label"].map(profile_order)
    return selected.sort_values(
        [
            "settlement_period",
            "P_b_max",
            "E_max",
            "c_ppa",
            "_profile_order",
            "lambda_PPA",
        ]
    ).reset_index(drop=True)


def latex_value_eur_to_keur(value: float) -> str:
    """Format EUR/year as signed kEUR/year with one decimal place."""
    value_keur = float(value) / 1_000.0
    sign = "-" if value_keur < 0 else ""
    number = f"{abs(value_keur):,.1f}".replace(",", r"\,")
    return f"${sign}{number}$"


def latex_table(
    data: pd.DataFrame,
    settlement_period: str,
    power: float,
    energy: float,
    c_ppa: float,
) -> str:
    table = data.loc[
        (data["settlement_period"].str.lower() == settlement_period.lower())
        & np.isclose(data["P_b_max"], power)
        & np.isclose(data["E_max"], energy)
        & np.isclose(data["c_ppa"], c_ppa)
    ]
    expected_rows = len(PROFILE_ORDER) * len(LAMBDA_PPA_VALUES)
    if len(table) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for c_ppa={c_ppa:g}, "
            f"found {len(table)}."
        )

    label = (
        "tab:revenue_decomposition_"
        f"{settlement_period.lower()}_"
        f"{battery_slug(power, energy)}_"
        f"cppa_{number_slug(c_ppa)}"
    )
    caption = (
        "Expected annual operating revenue and cost decomposition for "
        rf"$c^{{PPA}}={c_ppa:g}$ EUR/MWh, a "
        f"{power:g} MW/{energy:g} MWh battery and "
        f"{settlement_period.lower()} PPA deviation settlement. "
        "Revenue components are positive and cost components are negative."
    )

    lines = [
        r"\begin{landscape}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{longtable}{@{}llrrrrrrrrr@{}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}\\",
        r"\toprule",
        (
            r"Profile & $\lambda^{PPA}$ & PPA & DA & Imb.\ sett. & "
            r"Imb.\ pen. & mFRR up & mFRR down & PPA shortfall & "
            r"Battery deg. & Expected profit \\"
        ),
        (
            r" & [EUR/MWh] & \multicolumn{9}{c}{[kEUR/year]} \\"
        ),
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{11}}{{c}}{{\tablename\ \thetable\ -- continued}} \\",
        r"\toprule",
        (
            r"Profile & $\lambda^{PPA}$ & PPA & DA & Imb.\ sett. & "
            r"Imb.\ pen. & mFRR up & mFRR down & PPA shortfall & "
            r"Battery deg. & Expected profit \\"
        ),
        (
            r" & [EUR/MWh] & \multicolumn{9}{c}{[kEUR/year]} \\"
        ),
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{11}{r}{Continued on the next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    previous_profile = None
    for row in table.itertuples(index=False):
        if (
            previous_profile is not None
            and row.sdp_label != previous_profile
        ):
            lines.append(r"\addlinespace")
        previous_profile = row.sdp_label

        values = [
            row.ppa_revenue,
            row.spot_revenue,
            row.imbalance_settlement,
            row.imbalance_penalty,
            row.mfrr_up,
            row.mfrr_down,
            row.ppa_shortfall_cost,
            row.battery_degradation,
            row.expected_profit,
        ]
        lines.append(
            f"{row.sdp_label} & ${row.lambda_PPA:g}$ & "
            + " & ".join(latex_value_eur_to_keur(value) for value in values)
            + r" \\"
        )

    lines.extend(
        [
            r"\end{longtable}",
            r"\end{landscape}",
            "",
        ]
    )
    return "\n".join(lines)


def export_tables(data: pd.DataFrame) -> list[Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_paths = []

    csv_columns = [
        "settlement_period",
        "P_b_max",
        "E_max",
        "c_ppa",
        "lambda_PPA",
        "sdp_label",
        "S_PPA",
        *COMPONENT_KEYS,
        "expected_profit",
        "profit_reconciliation_difference",
    ]
    csv_path = OUTPUT_ROOT / "expected_profit_decomposition_appendix.csv"
    data[csv_columns].to_csv(csv_path, index=False, float_format="%.15g")
    output_paths.append(csv_path)

    combined_tables = []
    for settlement_period in SETTLEMENT_PERIODS:
        for power, energy in BATTERY_CASES:
            for c_ppa in C_PPA_VALUES:
                table_text = latex_table(
                    data,
                    str(settlement_period),
                    float(power),
                    float(energy),
                    float(c_ppa),
                )
                combined_tables.append(table_text)

                table_path = OUTPUT_ROOT / (
                    "expected_profit_decomposition_"
                    f"{str(settlement_period).lower()}_"
                    f"{battery_slug(power, energy)}_"
                    f"c_ppa_{number_slug(c_ppa)}.tex"
                )
                table_path.write_text(table_text, encoding="utf-8")
                output_paths.append(table_path)

    combined_path = OUTPUT_ROOT / "expected_profit_decomposition_tables.tex"
    combined_path.write_text("\n".join(combined_tables), encoding="utf-8")
    output_paths.append(combined_path)
    return output_paths


def main() -> None:
    data = load_selected_data()
    output_paths = export_tables(data)
    maximum_difference = data[
        "profit_reconciliation_difference"
    ].abs().max()

    print(f"Exported {len(data)} decomposition rows.")
    print(
        "Maximum component-to-profit reconciliation difference: "
        f"{maximum_difference:.6g} EUR"
    )
    print("Created:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
